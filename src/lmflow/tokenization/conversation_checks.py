#!/usr/bin/env python
"""Checks on conversation samples rendered with a jinja chat template.

They make sure each supervised assistant message is trained the way the model will run it: only on
what the model generates, in the same context the inference prompt gives it.
"""

import bisect
from typing import Optional, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from lmflow.utils.constants import CONVERSATION_ROLE_NAMES

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

_LOSS_PROBE = [
    {"role": "user", "content": "a"},
    {"role": "assistant", "content": "b"},
    {"role": "user", "content": "c"},
    {"role": "assistant", "content": "d"},
]
_loss_support_cache: dict[tuple[int, str], bool] = {}


def message_loss_enabled(message: dict) -> bool:
    """Whether an assistant message is a training target. A missing or null `loss` means yes."""
    return message.get("loss") is not False


def template_supports_message_loss(tokenizer: Tokenizer, chat_template: str) -> bool:
    """Ask the template, by rendering a probe, whether `loss: false` removes a message from the target."""
    key = (id(tokenizer), chat_template)
    if key not in _loss_support_cache:

        def supervised(messages):
            encoded = tokenizer.apply_chat_template(
                messages, chat_template=chat_template, return_assistant_tokens_mask=True, return_dict=True
            )
            return sum(encoded.get("assistant_masks") or [])

        with_loss = supervised(_LOSS_PROBE)
        probe = [dict(m) for m in _LOSS_PROBE]
        probe[1]["loss"] = False
        without_first = supervised(probe)
        _loss_support_cache[key] = 0 < without_first < with_loss
    return _loss_support_cache[key]


def validate_message_loss_controls(
    messages: list[dict],
    conversation_template,
    train_on_prompt: bool,
    tokenizer: Tokenizer,
) -> None:
    """Validate optional assistant-message `loss` flags before rendering."""
    has_explicit_control = False
    for message_index, message in enumerate(messages):
        loss = message.get("loss")
        if loss is None:
            continue
        if message.get("role") != CONVERSATION_ROLE_NAMES["assistant"]:
            raise ValueError(
                f"Message-level loss is only valid for assistant messages; "
                f"message {message_index} has role {message.get('role')!r}"
            )
        if type(loss) is not bool:
            raise ValueError(
                f"Message-level loss must be a boolean or null; message {message_index} has {type(loss).__name__}"
            )
        has_explicit_control = True

    if not has_explicit_control:
        return
    if train_on_prompt:
        raise ValueError("Message-level loss cannot be combined with train_on_prompt=True")
    if not isinstance(conversation_template, str) or not template_supports_message_loss(
        tokenizer, conversation_template
    ):
        raise ValueError(
            "The conversation template does not support message-level loss: setting `loss: false` on an assistant "
            "message does not remove it from the training target. Use a template that does, e.g. qwen3 or "
            "qwen3_nothink."
        )


def assistant_spans(assistant_masks: list[int]) -> list[tuple[int, int]]:
    """Maximal runs of supervised positions, as [start, end) token index pairs."""
    spans = []
    start = None
    for index, mask in enumerate(assistant_masks):
        if mask and start is None:
            start = index
        elif not mask and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(assistant_masks)))
    return spans


def turn_boundary_cut(spans: list[tuple[int, int]], max_length: int) -> Optional[int]:
    """Length to keep so the sample ends after the last supervised span that fits, or None if none fits."""
    ends = [end for _, end in spans if end <= max_length]
    return ends[-1] if ends else None


class PrefixMismatchError(ValueError):
    pass


_TEMPLATE_HINT = (
    "The template likely rewrites earlier turns (e.g. Qwen3 drops reasoning before the last user message and "
    "adds an empty think block to the last turn). Train and serve with one template that renders history "
    "verbatim (e.g. qwen3_nothink), set inference_conversation_template to the template you serve with, or set "
    "conversation_prefix_check=False."
)


def check_train_inference_prefix(
    tokenizer: Tokenizer,
    conversation: list[dict],
    tools: Optional[list],
    input_ids: list[int],
    assistant_masks: list[int],
    train_template: str,
    inference_template: str,
) -> None:
    """Fail if a supervised assistant message is trained in a context that inference would not give it.

    For each supervised assistant message k:

    - the inference prompt, ``inference_template`` applied to ``conversation[:k]`` with
      ``add_generation_prompt=True``, must tokenize to a prefix of the training tokens, and no token of
      it may be supervised;
    - ``train_template`` applied to ``conversation[:k + 1]`` must be a prefix of the training text, i.e.
      message k is trained as it renders when it is the latest message, the way the model produces it.
      Templates that rewrite earlier turns (Qwen3 drops reasoning before the last user message and adds
      an empty think block to the last turn) fail here.

    Prompts are compared as text, then tokenized only from the last special token before their end:
    special tokens split the text before BPE, so the tokens up to that point are those of the training
    text. This keeps the cost near one rendering per message instead of one tokenization per message.
    """
    supervised = [
        index
        for index, message in enumerate(conversation)
        if message.get("role") == CONVERSATION_ROLE_NAMES["assistant"] and message_loss_enabled(message)
    ]
    spans = assistant_spans(assistant_masks)
    if len(spans) != len(supervised):
        raise PrefixMismatchError(
            f"Cannot match supervised token spans to assistant messages: {len(spans)} spans for "
            f"{len(supervised)} supervised assistant messages. The prefix check needs one generation span per "
            "assistant message; set conversation_prefix_check=False to skip it for this template."
        )
    if not supervised:
        return
    if not getattr(tokenizer, "is_fast", False):
        raise PrefixMismatchError(
            "The train/inference prefix check needs a fast tokenizer (for offset mappings); "
            "set conversation_prefix_check=False to skip it."
        )

    train_text = tokenizer.apply_chat_template(conversation, tools=tools, chat_template=train_template, tokenize=False)
    encoded = tokenizer(train_text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != list(input_ids):
        raise PrefixMismatchError("Re-tokenizing the rendered training text does not reproduce its token ids.")
    offsets = encoded["offset_mapping"]
    starts = [start for start, _ in offsets]
    split_ids = set(tokenizer.added_tokens_decoder)

    for message_index, (span_start, _) in zip(supervised, spans, strict=True):
        prompt = tokenizer.apply_chat_template(
            conversation[:message_index],
            tools=tools,
            chat_template=inference_template,
            add_generation_prompt=True,
            tokenize=False,
        )
        where = f"assistant message {message_index}"
        if not train_text.startswith(prompt):
            pos = next((i for i, (a, b) in enumerate(zip(prompt, train_text, strict=False)) if a != b), len(train_text))
            raise PrefixMismatchError(
                f"Train/inference prefix mismatch at {where}: the inference prompt differs from the training text "
                f"at character {pos} (inference {prompt[pos : pos + 40]!r}, training {train_text[pos : pos + 40]!r}). "
                + _TEMPLATE_HINT
            )
        boundary = bisect.bisect_left(starts, len(prompt))
        if boundary > 0 and offsets[boundary - 1][1] > len(prompt):
            raise PrefixMismatchError(
                f"Train/inference prefix mismatch at {where}: a training token spans the end of the inference "
                f"prompt ({train_text[offsets[boundary - 1][0] : offsets[boundary - 1][1]]!r}), so the model is "
                "trained on a tokenization it never sees at inference."
            )
        last_split = next((i for i in range(boundary - 1, -1, -1) if input_ids[i] in split_ids), -1)
        tail_start = offsets[last_split][1] if last_split >= 0 else 0
        tail = tokenizer(prompt[tail_start:], add_special_tokens=False)["input_ids"]
        if tail != list(input_ids[last_split + 1 : boundary]):
            raise PrefixMismatchError(
                f"Train/inference prefix mismatch at {where}: the inference prompt tokenizes differently from the "
                f"training tokens before it, after character {tail_start}."
            )
        latest = tokenizer.apply_chat_template(
            conversation[: message_index + 1], tools=tools, chat_template=train_template, tokenize=False
        )
        if not train_text.startswith(latest):
            pos = next((i for i, (a, b) in enumerate(zip(latest, train_text, strict=False)) if a != b), len(train_text))
            raise PrefixMismatchError(
                f"Train/inference mismatch at {where}: it renders differently in the full conversation than as the "
                f"latest message, which is how the model produces it (at character {pos}: as latest "
                f"{latest[pos : pos + 40]!r}, in the conversation {train_text[pos : pos + 40]!r}). " + _TEMPLATE_HINT
            )
        if span_start < boundary:
            raise PrefixMismatchError(
                f"Train/inference prefix mismatch at {where}: {boundary - span_start} supervised token(s) belong "
                f"to the inference prompt ({tokenizer.decode(input_ids[span_start:boundary])!r}). The template's "
                "generation span should start after the assistant header."
            )
