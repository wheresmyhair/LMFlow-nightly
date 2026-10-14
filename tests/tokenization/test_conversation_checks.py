import pytest
from datasets import Dataset as HFDataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast

from lmflow.args import DatasetArguments
from lmflow.tokenization.conversation_checks import (
    PrefixMismatchError,
    assistant_spans,
    template_supports_message_loss,
    turn_boundary_cut,
)
from lmflow.tokenization.hf_decoder_model import conversation_tokenize_function
from lmflow.utils.conversation_template.qwen import QWEN2_5_TEMPLATE, QWEN3_NOTHINK_TEMPLATE, QWEN3_TEMPLATE

IGNORE = -100


def _tokenizer(*, merge_newlines=False):
    """Byte-level BPE with Qwen's chat markers as special tokens, so they split the text before BPE."""
    vocab = {token: index for index, token in enumerate(ByteLevel.alphabet())}
    merges = []
    if merge_newlines:
        vocab["ĊĊ"] = len(vocab)
        merges.append(("Ċ", "Ċ"))
    backend = Tokenizer(BPE(vocab=vocab, merges=merges))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=not merge_newlines)
    backend.decoder = ByteLevelDecoder()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", model_max_length=4096)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


def _tokenize(messages, template, *, tokenizer=None, system=None, inference_template=None, **data_kwargs):
    examples = {"messages": [messages]}
    if system is not None:
        examples["system"] = [system]
    data_args = DatasetArguments(block_size=data_kwargs.pop("block_size", 4096), disable_group_texts=True)
    for key, value in data_kwargs.items():
        setattr(data_args, key, value)
    tokenizer = tokenizer or _tokenizer()
    result = conversation_tokenize_function(
        examples=examples,
        data_args=data_args,
        tokenizer=tokenizer,
        column_names=list(examples),
        conversation_template=template,
        inference_conversation_template=inference_template,
    )
    return result, tokenizer


def _supervised_text(result, tokenizer):
    labels = result["labels"][0]
    return tokenizer.decode([label for label in labels if label != IGNORE])


AGENT_TURNS = [
    {"role": "user", "content": "Fix the bug."},
    {"role": "assistant", "content": "<function=bash>ls</function>"},
    {"role": "user", "content": "EXECUTION RESULT of [bash]:\na.py"},
    {"role": "assistant", "content": "<function=finish></function>"},
]


@pytest.mark.parametrize("template", [QWEN3_TEMPLATE, QWEN3_NOTHINK_TEMPLATE])
def test_loss_flags_do_not_change_rendered_text(template):
    tokenizer = _tokenizer()
    variants = [{**AGENT_TURNS[1], "loss": loss} for loss in (None, True, False)]
    rendered = {
        tokenizer.apply_chat_template([AGENT_TURNS[0], variant], chat_template=template, tokenize=False)
        for variant in [AGENT_TURNS[1], *variants]
    }
    assert len(rendered) == 1


def test_nothink_renders_history_verbatim_and_trains_generation_through_im_end():
    result, tokenizer = _tokenize(AGENT_TURNS, QWEN3_NOTHINK_TEMPLATE, system="SYS")
    text = tokenizer.decode([t for t, m in zip(result["input_ids"][0], result["attention_mask"][0], strict=True) if m])
    assert text == (
        "<|im_start|>system\nSYS<|im_end|>\n"
        "<|im_start|>user\nFix the bug.<|im_end|>\n"
        "<|im_start|>assistant\n<function=bash>ls</function><|im_end|>\n"
        "<|im_start|>user\nEXECUTION RESULT of [bash]:\na.py<|im_end|>\n"
        "<|im_start|>assistant\n<function=finish></function><|im_end|>\n"
    )
    assert _supervised_text(result, tokenizer) == (
        "<function=bash>ls</function><|im_end|><function=finish></function><|im_end|>"
    )


def test_nothink_generation_prompt_has_no_think_block():
    tokenizer = _tokenizer()
    prompt = tokenizer.apply_chat_template(
        AGENT_TURNS[:3], chat_template=QWEN3_NOTHINK_TEMPLATE, add_generation_prompt=True, tokenize=False
    )
    assert prompt.endswith("a.py<|im_end|>\n<|im_start|>assistant\n")


def test_nothink_rejects_reasoning_content():
    messages = [AGENT_TURNS[0], {**AGENT_TURNS[1], "reasoning_content": "why"}]
    with pytest.raises(Exception, match="reasoning_content"):
        _tokenize(messages, QWEN3_NOTHINK_TEMPLATE)


def test_qwen3_single_turn_trains_think_block_and_answer_not_header():
    result, tokenizer = _tokenize(AGENT_TURNS[:2], QWEN3_TEMPLATE)
    assert _supervised_text(result, tokenizer) == "<think>\n\n</think>\n\n<function=bash>ls</function><|im_end|>"


@pytest.mark.parametrize("template", [QWEN3_TEMPLATE, QWEN3_NOTHINK_TEMPLATE])
def test_loss_false_masks_only_that_turn(template):
    masked = [AGENT_TURNS[0], {**AGENT_TURNS[1], "loss": False}, AGENT_TURNS[2], {**AGENT_TURNS[3], "loss": True}]
    # Qwen3 rewrites the first turn of this conversation, which the prefix check would reject.
    baseline, tokenizer = _tokenize(AGENT_TURNS, template, conversation_prefix_check=False)
    result, _ = _tokenize(masked, template, tokenizer=tokenizer, conversation_prefix_check=False)
    assert result["input_ids"] == baseline["input_ids"]
    assert "ls" not in _supervised_text(result, tokenizer)
    assert "<function=finish></function><|im_end|>" in _supervised_text(result, tokenizer)
    assert "ls" in _supervised_text(baseline, tokenizer)


def test_qwen3_loss_false_masks_reasoning_and_tool_calls_together():
    messages = [
        {"role": "user", "content": "USER"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "REASON",
            "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {"path": "ARG"}}}],
            "loss": False,
        },
        {"role": "tool", "content": "OBS"},
        {"role": "assistant", "content": "FINAL"},
    ]
    result, tokenizer = _tokenize(messages, QWEN3_TEMPLATE)
    supervised = _supervised_text(result, tokenizer)
    assert "REASON" not in supervised and "ARG" not in supervised and "OBS" not in supervised
    assert supervised.endswith("FINAL<|im_end|>")


def test_loss_accepts_arrow_filled_nulls():
    dataset = HFDataset.from_list(
        [{"messages": [AGENT_TURNS[0], {**AGENT_TURNS[1], "loss": False}, AGENT_TURNS[2], AGENT_TURNS[3]]}]
    )
    messages = dataset[0]["messages"]
    assert [message.get("loss") for message in messages] == [None, False, None, None]
    result, tokenizer = _tokenize(messages, QWEN3_NOTHINK_TEMPLATE)
    assert _supervised_text(result, tokenizer) == "<function=finish></function><|im_end|>"


@pytest.mark.parametrize(
    ("messages", "template", "train_on_prompt", "error"),
    [
        ([{"role": "user", "content": "q", "loss": False}], QWEN3_NOTHINK_TEMPLATE, False, "only valid for assistant"),
        ([{"role": "assistant", "content": "a", "loss": 1}], QWEN3_NOTHINK_TEMPLATE, False, "boolean or null"),
        (
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "loss": False}],
            QWEN2_5_TEMPLATE,
            False,
            "does not support message-level loss",
        ),
        ([{"role": "assistant", "content": "a", "loss": False}], QWEN3_NOTHINK_TEMPLATE, True, "train_on_prompt"),
    ],
)
def test_invalid_message_loss_controls_fail_closed(messages, template, train_on_prompt, error):
    with pytest.raises(ValueError, match=error):
        _tokenize(messages, template, train_on_prompt=train_on_prompt)


def test_template_support_for_message_loss_is_probed():
    tokenizer = _tokenizer()
    assert template_supports_message_loss(tokenizer, QWEN3_TEMPLATE)
    assert template_supports_message_loss(tokenizer, QWEN3_NOTHINK_TEMPLATE)
    assert not template_supports_message_loss(tokenizer, QWEN2_5_TEMPLATE)


def test_prefix_check_passes_for_history_preserving_template():
    _tokenize(AGENT_TURNS, QWEN3_NOTHINK_TEMPLATE, system="SYS")
    _tokenize(AGENT_TURNS, QWEN3_NOTHINK_TEMPLATE, system="SYS", inference_template=QWEN3_TEMPLATE)


def test_prefix_check_rejects_qwen3_rewriting_earlier_turns():
    with pytest.raises(PrefixMismatchError, match="renders differently in the full conversation"):
        _tokenize(AGENT_TURNS, QWEN3_TEMPLATE)


def test_prefix_check_rejects_inference_prompt_the_training_text_lacks():
    served = QWEN3_NOTHINK_TEMPLATE.replace(
        "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}",
        "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}",
    )
    assert served != QWEN3_NOTHINK_TEMPLATE
    with pytest.raises(PrefixMismatchError, match="inference prompt differs"):
        _tokenize(AGENT_TURNS, QWEN3_NOTHINK_TEMPLATE, inference_template=served)


def test_prefix_check_rejects_supervised_assistant_header():
    template = (
        "{%- for m in messages %}{%- if m.role == 'assistant' %}"
        "{% generation %}{{- '<|im_start|>assistant\\n' + m.content + '<|im_end|>' }}{% endgeneration %}"
        "{{- '\\n' }}{%- else %}{{- '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endif %}"
        "{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
    )
    with pytest.raises(PrefixMismatchError, match="belong to the inference prompt"):
        _tokenize(AGENT_TURNS[:2], template)


def test_prefix_check_rejects_token_merged_across_prompt_end():
    tokenizer = _tokenizer(merge_newlines=True)
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "\nanswer"}]
    with pytest.raises(PrefixMismatchError, match="spans the end of the inference prompt"):
        _tokenize(messages, QWEN3_NOTHINK_TEMPLATE, tokenizer=tokenizer)


def test_prefix_check_can_be_disabled():
    _tokenize(AGENT_TURNS, QWEN3_TEMPLATE, conversation_prefix_check=False)


def test_assistant_spans_and_turn_boundary_cut():
    spans = assistant_spans([0, 1, 1, 0, 0, 1, 0, 1, 1])
    assert spans == [(1, 3), (5, 6), (7, 9)]
    assert turn_boundary_cut(spans, 8) == 6
    assert turn_boundary_cut(spans, 2) is None


def test_long_samples_are_cut_after_the_last_turn_that_fits():
    full, tokenizer = _tokenize(AGENT_TURNS, QWEN3_NOTHINK_TEMPLATE)
    length = sum(full["attention_mask"][0])
    first_end = assistant_spans([int(label != IGNORE) for label in full["labels"][0]])[0][1]
    result, _ = _tokenize(AGENT_TURNS, QWEN3_NOTHINK_TEMPLATE, tokenizer=tokenizer, block_size=length - 3)
    assert sum(result["attention_mask"][0]) == first_end
    assert result["input_ids"][0][:first_end] == full["input_ids"][0][:first_end]
    assert _supervised_text(result, tokenizer) == "<function=bash>ls</function><|im_end|>"

    token_cut, _ = _tokenize(
        AGENT_TURNS, QWEN3_NOTHINK_TEMPLATE, tokenizer=tokenizer, block_size=length - 3, truncate_at_turn_boundary=False
    )
    assert sum(token_cut["attention_mask"][0]) == length - 3
