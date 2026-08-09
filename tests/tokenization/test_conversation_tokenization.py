from types import SimpleNamespace

import pytest
from datasets import Dataset as HFDataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast

from lmflow.args import DatasetArguments
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.tokenization.hf_decoder_model import conversation_tokenize_function
from lmflow.utils.conversation_template.qwen import QWEN2_TEMPLATE, QWEN3_TEMPLATE


class _StaticChatTokenizer:
    model_max_length = 16
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"

    def __init__(self, *, input_ids=None, assistant_masks=None, include_masks=True):
        self.input_ids = input_ids or [10, 11, 12]
        self.assistant_masks = assistant_masks
        self.include_masks = include_masks

    def apply_chat_template(self, **kwargs):
        encoded = {
            "input_ids": list(self.input_ids),
            "attention_mask": [1] * len(self.input_ids),
        }
        if self.include_masks:
            encoded["assistant_masks"] = list(self.assistant_masks)
        return encoded

    def __str__(self):
        return "static-chat-tokenizer"


def _data_args(*, block_size=8, train_on_prompt=False):
    return DatasetArguments(
        block_size=block_size,
        disable_group_texts=True,
        train_on_prompt=train_on_prompt,
    )


def _tokenize(tokenizer, data_args):
    return conversation_tokenize_function(
        examples={"messages": [[{"role": "user", "content": "question"}]]},
        data_args=data_args,
        tokenizer=tokenizer,
        column_names=["messages"],
        conversation_template="{% generation %}{{ messages }}{% endgeneration %}",
    )


def _qwen_test_tokenizer():
    vocab = {token: index for index, token in enumerate(ByteLevel.alphabet())}
    vocab["<pad>"] = len(vocab)
    backend = Tokenizer(BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    backend.decoder = ByteLevelDecoder()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        model_max_length=2048,
    )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


def _find_subsequence_spans(sequence, subsequence):
    return [
        (start, start + len(subsequence))
        for start in range(len(sequence) - len(subsequence) + 1)
        if sequence[start : start + len(subsequence)] == subsequence
    ]


def _tokenize_qwen(
    messages,
    *,
    tokenizer=None,
    system=None,
    tools=None,
    block_size=2048,
    train_on_prompt=False,
):
    examples = {"messages": [messages]}
    if system is not None:
        examples["system"] = [system]
    if tools is not None:
        examples["tools"] = [tools]
    return conversation_tokenize_function(
        examples=examples,
        data_args=_data_args(block_size=block_size, train_on_prompt=train_on_prompt),
        tokenizer=tokenizer or _qwen_test_tokenizer(),
        column_names=list(examples),
        conversation_template=QWEN3_TEMPLATE,
    )


def _assert_token_labels(result, tokenizer, token, expected_labels):
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    spans = _find_subsequence_spans(result["input_ids"][0], token_ids)
    assert spans
    assert all(result["labels"][0][start:end] == expected_labels(token_ids) for start, end in spans)


def test_qwen_message_loss_control_preserves_rendered_prompt():
    tokenizer = _qwen_test_tokenizer()
    user = {"role": "user", "content": "USER_TOKEN"}
    assistant = {"role": "assistant", "content": "ASSISTANT_TOKEN"}
    variants = [
        [user, assistant],
        [user, {**assistant, "loss": None}],
        [user, {**assistant, "loss": True}],
        [user, {**assistant, "loss": False}],
    ]

    rendered = [
        tokenizer.apply_chat_template(messages, chat_template=QWEN3_TEMPLATE, tokenize=False) for messages in variants
    ]

    assert rendered == [rendered[0]] * len(rendered)
    assert rendered[0] == (
        "<|im_start|>user\nUSER_TOKEN<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nASSISTANT_TOKEN<|im_end|>\n"
    )


def test_qwen_message_loss_control_masks_only_selected_assistant_turn():
    tokenizer = _qwen_test_tokenizer()
    messages = [
        {"role": "user", "content": "FIRST_USER"},
        {"role": "assistant", "content": "FIRST_ASSISTANT"},
        {"role": "user", "content": "SECOND_USER"},
        {"role": "assistant", "content": "SECOND_ASSISTANT"},
    ]
    selected_messages = [
        messages[0],
        {**messages[1], "loss": False},
        messages[2],
        {**messages[3], "loss": True},
    ]
    null_messages = [messages[0], {**messages[1], "loss": None}, messages[2], messages[3]]

    baseline = _tokenize_qwen(messages, tokenizer=tokenizer)
    null_control = _tokenize_qwen(null_messages, tokenizer=tokenizer)
    selected = _tokenize_qwen(selected_messages, tokenizer=tokenizer)

    assert null_control == baseline
    assert selected["input_ids"] == baseline["input_ids"]
    assert selected["attention_mask"] == baseline["attention_mask"]
    _assert_token_labels(baseline, tokenizer, "FIRST_ASSISTANT", lambda token_ids: token_ids)
    _assert_token_labels(selected, tokenizer, "FIRST_ASSISTANT", lambda token_ids: [-100] * len(token_ids))
    _assert_token_labels(selected, tokenizer, "SECOND_ASSISTANT", lambda token_ids: token_ids)


def test_qwen_message_loss_control_masks_reasoning_and_tool_call_together():
    tokenizer = _qwen_test_tokenizer()
    messages = [
        {"role": "user", "content": "USER_TOKEN"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "REASON_TOKEN",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {"path": "ARG_TOKEN"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "OBS_TOKEN"},
        {"role": "assistant", "content": "FINAL_TOKEN"},
    ]
    selected_messages = [messages[0], {**messages[1], "loss": False}, messages[2], messages[3]]

    baseline = _tokenize_qwen(messages, tokenizer=tokenizer)
    selected = _tokenize_qwen(selected_messages, tokenizer=tokenizer)

    assert selected["input_ids"] == baseline["input_ids"]
    assert selected["attention_mask"] == baseline["attention_mask"]
    for token in ("REASON_TOKEN", "ARG_TOKEN"):
        _assert_token_labels(baseline, tokenizer, token, lambda token_ids: token_ids)
        _assert_token_labels(selected, tokenizer, token, lambda token_ids: [-100] * len(token_ids))
    _assert_token_labels(selected, tokenizer, "OBS_TOKEN", lambda token_ids: [-100] * len(token_ids))
    _assert_token_labels(selected, tokenizer, "FINAL_TOKEN", lambda token_ids: token_ids)


def test_qwen_message_loss_control_accepts_arrow_filled_nulls():
    dataset = HFDataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "UNCONTROLLED_USER"},
                    {"role": "assistant", "content": "UNCONTROLLED_ASSISTANT"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "FIRST_USER"},
                    {"role": "assistant", "content": "FIRST_ASSISTANT", "loss": False},
                    {"role": "user", "content": "SECOND_USER"},
                    {"role": "assistant", "content": "SECOND_ASSISTANT", "loss": True},
                ]
            },
        ]
    )
    uncontrolled_messages = dataset[0]["messages"]
    messages = dataset[1]["messages"]

    assert [message.get("loss") for message in uncontrolled_messages] == [None, None]
    assert [message.get("loss") for message in messages] == [None, False, None, True]
    tokenizer = _qwen_test_tokenizer()
    uncontrolled = _tokenize_qwen(uncontrolled_messages, tokenizer=tokenizer)
    result = _tokenize_qwen(messages, tokenizer=tokenizer)
    _assert_token_labels(uncontrolled, tokenizer, "UNCONTROLLED_ASSISTANT", lambda token_ids: token_ids)
    _assert_token_labels(result, tokenizer, "FIRST_ASSISTANT", lambda token_ids: [-100] * len(token_ids))
    _assert_token_labels(result, tokenizer, "SECOND_ASSISTANT", lambda token_ids: token_ids)


@pytest.mark.parametrize(
    ("messages", "conversation_template", "train_on_prompt", "error"),
    [
        (
            [{"role": "user", "content": "question", "loss": False}],
            QWEN3_TEMPLATE,
            False,
            "only valid for assistant messages",
        ),
        (
            [{"role": "assistant", "content": "answer", "loss": 1}],
            QWEN3_TEMPLATE,
            False,
            "must be a boolean or null",
        ),
        (
            [{"role": "assistant", "content": "answer", "loss": False}],
            "{% generation %}{{ messages }}{% endgeneration %}",
            False,
            "only by the Qwen3 conversation template",
        ),
        (
            [{"role": "assistant", "content": "answer", "loss": False}],
            QWEN2_TEMPLATE,
            False,
            "only by the Qwen3 conversation template",
        ),
        (
            [{"role": "assistant", "content": "answer", "loss": False}],
            QWEN3_TEMPLATE,
            True,
            "cannot be combined with train_on_prompt=True",
        ),
    ],
)
def test_invalid_message_loss_controls_fail_closed(messages, conversation_template, train_on_prompt, error):
    examples = {"messages": [messages]}
    with pytest.raises(ValueError, match=error):
        conversation_tokenize_function(
            examples=examples,
            data_args=_data_args(train_on_prompt=train_on_prompt),
            tokenizer=_StaticChatTokenizer(assistant_masks=[0, 1, 1]),
            column_names=list(examples),
            conversation_template=conversation_template,
        )


def test_all_assistant_turns_can_be_excluded_from_loss():
    result = _tokenize_qwen(
        [
            {"role": "user", "content": "USER_TOKEN"},
            {"role": "assistant", "content": "ASSISTANT_TOKEN", "loss": False},
        ],
        block_size=256,
    )

    assert set(result["labels"][0]) == {-100}


def test_qwen_tool_trajectory_trains_only_assistant_generation_spans():
    tokenizer = _qwen_test_tokenizer()
    messages = [
        {"role": "user", "content": "USER_TOKEN"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": {"path": "ARG_TOKEN"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "OBS_TOKEN",
        },
        {"role": "assistant", "content": "FINAL_TOKEN"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
    ]
    examples = {
        "system": ["SYSTEM_TOKEN"],
        "messages": [messages],
        "tools": [tools],
    }

    result = conversation_tokenize_function(
        examples=examples,
        data_args=_data_args(block_size=2048),
        tokenizer=tokenizer,
        column_names=list(examples),
        conversation_template=QWEN3_TEMPLATE,
    )

    input_ids = result["input_ids"][0]
    attention_mask = result["attention_mask"][0]
    labels = result["labels"][0]

    for token in ("SYSTEM_TOKEN", "USER_TOKEN", "OBS_TOKEN"):
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        spans = _find_subsequence_spans(input_ids, token_ids)
        assert spans
        assert all(labels[start:end] == [-100] * len(token_ids) for start, end in spans)

    for token in ("ARG_TOKEN", "FINAL_TOKEN"):
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        spans = _find_subsequence_spans(input_ids, token_ids)
        assert spans
        assert all(labels[start:end] == token_ids for start, end in spans)

    padding_positions = [index for index, attention in enumerate(attention_mask) if attention == 0]
    assert padding_positions
    assert all(labels[index] == -100 for index in padding_positions)

    repeated = conversation_tokenize_function(
        examples=examples,
        data_args=_data_args(block_size=2048),
        tokenizer=tokenizer,
        column_names=list(examples),
        conversation_template=QWEN3_TEMPLATE,
    )
    assert repeated == result


@pytest.mark.parametrize(
    ("tokenizer", "error", "message"),
    [
        (
            _StaticChatTokenizer(include_masks=False),
            RuntimeError,
            "requires `assistant_masks`",
        ),
        (
            _StaticChatTokenizer(assistant_masks=[0, 2, 1]),
            ValueError,
            "only 0/1 values",
        ),
        (
            _StaticChatTokenizer(assistant_masks=[0, 1]),
            ValueError,
            "must align with `input_ids`",
        ),
    ],
)
def test_invalid_assistant_masks_fail_closed(tokenizer, error, message):
    with pytest.raises(error, match=message):
        _tokenize(tokenizer, _data_args())


def test_train_on_prompt_does_not_require_assistant_masks():
    result = _tokenize(
        _StaticChatTokenizer(include_masks=False),
        _data_args(block_size=5, train_on_prompt=True),
    )

    assert result == {
        "input_ids": [[10, 11, 12, 0, 0]],
        "attention_mask": [[1, 1, 1, 0, 0]],
        "labels": [[10, 11, 12, -100, -100]],
    }


def test_zero_assistant_mask_produces_fully_masked_labels():
    result = _tokenize(
        _StaticChatTokenizer(assistant_masks=[0, 0, 0]),
        _data_args(block_size=5),
    )

    assert result["labels"] == [[-100, -100, -100, -100, -100]]


def test_truncation_that_removes_all_assistant_tokens_produces_fully_masked_sample():
    tokenizer = _StaticChatTokenizer(
        input_ids=[10, 11, 12, 13],
        assistant_masks=[0, 0, 0, 1],
    )

    result = _tokenize(tokenizer, _data_args(block_size=3))

    assert result == {
        "input_ids": [[10, 11, 12]],
        "attention_mask": [[1, 1, 1]],
        "labels": [[-100, -100, -100]],
    }


class _FingerprintDataset:
    def __init__(self, data_args):
        self.data_args = data_args
        self.backend_dataset = SimpleNamespace(features={"messages": object()})
        self.new_fingerprint = None

    def get_backend(self):
        return "huggingface"

    def get_type(self):
        return "conversation"

    def get_backend_dataset(self):
        return self.backend_dataset

    def get_data_args(self):
        return self.data_args

    def get_fingerprint(self):
        return "raw-dataset-fingerprint"

    def map(self, function, **kwargs):
        self.new_fingerprint = kwargs["new_fingerprint"]
        return self


def _conversation_fingerprint(train_on_prompt, *, truncation_side="right"):
    model = object.__new__(HFDecoderModel)
    model.model_args = SimpleNamespace(use_lora=False)
    model.tokenizer = _StaticChatTokenizer(assistant_masks=[0, 1, 1])
    model.tokenizer.chat_template = QWEN3_TEMPLATE
    model.tokenizer.truncation_side = truncation_side
    dataset = _FingerprintDataset(
        DatasetArguments(
            block_size=16,
            conversation_template="qwen3",
            train_on_prompt=train_on_prompt,
        )
    )

    model.tokenize(dataset)
    return dataset.new_fingerprint


def test_conversation_cache_fingerprint_includes_train_on_prompt():
    assert _conversation_fingerprint(False) != _conversation_fingerprint(True)


def test_conversation_cache_fingerprint_includes_truncation_side():
    assert _conversation_fingerprint(False, truncation_side="left") != _conversation_fingerprint(False)
