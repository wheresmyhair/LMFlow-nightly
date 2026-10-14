from types import SimpleNamespace

import pytest

from lmflow.args import DatasetArguments
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.tokenization.hf_decoder_model import conversation_tokenize_function
from lmflow.utils.conversation_template.qwen import QWEN3_TEMPLATE


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
    # The static tokenizer cannot render text, so these tests leave the prefix check out.
    return DatasetArguments(
        block_size=block_size,
        disable_group_texts=True,
        train_on_prompt=train_on_prompt,
        conversation_prefix_check=False,
    )


def _tokenize(tokenizer, data_args):
    return conversation_tokenize_function(
        examples={"messages": [[{"role": "user", "content": "question"}]]},
        data_args=data_args,
        tokenizer=tokenizer,
        column_names=["messages"],
        conversation_template="{% generation %}{{ messages }}{% endgeneration %}",
    )


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
