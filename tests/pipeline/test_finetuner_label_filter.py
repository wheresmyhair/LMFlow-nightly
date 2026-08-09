import logging

import pytest
from datasets import Dataset, IterableDataset

from lmflow.pipeline.finetuner import (
    _filter_samples_without_loss_labels,
    _limit_dataset_samples,
    _prepare_dataset_for_loss,
)


def test_filter_samples_without_loss_labels(caplog):
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2], [3, 4], [5, 6]],
            "attention_mask": [[1, 1], [1, 1], [1, 1]],
            "labels": [[-100, 2], [-100, -100], [5, 6]],
        }
    )

    with caplog.at_level(logging.WARNING):
        filtered = _filter_samples_without_loss_labels(dataset, "eval")

    assert len(filtered) == 2
    assert filtered["input_ids"] == [[1, 2], [5, 6]]
    assert "Dropped 1 eval samples" in caplog.text


def test_filter_samples_without_loss_labels_rejects_empty_split():
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2]],
            "attention_mask": [[1, 1]],
            "labels": [[-100, -100]],
        }
    )

    with pytest.raises(ValueError, match="No train samples contain loss-bearing labels"):
        _filter_samples_without_loss_labels(dataset, "train")


def test_filter_samples_without_loss_labels_is_lazy_for_streaming(caplog):
    consumed = []

    def generate_samples():
        for sample in (
            {"input_ids": [1], "labels": [-100]},
            {"input_ids": [2], "labels": [2]},
            {"input_ids": [3], "labels": [-100, 3]},
        ):
            consumed.append(sample["input_ids"][0])
            yield sample

    dataset = IterableDataset.from_generator(generate_samples)
    with caplog.at_level(logging.WARNING):
        filtered = _filter_samples_without_loss_labels(dataset, "train")

    assert consumed == []
    assert [sample["input_ids"] for sample in filtered] == [[2], [3]]
    assert "Filtering train samples" in caplog.text
    assert "lazily" in caplog.text


def test_streaming_eval_filter_and_max_samples_uses_take():
    def generate_samples():
        for index in range(5):
            labels = [index] if index % 2 else [-100]
            yield {"input_ids": [index], "labels": labels}

    dataset = IterableDataset.from_generator(generate_samples)

    filtered = _filter_samples_without_loss_labels(dataset, "eval")
    limited = _limit_dataset_samples(filtered, 2)

    assert isinstance(limited, IterableDataset)
    assert [sample["input_ids"] for sample in limited] == [[1], [3]]


def test_custom_multi_modal_backend_skips_huggingface_filter():
    class FakeCustomMultiModalDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"input_ids": [index], "labels": [index]}

    dataset = FakeCustomMultiModalDataset()

    prepared = _prepare_dataset_for_loss(dataset, "train", "custom_multi_modal")

    assert prepared is dataset
