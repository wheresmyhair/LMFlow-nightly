import logging
from contextlib import contextmanager

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
            "input_ids": [[1, 2], [3, 4], [5, 6], [7, 8]],
            "attention_mask": [[1, 1], [1, 1], [1, 1], [1, 1]],
            "labels": [[-100, 2], [-100, -100], [5, 6], [7, -100]],
        }
    )

    with caplog.at_level(logging.WARNING):
        filtered = _filter_samples_without_loss_labels(dataset, "eval")

    assert len(filtered) == 2
    assert filtered["input_ids"] == [[1, 2], [5, 6]]
    assert "Dropped 2 eval samples" in caplog.text


def test_filter_samples_without_loss_labels_ignores_first_label():
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2]],
            "attention_mask": [[1, 1]],
            "labels": [[1, -100]],
        }
    )

    with pytest.raises(ValueError, match="No train samples contain loss-bearing labels"):
        _filter_samples_without_loss_labels(dataset, "train")


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
            {"input_ids": [1, 10], "labels": [-100, -100]},
            {"input_ids": [2, 20], "labels": [-100, 2]},
            {"input_ids": [3, 30], "labels": [-100, 3]},
        ):
            consumed.append(sample["input_ids"][0])
            yield sample

    dataset = IterableDataset.from_generator(generate_samples)
    with caplog.at_level(logging.WARNING):
        filtered = _filter_samples_without_loss_labels(dataset, "train")

    assert consumed == []
    assert [sample["input_ids"] for sample in filtered] == [[2, 20], [3, 30]]
    assert "Filtering train samples" in caplog.text
    assert "lazily" in caplog.text


def test_all_masked_stream_cannot_be_rejected_up_front(caplog):
    def generate_samples():
        yield {"input_ids": [1, 2], "labels": [-100, -100]}
        yield {"input_ids": [3, 4], "labels": [3, -100]}

    dataset = IterableDataset.from_generator(generate_samples)

    with caplog.at_level(logging.WARNING):
        filtered = _filter_samples_without_loss_labels(dataset, "train")

    assert list(filtered) == []
    assert "empty-split check are unavailable for streaming datasets" in caplog.text


def test_streaming_eval_filter_and_max_samples_uses_take():
    def generate_samples():
        for index in range(5):
            labels = [-100, index] if index % 2 else [-100, -100]
            yield {"input_ids": [index, index + 10], "labels": labels}

    dataset = IterableDataset.from_generator(generate_samples)

    filtered = _filter_samples_without_loss_labels(dataset, "eval")
    limited = _limit_dataset_samples(filtered, 2)

    assert isinstance(limited, IterableDataset)
    assert [sample["input_ids"] for sample in limited] == [[1, 11], [3, 13]]


def test_custom_multi_modal_backend_skips_huggingface_filter():
    class FakeCustomMultiModalDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"input_ids": [index], "labels": [index]}

    dataset = FakeCustomMultiModalDataset()

    prepared = _prepare_dataset_for_loss(dataset, "train", "custom_multi_modal")

    assert prepared is dataset


def test_map_filter_runs_in_global_main_process_first_context():
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1], [2]],
            "labels": [[-100], [2]],
        }
    )
    events = []

    @contextmanager
    def main_process_first(*, local, desc):
        events.append(("enter", local, desc))
        yield
        events.append(("exit", local, desc))

    filtered = _prepare_dataset_for_loss(
        dataset,
        "train",
        "huggingface",
        main_process_first,
    )

    assert filtered["input_ids"] == [[2]]
    assert events == [
        ("enter", False, "filtering train samples with loss labels"),
        ("exit", False, "filtering train samples with loss labels"),
    ]
