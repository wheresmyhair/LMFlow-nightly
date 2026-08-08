import logging

import pytest
from datasets import Dataset

from lmflow.pipeline.finetuner import _filter_samples_without_loss_labels


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
