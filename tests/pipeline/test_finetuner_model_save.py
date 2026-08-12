from pathlib import Path
from types import SimpleNamespace

import pytest
from accelerate.commands.config.config_args import load_config_from_file

from lmflow.pipeline.finetuner import _save_finetuned_model

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class _RecordingTrainer:
    def __init__(self, *, is_fsdp_enabled=False):
        self.is_fsdp_enabled = is_fsdp_enabled
        self.save_calls = 0

    def save_model(self):
        self.save_calls += 1


class _RecordingModel:
    def __init__(self):
        self.merge_calls = 0
        self.save_calls = []

    def merge_lora_weights(self):
        self.merge_calls += 1

    def save(self, output_dir, *, save_full_model):
        self.save_calls.append((output_dir, save_full_model))


@pytest.mark.parametrize(
    ("use_lora", "is_fsdp_enabled", "expected_trainer_saves", "expected_model_saves"),
    [
        (False, False, 1, []),
        (True, False, 0, [("adapter-output", False)]),
        (True, True, 1, []),
    ],
)
def test_save_finetuned_model_selects_distributed_safe_export(
    use_lora,
    is_fsdp_enabled,
    expected_trainer_saves,
    expected_model_saves,
):
    trainer = _RecordingTrainer(is_fsdp_enabled=is_fsdp_enabled)
    model = _RecordingModel()

    _save_finetuned_model(
        trainer,
        model,
        SimpleNamespace(use_lora=use_lora, save_aggregated_lora=False),
        SimpleNamespace(output_dir="adapter-output"),
    )

    assert trainer.save_calls == expected_trainer_saves
    assert model.merge_calls == 0
    assert model.save_calls == expected_model_saves


def test_save_finetuned_model_merges_aggregated_lora_before_export():
    trainer = _RecordingTrainer(is_fsdp_enabled=False)
    model = _RecordingModel()

    _save_finetuned_model(
        trainer,
        model,
        SimpleNamespace(use_lora=True, save_aggregated_lora=True),
        SimpleNamespace(output_dir="merged-output"),
    )

    assert trainer.save_calls == 0
    assert model.merge_calls == 1
    assert model.save_calls == [("merged-output", True)]


def test_save_finetuned_model_rejects_aggregated_lora_under_fsdp():
    trainer = _RecordingTrainer(is_fsdp_enabled=True)
    model = _RecordingModel()

    with pytest.raises(ValueError, match="not supported under FSDP"):
        _save_finetuned_model(
            trainer,
            model,
            SimpleNamespace(use_lora=True, save_aggregated_lora=True),
            SimpleNamespace(output_dir="merged-output"),
        )

    assert trainer.save_calls == 0
    assert model.merge_calls == 0
    assert model.save_calls == []


def test_fsdp2_config_requests_reloadable_full_state_export():
    config = load_config_from_file(str(REPOSITORY_ROOT / "configs" / "accelerate_fsdp2_config.yaml"))

    assert str(config.distributed_type) == "DistributedType.FSDP"
    assert config.mixed_precision == "bf16"
    assert config.num_processes == 2
    assert config.fsdp_config["fsdp_version"] == 2
    assert config.fsdp_config["fsdp_reshard_after_forward"] is True
    assert config.fsdp_config["fsdp_state_dict_type"] == "FULL_STATE_DICT"
