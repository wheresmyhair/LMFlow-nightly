import copy
from types import SimpleNamespace

import pytest
from torch.distributed.fsdp import StateDictType

from lmflow.agentic.trl_dpo_trainer import (
    TRLDPOTrainer,
    _paired_conversation_to_preference,
    _prepare_paired_conversation_dataset,
)
from lmflow.datasets.dataset import Dataset


def _tool_definition():
    return {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }


def _paired_conversation():
    prompt_messages = [
        {"role": "user", "content": "Fix the failing test."},
        {
            "role": "assistant",
            "content": "",
            "loss": False,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"pytest -q"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "bash",
            "content": "tests/test_app.py::test_value FAILED",
        },
    ]

    def conversation(final_content):
        return {
            "conversation_id": "trajectory-1",
            "system": "You are a coding agent.",
            "tools": [_tool_definition()],
            "messages": copy.deepcopy(prompt_messages) + [{"role": "assistant", "content": final_content}],
        }

    return {
        "chosen": conversation("Patched the value calculation and the test passes."),
        "rejected": conversation("Changed an unrelated comment."),
    }


def test_converts_only_final_assistant_messages_to_dpo_completions():
    pair = _paired_conversation()
    original = copy.deepcopy(pair)

    preference = _paired_conversation_to_preference(pair)

    assert pair == original
    assert preference["prompt"] == [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the failing test."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"pytest -q"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "bash",
            "content": "tests/test_app.py::test_value FAILED",
        },
    ]
    assert preference["chosen"] == [
        {"role": "assistant", "content": "Patched the value calculation and the test passes."}
    ]
    assert preference["rejected"] == [{"role": "assistant", "content": "Changed an unrelated comment."}]
    assert preference["tools"] == [_tool_definition()]
    assert all("loss" not in message for message in preference["prompt"])


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda pair: pair["rejected"].__setitem__("system", "Use a different system prompt."),
            "same system prompt",
        ),
        (
            lambda pair: pair["rejected"].__setitem__("tools", []),
            "same tool definitions",
        ),
        (
            lambda pair: pair["rejected"]["messages"][2].__setitem__("content", "different observation"),
            "share every message",
        ),
        (
            lambda pair: pair["rejected"]["messages"][-1].__setitem__(
                "content", pair["chosen"]["messages"][-1]["content"]
            ),
            "must differ",
        ),
        (
            lambda pair: pair["chosen"]["messages"][-1].__setitem__("role", "user"),
            "end with an assistant",
        ),
        (
            lambda pair: pair["chosen"]["messages"][-1].__setitem__("loss", False),
            "cannot set loss=false",
        ),
        (
            lambda pair: pair["chosen"]["messages"][0].__setitem__("loss", False),
            "only valid on assistant",
        ),
    ],
)
def test_rejects_pairs_that_do_not_share_one_rendered_prompt(mutate, error):
    pair = _paired_conversation()
    mutate(pair)

    with pytest.raises(ValueError, match=error):
        _paired_conversation_to_preference(pair)


def test_prepares_lmflow_dataset_and_normalizes_arrow_null_loss_fields():
    first_pair = _paired_conversation()
    second_pair = _paired_conversation()
    second_pair["chosen"]["conversation_id"] = "trajectory-2"
    second_pair["rejected"]["conversation_id"] = "trajectory-2"
    for side in ("chosen", "rejected"):
        second_pair[side]["messages"][1].pop("loss")
        second_pair[side]["messages"][-1]["loss"] = True
    dataset = Dataset.create_from_dict(
        {
            "type": "paired_conversation",
            "instances": [first_pair, second_pair],
        }
    )

    prepared = _prepare_paired_conversation_dataset(dataset, "train")

    assert len(prepared) == 2
    assert set(prepared.column_names) == {"prompt", "chosen", "rejected", "tools"}
    raw_messages = dataset.get_backend_dataset()[1]["chosen"]["messages"]
    assert raw_messages[0].get("loss") is None
    assert raw_messages[1].get("loss") is None
    assert raw_messages[-1]["loss"] is True
    assert all("loss" not in message for row in prepared for message in row["prompt"])


def test_rejects_non_paired_conversation_dataset():
    dataset = Dataset.create_from_dict(
        {
            "type": "conversation",
            "instances": [{"messages": [{"role": "user", "content": "Hello"}]}],
        }
    )

    with pytest.raises(ValueError, match="paired_conversation"):
        _prepare_paired_conversation_dataset(dataset, "train")


class _SaveRecorder:
    def __init__(self, *, fsdp_enabled, state_dict_type=None, error=None):
        self.is_fsdp_enabled = fsdp_enabled
        self.fsdp_plugin = _FSDPPlugin(state_dict_type) if fsdp_enabled else None
        self.accelerator = SimpleNamespace(state=SimpleNamespace(fsdp_plugin=self.fsdp_plugin))
        self.error = error
        self.observed_fsdp_configs = []
        self.output_dirs = []

    def save_model(self, output_dir):
        if self.fsdp_plugin is not None:
            self.observed_fsdp_configs.append(
                (
                    self.fsdp_plugin.state_dict_type,
                    self.fsdp_plugin.state_dict_config,
                    self.fsdp_plugin.optim_state_dict_config,
                )
            )
        self.output_dirs.append(output_dir)
        if self.error is not None:
            raise self.error


class _FSDPPlugin:
    def __init__(self, state_dict_type):
        self.state_dict_type = state_dict_type
        self.state_dict_config = "original model state config"
        self.optim_state_dict_config = "original optimizer state config"

    def set_state_dict_type(self, state_dict_type):
        self.state_dict_type = state_dict_type
        self.state_dict_config = "full model state config"
        self.optim_state_dict_config = "full optimizer state config"


def _adapter_with_trainer(trainer):
    adapter = TRLDPOTrainer.__new__(TRLDPOTrainer)
    adapter._trainer = trainer
    return adapter


def test_save_model_temporarily_gathers_a_full_fsdp_state_dict():
    trainer = _SaveRecorder(fsdp_enabled=True, state_dict_type=StateDictType.SHARDED_STATE_DICT)

    _adapter_with_trainer(trainer).save_model("adapter-output")

    assert trainer.output_dirs == ["adapter-output"]
    assert trainer.observed_fsdp_configs == [
        (StateDictType.FULL_STATE_DICT, "full model state config", "full optimizer state config")
    ]
    assert trainer.fsdp_plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT
    assert trainer.fsdp_plugin.state_dict_config == "original model state config"
    assert trainer.fsdp_plugin.optim_state_dict_config == "original optimizer state config"


def test_save_model_restores_fsdp_state_dict_type_after_failure():
    trainer = _SaveRecorder(
        fsdp_enabled=True,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        error=RuntimeError("save failed"),
    )

    with pytest.raises(RuntimeError, match="save failed"):
        _adapter_with_trainer(trainer).save_model("adapter-output")

    assert trainer.fsdp_plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT
    assert trainer.fsdp_plugin.state_dict_config == "original model state config"
    assert trainer.fsdp_plugin.optim_state_dict_config == "original optimizer state config"


@pytest.mark.parametrize("fsdp_enabled", [False, True])
def test_save_model_keeps_non_sharded_paths_unchanged(fsdp_enabled):
    trainer = _SaveRecorder(fsdp_enabled=fsdp_enabled, state_dict_type=StateDictType.FULL_STATE_DICT)

    _adapter_with_trainer(trainer).save_model(None)

    assert trainer.output_dirs == [None]
    expected_configs = (
        [(StateDictType.FULL_STATE_DICT, "original model state config", "original optimizer state config")]
        if fsdp_enabled
        else []
    )
    assert trainer.observed_fsdp_configs == expected_configs
