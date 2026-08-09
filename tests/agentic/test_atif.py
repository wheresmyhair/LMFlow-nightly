import copy
import json
import math

import pytest
import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

import lmflow.pipeline.finetuner as finetuner_module
from lmflow.agentic import atif_trajectory_to_conversation
from lmflow.args import DatasetArguments, FinetunerArguments, ModelArguments
from lmflow.datasets.dataset import Dataset
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.pipeline.finetuner import Finetuner


def _atif_trajectory():
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "run-123",
        "trajectory_id": "trajectory-456",
        "agent": {
            "name": "minimal-bash-agent",
            "version": "1.0.0",
            "model_name": "offline-test-model",
            "tool_definitions": [
                {
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
            ],
        },
        "steps": [
            {
                "step_id": 1,
                "source": "system",
                "message": "SYSTEM_TOKEN",
            },
            {
                "step_id": 2,
                "source": "user",
                "message": "USER_TOKEN",
            },
            {
                "step_id": 3,
                "source": "agent",
                "message": "",
                "reasoning_content": "REASON_TOKEN",
                "llm_call_count": 1,
                "tool_calls": [
                    {
                        "tool_call_id": "call-bash-1",
                        "function_name": "bash",
                        "arguments": {"command": "printf TARGET_ARG"},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "call-bash-1",
                            "content": "OBS_TOKEN",
                        }
                    ]
                },
            },
            {
                "step_id": 4,
                "source": "agent",
                "message": "FINAL_TOKEN",
                "llm_call_count": 1,
            },
        ],
    }


def _set_nested(value, path, replacement):
    current = value
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = replacement


def _byte_level_tokenizer(model_max_length=2048):
    vocab = {token: index for index, token in enumerate(ByteLevel.alphabet())}
    vocab["<pad>"] = len(vocab)
    backend = Tokenizer(BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    backend.decoder = ByteLevelDecoder()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        model_max_length=model_max_length,
    )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


class _TinyDecoderModel:
    tokenize = HFDecoderModel.tokenize

    def __init__(self, backend_model, tokenizer, model_args):
        self.backend_model = backend_model
        self.tokenizer = tokenizer
        self.model_args = model_args

    def get_backend_model(self):
        return self.backend_model

    def get_tokenizer(self):
        return self.tokenizer

    def get_max_length(self):
        return self.tokenizer.model_max_length

    def save(self, *args, **kwargs):
        return None


def test_converts_atif_tool_trajectory_without_mutating_source():
    trajectory = _atif_trajectory()
    original = copy.deepcopy(trajectory)

    conversation = atif_trajectory_to_conversation(trajectory)

    assert trajectory == original
    assert conversation == {
        "conversation_id": "trajectory-456",
        "system": "SYSTEM_TOKEN",
        "tools": trajectory["agent"]["tool_definitions"],
        "messages": [
            {"role": "user", "content": "USER_TOKEN"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "REASON_TOKEN",
                "tool_calls": [
                    {
                        "id": "call-bash-1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"printf TARGET_ARG"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-bash-1",
                "name": "bash",
                "content": "OBS_TOKEN",
            },
            {"role": "assistant", "content": "FINAL_TOKEN"},
        ],
    }
    assert conversation["tools"] is not trajectory["agent"]["tool_definitions"]


def test_reorders_multi_tool_results_to_match_call_order_and_canonicalizes_arguments():
    trajectory = _atif_trajectory()
    agent_step = trajectory["steps"][2]
    agent_step["tool_calls"].append(
        {
            "tool_call_id": "call-bash-2",
            "function_name": "bash",
            "arguments": {"z": 2, "a": 1},
        }
    )
    agent_step["observation"]["results"].insert(
        0,
        {
            "source_call_id": "call-bash-2",
            "content": "SECOND_OBSERVATION",
        },
    )

    conversation = atif_trajectory_to_conversation(trajectory)

    tool_call_messages = conversation["messages"][1]["tool_calls"]
    assert [call["id"] for call in tool_call_messages] == ["call-bash-1", "call-bash-2"]
    assert tool_call_messages[1]["function"]["arguments"] == '{"a":1,"z":2}'
    tool_results = conversation["messages"][2:4]
    assert [result["tool_call_id"] for result in tool_results] == ["call-bash-1", "call-bash-2"]
    assert [result["content"] for result in tool_results] == ["OBS_TOKEN", "SECOND_OBSERVATION"]


def test_uses_session_id_when_trajectory_id_is_absent():
    trajectory = _atif_trajectory()
    trajectory.pop("trajectory_id")

    conversation = atif_trajectory_to_conversation(trajectory)

    assert conversation["conversation_id"] == "run-123"


@pytest.mark.parametrize(
    ("path", "replacement", "error"),
    [
        (("schema_version",), "ATIF-v1.6", "schema_version"),
        (("continued_trajectory_ref",), "next.json", "continued_trajectory_ref"),
        (("subagent_trajectories",), [_atif_trajectory()], "subagent_trajectories"),
        (("steps", 2, "is_copied_context"), True, "per-step loss control"),
        (("steps", 2, "message"), None, "message must be a string"),
        (("steps", 2, "message"), [{"type": "text", "text": "hello"}], "message must be a string"),
        (("steps", 2, "observation", "results", 0, "source_call_id"), None, "source_call_id"),
        (("steps", 2, "observation", "results", 0, "content"), None, "content must be a string"),
    ],
)
def test_rejects_unsupported_atif_semantics(path, replacement, error):
    trajectory = _atif_trajectory()
    _set_nested(trajectory, path, replacement)

    with pytest.raises(ValueError, match=error):
        atif_trajectory_to_conversation(trajectory)


def test_rejects_context_replacement_system_step():
    trajectory = _atif_trajectory()
    system_step = trajectory["steps"][0]
    system_step["observation"] = {"results": [{"content": "SUMMARY_TOKEN"}]}
    system_step["extra"] = {
        "context_management": {
            "type": "compaction",
            "boundary": "replace",
        }
    }

    with pytest.raises(ValueError, match="context_management is not supported"):
        atif_trajectory_to_conversation(trajectory)


def test_rejects_deterministic_agent_dispatch():
    trajectory = _atif_trajectory()
    agent_step = trajectory["steps"][2]
    agent_step["llm_call_count"] = 0
    agent_step.pop("reasoning_content")

    with pytest.raises(ValueError, match="llm_call_count must be 1"):
        atif_trajectory_to_conversation(trajectory)


@pytest.mark.parametrize("tool_definitions", [None, []])
def test_requires_tool_definitions_for_structured_calls(tool_definitions):
    trajectory = _atif_trajectory()
    trajectory["agent"]["tool_definitions"] = tool_definitions

    with pytest.raises(ValueError, match="references undefined tool 'bash'"):
        atif_trajectory_to_conversation(trajectory)


def test_rejects_mismatched_or_duplicate_tool_definitions():
    trajectory = _atif_trajectory()
    trajectory["agent"]["tool_definitions"][0]["function"]["name"] = "python"
    with pytest.raises(ValueError, match="references undefined tool 'bash'"):
        atif_trajectory_to_conversation(trajectory)

    trajectory = _atif_trajectory()
    tools = trajectory["agent"]["tool_definitions"]
    tools.append(copy.deepcopy(tools[0]))
    with pytest.raises(ValueError, match="function.name duplicates 'bash'"):
        atif_trajectory_to_conversation(trajectory)


def test_requires_every_tool_call_to_have_one_observation():
    trajectory = _atif_trajectory()
    trajectory["steps"][2]["observation"]["results"] = []

    with pytest.raises(ValueError, match="missing results for tool calls: call-bash-1"):
        atif_trajectory_to_conversation(trajectory)


def test_rejects_unknown_or_duplicate_observation_ids():
    trajectory = _atif_trajectory()
    result = trajectory["steps"][2]["observation"]["results"][0]
    result["source_call_id"] = "unknown-call"
    with pytest.raises(ValueError, match="references unknown tool call"):
        atif_trajectory_to_conversation(trajectory)

    trajectory = _atif_trajectory()
    results = trajectory["steps"][2]["observation"]["results"]
    results.append(copy.deepcopy(results[0]))
    with pytest.raises(ValueError, match="duplicates observation"):
        atif_trajectory_to_conversation(trajectory)


def test_requires_tool_call_ids_to_be_unique_across_steps():
    trajectory = _atif_trajectory()
    trajectory["steps"][3]["tool_calls"] = copy.deepcopy(trajectory["steps"][2]["tool_calls"])
    trajectory["steps"][3]["observation"] = copy.deepcopy(trajectory["steps"][2]["observation"])

    with pytest.raises(ValueError, match="tool_call_id duplicates 'call-bash-1'"):
        atif_trajectory_to_conversation(trajectory)


def test_rejects_empty_agent_steps_and_unknown_safety_fields():
    trajectory = _atif_trajectory()
    trajectory["steps"] = trajectory["steps"][:2] + [
        {
            "step_id": 3,
            "source": "agent",
            "message": "",
            "llm_call_count": 1,
        }
    ]
    with pytest.raises(ValueError, match="no trainable agent content"):
        atif_trajectory_to_conversation(trajectory)

    trajectory = _atif_trajectory()
    trajectory["steps"][2]["is_copy_context"] = True
    with pytest.raises(ValueError, match="unsupported fields: 'is_copy_context'"):
        atif_trajectory_to_conversation(trajectory)


def test_requires_a_trainable_agent_step():
    trajectory = _atif_trajectory()
    trajectory["steps"] = trajectory["steps"][:2]

    with pytest.raises(ValueError, match="no trainable agent steps"):
        atif_trajectory_to_conversation(trajectory)


def test_converted_json_completes_tiny_lora_finetuner_update(tmp_path, monkeypatch):
    block_size = 2048
    torch.manual_seed(17)
    conversation = atif_trajectory_to_conversation(_atif_trajectory())
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "type": "conversation",
                "instances": [conversation],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    data_args = DatasetArguments(
        dataset_path=str(dataset_dir),
        block_size=block_size,
        disable_group_texts=True,
        conversation_template="qwen3",
        train_on_prompt=False,
        overwrite_cache=True,
    )
    dataset = Dataset(data_args)
    assert dataset.to_dict()["instances"][0]["conversation_id"] == "trajectory-456"

    tokenizer = _byte_level_tokenizer(model_max_length=block_size)
    model_args = ModelArguments(
        model_name_or_path="offline-tiny-gpt2",
        model_max_length=block_size,
        use_flash_attention=False,
        use_lora=True,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
    )
    backend_model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=block_size,
            n_ctx=block_size,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    backend_model = get_peft_model(
        backend_model,
        LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=["c_attn"],
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            fan_in_fan_out=True,
        ),
    )
    model = _TinyDecoderModel(backend_model, tokenizer, model_args)

    inspection_dataset = Dataset(data_args)
    tokenized = model.tokenize(inspection_dataset).get_backend_dataset()[0]
    supervised_text = tokenizer.decode([label for label in tokenized["labels"] if label != -100])
    assert "REASON_TOKEN" in supervised_text
    assert "TARGET_ARG" in supervised_text
    assert "FINAL_TOKEN" in supervised_text
    assert "SYSTEM_TOKEN" not in supervised_text
    assert "USER_TOKEN" not in supervised_text
    assert "OBS_TOKEN" not in supervised_text

    trainable_before = {
        name: parameter.detach().clone()
        for name, parameter in backend_model.named_parameters()
        if parameter.requires_grad
    }
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in backend_model.named_parameters()
        if not parameter.requires_grad
    }
    assert trainable_before
    assert frozen_before
    assert all("lora_" in name for name in trainable_before)
    assert all("lora_" not in name for name in frozen_before)

    output_dir = tmp_path / "output"
    cpu_arguments = {"use_cpu": True} if "use_cpu" in FinetunerArguments.__dataclass_fields__ else {"no_cuda": True}
    finetuner_args = FinetunerArguments(
        output_dir=str(output_dir),
        do_train=True,
        max_steps=1,
        per_device_train_batch_size=1,
        learning_rate=1e-2,
        lr_scheduler_type="constant",
        optim="adamw_torch",
        logging_strategy="no",
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        dataloader_pin_memory=False,
        seed=17,
        **cpu_arguments,
    )
    monkeypatch.setattr(finetuner_module, "send_example_telemetry", None)
    finetuner = Finetuner(model_args, data_args, finetuner_args)

    finetuner.tune(model, dataset)

    train_metrics = json.loads((output_dir / "train_results.json").read_text(encoding="utf-8"))
    trainer_state = json.loads((output_dir / "trainer_state.json").read_text(encoding="utf-8"))
    assert math.isfinite(train_metrics["train_loss"])
    assert trainer_state["global_step"] == 1
    assert any(
        not torch.equal(parameter.detach(), trainable_before[name])
        for name, parameter in backend_model.named_parameters()
        if name in trainable_before
    )
    assert all(
        torch.equal(parameter.detach(), frozen_before[name])
        for name, parameter in backend_model.named_parameters()
        if name in frozen_before
    )
