import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lmflow.agentic.contracts import build_task_batch
from lmflow.agentic.grpo_controller import run_synchronous_grpo_step
from lmflow.agentic.gsm8k_grpo import (
    GSM8K_GRPO_ROLLOUT_FORMAT,
    GSM8KTokenNativeRollout,
    gsm8k_correctness_rewards,
    gsm8k_grpo_task_from_row,
    gsm8k_protocol_rewards,
    summarize_gsm8k_group_variance,
)
from lmflow.agentic.policy import grpo_loss_from_model


def _canonical_row():
    return {
        "question": "What is 2 + 3?",
        "answer": "Add the numbers. #### 5",
        "instance_id": "openai/gsm8k@revision/main/train/0000000",
        "source_index": 0,
        "row_digest": "a" * 64,
    }


def _response(
    request,
    *,
    prompt_token_ids,
    output_token_ids,
    content="",
    tool_calls=None,
    finish_reason="stop",
    logprobs=None,
    reasoning=None,
):
    request_id = request["model_kwargs"]["extra_body"]["request_id"]
    if logprobs is None:
        logprobs = [-0.1] * len(output_token_ids)
    message = {
        "role": "assistant",
        "content": content,
        "tool_calls": copy.deepcopy(tool_calls or []),
    }
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {
        "message": message,
        "finish_reason": finish_reason,
        "raw_response": {
            "id": f"chatcmpl-{request_id}",
            "prompt_token_ids": list(prompt_token_ids),
            "choices": [
                {
                    "message": copy.deepcopy(message),
                    "finish_reason": finish_reason,
                    "token_ids": list(output_token_ids),
                    "logprobs": {
                        "content": [
                            {"token": f"token_id:{token_id}", "logprob": logprob}
                            for token_id, logprob in zip(output_token_ids, logprobs, strict=True)
                        ]
                    },
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(output_token_ids),
                "total_tokens": len(prompt_token_ids) + len(output_token_ids),
            },
        },
    }


class _TwoRolloutBackend:
    def __init__(self, *, drift=False):
        self.requests = []
        self.drift = drift

    def complete(self, **request):
        self.requests.append(copy.deepcopy(request))
        request_index = len(self.requests) - 1
        rollout_index, call_index = divmod(request_index, 2)
        base = 10 + rollout_index
        if call_index == 0:
            return _response(
                request,
                prompt_token_ids=[1, base],
                output_token_ids=[20 + rollout_index, 30 + rollout_index],
                tool_calls=[
                    {
                        "id": f"call-{rollout_index}",
                        "type": "function",
                        "function": {"name": "calculate", "arguments": '{"expression":"2+3"}'},
                    }
                ],
                finish_reason="tool_calls",
                logprobs=[-0.2, -0.3],
                reasoning="Use the calculator before answering.",
            )

        prior_output = [20 + rollout_index, 30 + rollout_index]
        if self.drift:
            prior_output[-1] = 99
        return _response(
            request,
            prompt_token_ids=[1, base, *prior_output, 40 + rollout_index],
            output_token_ids=[50 + rollout_index, 60 + rollout_index],
            content="Reasoning. #### 5" if rollout_index == 0 else "Reasoning. #### 6",
            finish_reason="stop",
            logprobs=[-0.4, -0.5],
        )


class _InvalidToolBackend:
    def __init__(self):
        self.requests = []

    def complete(self, **request):
        self.requests.append(copy.deepcopy(request))
        return _response(
            request,
            prompt_token_ids=[1, 2],
            output_token_ids=[3, 4],
            tool_calls=[
                {
                    "id": "invalid-call",
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": '{"expression":{"expression":"2+3"}}',
                    },
                }
            ],
            finish_reason="tool_calls",
        )


class _TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 16)
        self.lm_head = torch.nn.Linear(16, 128, bias=False)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(logits=self.lm_head(self.embedding(input_ids)))


class _TinyTrainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        self.data = None

    def train_step(self, data):
        self.data = data
        self.optimizer.zero_grad(set_to_none=True)
        loss = grpo_loss_from_model(self.model, data)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.detach().item(), "selected_tokens": data.batch["loss_mask"].sum().item()}


def _requests(group_size=2):
    task, gold_answer = gsm8k_grpo_task_from_row(_canonical_row())
    requests = build_task_batch([task] * group_size)
    requests.non_tensor_batch.update(
        {
            "group_ids": np.zeros(group_size, dtype=np.int64),
            "rollout_ids": np.arange(group_size, dtype=np.int64),
        }
    )
    requests.meta_info["policy_version"] = "sft64-final"
    return requests, task, gold_answer


def _rollout(backend, *, clock=None):
    kwargs = {} if clock is None else {"clock": clock}
    return GSM8KTokenNativeRollout(
        backend,
        model_name="qwen3-8b-sft64",
        pad_token_id=0,
        model_revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        policy_checkpoint_sha256="b" * 64,
        model_kwargs={
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": True},
            }
        },
        base_seed=7,
        **kwargs,
    )


def test_builds_safe_calculator_task_and_keeps_gold_separate():
    task, gold_answer = gsm8k_grpo_task_from_row(_canonical_row())

    assert gold_answer == "5"
    assert task.task_id == _canonical_row()["instance_id"]
    assert task.environment == {"evaluation_mode": "calculator"}
    assert task.metadata == {
        "source_index": 0,
        "row_digest": "a" * 64,
        "identity_scope": "canonical_source",
    }
    assert "#### 5" not in repr(task.messages)
    assert "answer" not in repr(task.metadata).lower()


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"temperature": 0.6}, "temperature=1.0"),
        ({"top_p": 0.95}, "top_p=1.0"),
        ({"model_kwargs": {"extra_body": {"top_k": 20}}}, "does not reproduce provider sampling"),
    ],
)
def test_rejects_sampling_transforms_not_reproduced_by_current_trainer(overrides, match):
    backend = _TwoRolloutBackend()
    kwargs = {
        "model_name": "qwen3-8b-sft64",
        "pad_token_id": 0,
        "model_revision": "model-revision",
        "tokenizer_revision": "tokenizer-revision",
        "policy_checkpoint_sha256": "b" * 64,
        **overrides,
    }

    with pytest.raises(ValueError, match=match):
        GSM8KTokenNativeRollout(backend, **kwargs)


def test_projects_multi_turn_vllm_tokens_to_existing_dataproto_and_separate_rewards():
    requests, task, gold_answer = _requests()
    backend = _TwoRolloutBackend()
    timestamps = iter([0.0, 0.25, 1.0, 1.25, 2.0, 2.25, 3.0, 3.25])

    rollouts = _rollout(backend, clock=lambda: next(timestamps))(requests)

    assert rollouts.meta_info["policy_version"] == "sft64-final"
    assert rollouts.meta_info["rollout_format"] == GSM8K_GRPO_ROLLOUT_FORMAT
    assert "tasks" not in rollouts.non_tensor_batch
    assert rollouts.non_tensor_batch["task_ids"].tolist() == [task.task_id, task.task_id]
    assert rollouts.non_tensor_batch["termination_reasons"].tolist() == ["completed", "completed"]
    assert rollouts.non_tensor_batch["successful_tool_call_counts"].tolist() == [1, 1]
    assert rollouts.non_tensor_batch["model_call_counts"].tolist() == [2, 2]
    assert rollouts.non_tensor_batch["model_input_token_counts"].tolist() == [7, 7]
    assert rollouts.non_tensor_batch["model_output_token_counts"].tolist() == [4, 4]
    assert rollouts.non_tensor_batch["model_latency_seconds"].tolist() == [0.5, 0.5]
    assert rollouts.non_tensor_batch["model_costs"].tolist() == [0.0, 0.0]
    assert rollouts.non_tensor_batch["selected_token_counts"].tolist() == [2, 2]
    torch.testing.assert_close(
        rollouts.batch["input_ids"],
        torch.tensor(
            [
                [1, 10, 20, 30, 40, 50, 60],
                [1, 11, 21, 31, 41, 51, 61],
            ]
        ),
    )
    torch.testing.assert_close(
        rollouts.batch["loss_mask"],
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            ]
        ),
    )
    torch.testing.assert_close(
        rollouts.batch["old_log_probs"],
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.4, -0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.4, -0.5],
            ]
        ),
    )

    assert backend.requests[0]["model_kwargs"]["tool_choice"]["function"]["name"] == "calculate"
    assert "tool_choice" not in backend.requests[1]["model_kwargs"]
    assert backend.requests[0]["model_kwargs"]["seed"] == 7
    assert backend.requests[1]["model_kwargs"]["seed"] == 8
    assert backend.requests[2]["model_kwargs"]["seed"] == 11
    assert backend.requests[1]["messages"][-2]["reasoning_content"] == "Use the calculator before answering."
    assert all(request["model_kwargs"]["logprobs"] is True for request in backend.requests)
    assert all(request["model_kwargs"]["extra_body"]["return_token_ids"] is True for request in backend.requests)
    assert rollouts.non_tensor_batch["rollout_metadata"][0]["training_anchor_call_index"] == 1
    assert rollouts.non_tensor_batch["rollout_metadata"][0]["calls"] == [
        {
            "request_id": backend.requests[0]["model_kwargs"]["extra_body"]["request_id"],
            "response_id": f"chatcmpl-{backend.requests[0]['model_kwargs']['extra_body']['request_id']}",
            "sampling_seed": 7,
            "finish_reason": "tool_calls",
            "input_tokens": 2,
            "output_tokens": 2,
            "prompt_token_ids": [1, 10],
            "output_token_ids": [20, 30],
            "output_log_probs": [-0.2, -0.3],
            "latency_seconds": 0.25,
            "cost": 0.0,
            "optimized": False,
        },
        {
            "request_id": backend.requests[1]["model_kwargs"]["extra_body"]["request_id"],
            "response_id": f"chatcmpl-{backend.requests[1]['model_kwargs']['extra_body']['request_id']}",
            "sampling_seed": 8,
            "finish_reason": "stop",
            "input_tokens": 5,
            "output_tokens": 2,
            "prompt_token_ids": [1, 10, 20, 30, 40],
            "output_token_ids": [50, 60],
            "output_log_probs": [-0.4, -0.5],
            "latency_seconds": 0.25,
            "cost": 0.0,
            "optimized": True,
        },
    ]

    gold_answers = {task.task_id: gold_answer}
    correctness = gsm8k_correctness_rewards(rollouts, gold_answers)
    protocol = gsm8k_protocol_rewards(rollouts, gold_answers)
    torch.testing.assert_close(correctness, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(protocol, torch.tensor([1.0, 0.0]))
    assert summarize_gsm8k_group_variance(rollouts, protocol) == {
        "rollout/groups": 1.0,
        "rollout/mixed_groups": 1.0,
        "rollout/mixed_group_rate": 1.0,
        "rollout/update_ready_trajectories": 2.0,
        "rollout/update_ready_groups": 1.0,
        "rollout/update_ready_mixed_groups": 1.0,
        "reward/total_mean": 0.5,
    }


def test_anchors_training_to_actual_prompt_when_forced_call_is_rerendered():
    requests, _, _ = _requests(group_size=1)

    rollouts = _rollout(_TwoRolloutBackend(drift=True))(requests)

    torch.testing.assert_close(
        rollouts.batch["input_ids"],
        torch.tensor([[1, 10, 20, 99, 40, 50, 60]]),
    )
    torch.testing.assert_close(
        rollouts.batch["old_log_probs"],
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, -0.4, -0.5]]),
    )
    first_call = rollouts.non_tensor_batch["rollout_metadata"][0]["calls"][0]
    assert first_call["output_token_ids"] == [20, 30]
    assert first_call["output_log_probs"] == [-0.2, -0.3]
    assert first_call["optimized"] is False
    assert rollouts.non_tensor_batch["rollout_metadata"][0]["training_anchor_call_index"] == 1


def test_preserves_invalid_sampled_tool_call_as_terminal_negative_rollout():
    requests, task, gold_answer = _requests(group_size=1)

    rollouts = _rollout(_InvalidToolBackend())(requests)

    assert rollouts.non_tensor_batch["termination_reasons"].tolist() == ["invalid_tool_call"]
    assert rollouts.non_tensor_batch["tool_call_counts"].tolist() == [1]
    assert rollouts.non_tensor_batch["valid_tool_call_counts"].tolist() == [0]
    assert rollouts.non_tensor_batch["selected_token_counts"].tolist() == [0]
    assert rollouts.non_tensor_batch["rollout_metadata"][0]["training_anchor_call_index"] == 0
    assert rollouts.batch["loss_mask"].sum().item() == 0.0
    torch.testing.assert_close(
        gsm8k_protocol_rewards(rollouts, {task.task_id: gold_answer}),
        torch.tensor([0.0]),
    )


def test_connects_real_token_native_rollouts_to_synchronous_policy_update():
    task, gold_answer = gsm8k_grpo_task_from_row(_canonical_row())
    rollout = _rollout(_TwoRolloutBackend())
    model = _TinyCausalLM()
    trainer = _TinyTrainer(model)
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = run_synchronous_grpo_step(
        trainer,
        [task],
        rollout_fn=rollout,
        reward_fn=lambda data: gsm8k_protocol_rewards(data, {task.task_id: gold_answer}),
        group_size=2,
        policy_version="sft64-final",
    )

    assert metrics["rollout/groups"] == 1.0
    assert metrics["rollout/trajectories"] == 2.0
    assert metrics["train/tokens"] == 4.0
    assert math.isfinite(metrics["train/loss"])
    assert any(
        not torch.equal(parameter.detach(), before)
        for parameter, before in zip(model.parameters(), parameters_before, strict=True)
    )
