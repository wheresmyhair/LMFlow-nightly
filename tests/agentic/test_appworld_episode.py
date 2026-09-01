import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lmflow.agentic.appworld_data_factory import (
    _candidate_class,
    finalize_appworld_candidate,
    summarize_appworld_candidates,
)
from lmflow.agentic.appworld_episode import (
    APPWORLD_PYTHON_TOOL_NAME,
    APPWORLD_REACT_TRAINING_PROJECTION_FORMAT_VERSION,
    project_appworld_conversation_for_react_scaffold,
    project_appworld_messages_for_react_scaffold,
    replay_appworld_episode,
    run_appworld_episode,
)
from lmflow.agentic.appworld_protocol import (
    APPWORLD_CONTEXT_BUDGET_EXHAUSTED,
    APPWORLD_TINY_TASK_IDS,
    AppWorldContextBudgetExhaustedError,
    verify_manifest_digest,
)
from lmflow.agentic.appworld_token_native import (
    AppWorldTokenNativeCompletionRecorder,
    assemble_verified_appworld_token_sequence,
    build_appworld_token_native_audit,
    qwen3_appworld_prompt_token_ids,
    qwen3_appworld_replay_chat_template,
    qwen3_appworld_replay_chat_template_identity,
)
from lmflow.agentic.vllm_token_native import VLLMChatTokenData
from lmflow.datasets.dataset import Dataset


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.responses = [
            "I will probe.\n```python\nfail()\n```",
            "I will recover and finish.\n```python\ncomplete()\n```",
        ]

    def complete(self, *, messages, tools, model_name, model_kwargs):
        self.calls.append(messages)
        content = self.responses.pop(0)
        return {
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
            "raw_response": {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        }


class FakeTracker:
    def __init__(self, success):
        self.success = success

    def to_dict(self, stats_only=False):
        result = {"success": self.success, "difficulty": 1, "num_tests": 1}
        if not stats_only:
            result.update(
                {
                    "passes": [{"requirement": "done", "label": None}] if self.success else [],
                    "failures": [] if self.success else [{"requirement": "done", "trace": "", "label": None}],
                }
            )
        return result


class FakeWorld:
    def __init__(self, root: Path, *, state_value: str = "changed"):
        self.task = SimpleNamespace(
            instruction="Complete the fake task.",
            supervisor=SimpleNamespace(first_name="Ada", last_name="Lovelace"),
            app_descriptions={"fake": "A fake app."},
        )
        self.output_directory = str(root / "output")
        self.output_db_home_path_on_disk = str(root / "output" / "dbs")
        Path(self.output_db_home_path_on_disk).mkdir(parents=True)
        self.requester = SimpleNamespace(request_tracker=SimpleNamespace(requests=[]))
        self.completed = False
        self.state_value = state_value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def execute(self, code):
        if code == "fail()":
            self.requester.request_tracker.requests.append(
                {"method": "post", "url": "/fake/login", "data": {"password": "synthetic"}}
            )
            return "Execution failed. Traceback:\nintentional"
        assert code == "complete()"
        self.requester.request_tracker.requests.append({"method": "post", "url": "/fake/complete", "data": {}})
        Path(self.output_db_home_path_on_disk, "state.json").write_text(self.state_value, encoding="utf-8")
        self.completed = True
        return "Marked complete."

    def task_completed(self):
        return self.completed

    def evaluate(self, suppress_errors=True):
        return FakeTracker(self.completed)


def test_episode_records_failure_recovery_state_and_training_projection(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.load_reference_prompt",
        lambda source: "USER:\nTask: {{ instruction }}",
    )
    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.scaffold_identity",
        lambda source: {"id": "fake-reference", "prompt_sha256": "0" * 64},
    )

    def unexpected_freezegun_configure():
        raise AssertionError("custom world factories must not require AppWorld dependencies")

    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.configure_appworld_freezegun",
        unexpected_freezegun_configure,
    )
    world = FakeWorld(tmp_path)
    backend = FakeBackend()

    def factory(**kwargs):
        return world

    original_appworld_root = os.environ.get("APPWORLD_ROOT")
    try:
        result = run_appworld_episode(
            backend,
            task_id=APPWORLD_TINY_TASK_IDS[0],
            model_name="Qwen/Qwen3-8B",
            model_revision="revision",
            trajectory_id="run:task",
            appworld_root=tmp_path,
            appworld_source=tmp_path,
            experiment_name="test",
            world_factory=factory,
        )
    finally:
        if original_appworld_root is None:
            os.environ.pop("APPWORLD_ROOT", None)
        else:
            os.environ["APPWORLD_ROOT"] = original_appworld_root

    verify_manifest_digest(result.artifact)
    metrics = result.artifact["metrics"]
    assert metrics["success"] is True
    assert metrics["tool_calls"] == 2
    assert metrics["valid_tool_calls"] == 1
    assert metrics["invalid_tool_calls"] == 1
    assert metrics["recovery_count"] == 1
    assert metrics["state_change_steps"] == 1
    assert metrics["termination_reason"] == "task_completed"
    assert metrics["usage"]["input_tokens"] == 20
    assert metrics["latency_seconds"]["initialization"] < 5
    assert metrics["latency_seconds"]["evaluation"] < 5
    assert result.artifact["action_steps"][0]["api_calls_redacted"][0]["data"]["password"] == "[REDACTED]"

    messages = result.training_projection["instances"][0]["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "assistant"]
    first_call = messages[1]["tool_calls"][0]
    assert first_call["function"]["name"] == APPWORLD_PYTHON_TOOL_NAME
    assert first_call["extra"] == {
        "origin": "parsed_assistant_content",
        "native_provider_tool_call": False,
    }
    assert messages[2]["tool_call_id"] == first_call["id"]
    assert messages[2]["name"] == APPWORLD_PYTHON_TOOL_NAME
    assert messages[1]["loss"] is False
    assert messages[-1]["loss"] is True
    assert messages[-1]["content"].endswith("\n\n")
    assert [message["role"] for message in backend.calls[0]] == ["user"]
    assert [message["role"] for message in backend.calls[1]] == ["user", "assistant", "user"]
    assert backend.calls[1][1]["content"] == messages[1]["sampled_content"]
    assert "tool_calls" not in backend.calls[1][1]
    assert backend.calls[1][2]["content"] == messages[2]["content"]
    assert project_appworld_messages_for_react_scaffold(messages)[:3] == backend.calls[1]
    scaffold_projection = project_appworld_conversation_for_react_scaffold(result.training_projection)
    assert len(Dataset.create_from_dict(result.training_projection)) == 1
    assert len(Dataset.create_from_dict(scaffold_projection)) == 1
    scaffold_messages = scaffold_projection["instances"][0]["messages"]
    assert scaffold_messages[1]["loss"] is False
    assert (
        scaffold_projection["instances"][0]["metadata"]["format_version"]
        == APPWORLD_REACT_TRAINING_PROJECTION_FORMAT_VERSION
    )
    assert scaffold_messages[-1] == {
        "role": "assistant",
        "content": messages[-1]["sampled_content"],
        "loss": True,
    }
    metadata = result.training_projection["instances"][0]["metadata"]
    assert metadata["semantic_roles"] is True
    assert metadata["observation_role"] == "tool"
    assert metadata["requires_scaffold_projection"] is True
    assert metadata["replay_required"] is True
    assert metadata["eligible_for_success_only_sft"] is False
    assert metadata["eligible_for_success_plus_recovery_sft"] is False


def test_episode_scores_context_budget_terminal_after_executed_step(monkeypatch, tmp_path):
    monkeypatch.setenv("APPWORLD_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.load_reference_prompt",
        lambda source: "USER:\nTask: {{ instruction }}",
    )
    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.scaffold_identity",
        lambda source: {"id": "fake-reference", "prompt_sha256": "0" * 64},
    )
    monkeypatch.setattr("lmflow.agentic.appworld_episode.configure_appworld_freezegun", lambda: None)

    class ContextExhaustedAfterOneStepBackend:
        def __init__(self):
            self.calls = 0

        def complete(self, *, messages, tools, model_name, model_kwargs):
            del messages, tools, model_name, model_kwargs
            self.calls += 1
            if self.calls == 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "I will probe.\n```python\nfail()\n```",
                    },
                    "finish_reason": "stop",
                    "raw_response": {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
                }
            raise AppWorldContextBudgetExhaustedError("exact prompt leaves no completion budget")

    result = run_appworld_episode(
        ContextExhaustedAfterOneStepBackend(),
        task_id=APPWORLD_TINY_TASK_IDS[0],
        model_name="student-model",
        model_revision="fixed",
        trajectory_id="context-budget:task",
        appworld_root=tmp_path,
        appworld_source=tmp_path,
        experiment_name="context-budget-original",
        max_steps=2,
        world_factory=lambda **kwargs: FakeWorld(tmp_path / "original-context-budget"),
    )

    verify_manifest_digest(result.artifact)
    metrics = result.artifact["metrics"]
    assert metrics["success"] is False
    assert metrics["task_status"] == "incomplete"
    assert metrics["steps"] == 1
    assert metrics["model_calls"] == 1
    assert metrics["termination_reason"] == APPWORLD_CONTEXT_BUDGET_EXHAUSTED
    assert metrics["failure_type"] == APPWORLD_CONTEXT_BUDGET_EXHAUSTED
    assert result.artifact["runner_error"] is None
    assert result.artifact["evaluator_error"] is None
    assert result.artifact["official_evaluation"]["success"] is False
    assert result.artifact["official_evaluation"]["num_tests"] == 1
    assert result.artifact["official_evaluation"]["passes"] == []

    replay = replay_appworld_episode(
        result.artifact,
        appworld_root=tmp_path,
        experiment_name="context-budget-replay",
        world_factory=lambda **kwargs: FakeWorld(tmp_path / "replay-context-budget"),
    )
    finalized = finalize_appworld_candidate(result, replay)
    assert replay["replay_match"] is True
    assert finalized.admission["data_class"] == "D"
    assert finalized.admission["failure_type"] == APPWORLD_CONTEXT_BUDGET_EXHAUSTED
    assert finalized.admission["admitted_for_sft"] is False
    assert finalized.training_projection is None


def test_episode_keeps_ordinary_backend_error_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("APPWORLD_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.load_reference_prompt",
        lambda source: "USER:\nTask: {{ instruction }}",
    )
    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.scaffold_identity",
        lambda source: {"id": "fake-reference", "prompt_sha256": "0" * 64},
    )
    monkeypatch.setattr("lmflow.agentic.appworld_episode.configure_appworld_freezegun", lambda: None)

    class FailingBackend:
        def complete(self, *, messages, tools, model_name, model_kwargs):
            del messages, tools, model_name, model_kwargs
            raise RuntimeError("ordinary backend failure")

    result = run_appworld_episode(
        FailingBackend(),
        task_id=APPWORLD_TINY_TASK_IDS[0],
        model_name="student-model",
        model_revision="fixed",
        trajectory_id="backend-failure:task",
        appworld_root=tmp_path,
        appworld_source=tmp_path,
        experiment_name="backend-failure",
        max_steps=1,
        world_factory=lambda **kwargs: FakeWorld(tmp_path / "ordinary-backend-failure"),
    )

    assert result.artifact["metrics"]["termination_reason"] == "model_backend_error"
    assert result.artifact["metrics"]["failure_type"] == "model_backend_error"
    assert result.artifact["metrics"]["task_status"] == "runner_error"
    assert result.artifact["runner_error"] == {
        "type": "RuntimeError",
        "message": "ordinary backend failure",
    }


def test_context_budget_terminal_cannot_be_promoted_to_partial_sft_data():
    data_class = _candidate_class(
        {
            "runner_error": None,
            "evaluator_error": None,
            "metrics": {
                "success": False,
                "failure_type": APPWORLD_CONTEXT_BUDGET_EXHAUSTED,
                "invalid_tool_calls": 0,
                "recovery_count": 0,
            },
        },
        {
            "replay_error": None,
            "replay_match": True,
            "collateral_invariant_passed": True,
            "sealed_partial_signal": True,
        },
    )

    assert data_class == "D"


def test_replay_gate_admits_verified_recovery_and_masks_failed_action(monkeypatch, tmp_path):
    monkeypatch.setenv("APPWORLD_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.load_reference_prompt",
        lambda source: "USER:\nTask: {{ instruction }}",
    )
    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.scaffold_identity",
        lambda source: {"id": "fake-reference", "prompt_sha256": "0" * 64},
    )
    monkeypatch.setattr("lmflow.agentic.appworld_episode.configure_appworld_freezegun", lambda: None)

    original_world = FakeWorld(tmp_path / "original")
    result = run_appworld_episode(
        FakeBackend(),
        task_id=APPWORLD_TINY_TASK_IDS[0],
        model_name="teacher-model",
        model_revision="fixed",
        trajectory_id="pilot:task:candidate-0",
        appworld_root=tmp_path,
        appworld_source=tmp_path,
        experiment_name="pilot-original",
        world_factory=lambda **kwargs: original_world,
    )
    replay = replay_appworld_episode(
        result.artifact,
        appworld_root=tmp_path,
        experiment_name="pilot-replay",
        world_factory=lambda **kwargs: FakeWorld(tmp_path / "replay"),
    )

    verify_manifest_digest(replay)
    assert replay["replay_match"] is True
    assert replay["collateral_invariant_passed"] is True
    finalized = finalize_appworld_candidate(result, replay)
    assert finalized.admission["data_class"] == "B"
    assert finalized.admission["admitted_for_sft"] is True
    assert finalized.admission["sft_arms"] == {"success_only": False, "success_plus_recovery": True}
    assert finalized.admission["usage"]["trainable_output_tokens"] == 5
    messages = finalized.training_projection["instances"][0]["messages"]
    assert [message["loss"] for message in messages if message["role"] == "assistant"] == [False, True]
    assert finalized.training_projection["instances"][0]["metadata"]["hidden_verifier_material_included"] is False

    summary = summarize_appworld_candidates([finalized.admission])
    assert summary["accepted_yield"] == 1.0
    assert summary["class_counts"]["B"] == 1
    assert summary["trainable_output_tokens"] == {"reported_count": 1, "p50": 5, "p95": 5}


def test_replay_mismatch_is_excluded_as_invalid_infrastructure(monkeypatch, tmp_path):
    monkeypatch.setenv("APPWORLD_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.load_reference_prompt",
        lambda source: "USER:\nTask: {{ instruction }}",
    )
    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.scaffold_identity",
        lambda source: {"id": "fake-reference", "prompt_sha256": "0" * 64},
    )
    monkeypatch.setattr("lmflow.agentic.appworld_episode.configure_appworld_freezegun", lambda: None)

    result = run_appworld_episode(
        FakeBackend(),
        task_id=APPWORLD_TINY_TASK_IDS[0],
        model_name="teacher-model",
        model_revision="fixed",
        trajectory_id="pilot:task:candidate-mismatch",
        appworld_root=tmp_path,
        appworld_source=tmp_path,
        experiment_name="pilot-original-mismatch",
        world_factory=lambda **kwargs: FakeWorld(tmp_path / "original-mismatch"),
    )
    replay = replay_appworld_episode(
        result.artifact,
        appworld_root=tmp_path,
        experiment_name="pilot-replay-mismatch",
        world_factory=lambda **kwargs: FakeWorld(tmp_path / "replay-mismatch", state_value="different"),
    )
    finalized = finalize_appworld_candidate(result, replay)

    assert replay["replay_match"] is False
    assert finalized.admission["data_class"] == "E"
    assert finalized.admission["collateral_rejected"] is True
    assert finalized.training_projection is None


def test_scaffold_projection_rejects_unlinked_or_duplicate_observations():
    call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": APPWORLD_PYTHON_TOOL_NAME, "arguments": '{"code":"complete()"}'},
    }
    messages = [
        {"role": "assistant", "content": "```python\ncomplete()\n```", "tool_calls": [call]},
        {"role": "tool", "name": APPWORLD_PYTHON_TOOL_NAME, "tool_call_id": "call-1", "content": "done"},
    ]
    assert project_appworld_messages_for_react_scaffold(messages) == [
        {"role": "assistant", "content": "```python\ncomplete()\n```"},
        {"role": "user", "content": "done"},
    ]

    unlinked = [{"role": "tool", "name": APPWORLD_PYTHON_TOOL_NAME, "tool_call_id": "missing", "content": "done"}]
    with pytest.raises(ValueError, match="unknown action"):
        project_appworld_messages_for_react_scaffold(unlinked)

    with pytest.raises(ValueError, match="duplicate observations"):
        project_appworld_messages_for_react_scaffold(messages + [messages[-1]])


class FakeTokenNativeBackend:
    def __init__(self):
        self.calls = []
        self.contents = [
            "I will probe.\n```python\nfail()\n```",
            "I will recover and finish.\n```python\ncomplete()\n```",
        ]
        self.prompts = [(1, 2), (1, 2, 3, 4, 5, 6)]
        self.outputs = [(3, 4), (7,)]

    def complete(self, *, messages, tools, model_name, model_kwargs):
        call_index = len(self.calls)
        self.calls.append({"messages": messages, "model_kwargs": model_kwargs})
        request_id = model_kwargs["extra_body"]["request_id"]
        output_token_ids = self.outputs[call_index]
        return {
            "message": {"role": "assistant", "content": self.contents[call_index]},
            "finish_reason": "stop",
            "raw_response": {
                "id": f"chatcmpl-{request_id}",
                "prompt_token_ids": list(self.prompts[call_index]),
                "choices": [
                    {
                        "finish_reason": "stop",
                        "token_ids": list(output_token_ids),
                        "logprobs": {
                            "content": [
                                {"token": f"token_id:{token_id}", "logprob": -0.1 - index / 10}
                                for index, token_id in enumerate(output_token_ids)
                            ]
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": len(self.prompts[call_index]),
                    "completion_tokens": len(output_token_ids),
                    "total_tokens": len(self.prompts[call_index]) + len(output_token_ids),
                },
            },
        }


def test_token_native_appworld_fixture_preserves_policy_and_observation_tokens(monkeypatch, tmp_path):
    # run_appworld_episode configures AppWorld through its process-wide root.
    # Register the fixture root with monkeypatch so the caller's root is restored
    # before optional real-environment tests run in the same pytest process.
    monkeypatch.setenv("APPWORLD_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.load_reference_prompt",
        lambda source: "USER:\nTask: {{ instruction }}",
    )
    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.scaffold_identity",
        lambda source: {"id": "fake-reference", "prompt_sha256": "0" * 64},
    )
    monkeypatch.setattr("lmflow.agentic.appworld_episode.configure_appworld_freezegun", lambda: None)

    backend = FakeTokenNativeBackend()
    stages = []
    step_stages = []

    def render_prompt_token_ids(messages, model_kwargs):
        assert model_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        return backend.prompts[0 if len(messages) == 1 else 1]

    recorder = AppWorldTokenNativeCompletionRecorder(
        backend,
        request_id_prefix="fixture-policy-task-rollout-0",
        prompt_token_ids_renderer=render_prompt_token_ids,
        max_model_len=32768,
        evidence_sink=lambda stage, call_index, evidence: stages.append((stage, call_index, evidence)),
    )
    result = run_appworld_episode(
        recorder,
        task_id=APPWORLD_TINY_TASK_IDS[0],
        model_name="Qwen/Qwen3-4B",
        model_revision="revision",
        trajectory_id="token-native:task",
        appworld_root=tmp_path,
        appworld_source=tmp_path,
        experiment_name="token-native-fixture",
        world_factory=lambda **kwargs: FakeWorld(tmp_path),
        step_evidence_sink=lambda stage, step_number, evidence: step_stages.append((stage, step_number, evidence)),
    )

    assert result.artifact["metrics"]["model_calls"] == 2
    assert [message["role"] for message in backend.calls[1]["messages"]] == ["user", "assistant", "user"]
    assert backend.calls[0]["model_kwargs"]["logprobs"] is True
    assert backend.calls[0]["model_kwargs"]["extra_body"]["return_token_ids"] is True
    assert [stage for stage, _, _ in stages] == [
        "request_intent",
        "raw_response",
        "normalized_response",
        "token_evidence",
        "request_termination",
    ] * 2
    first_request_intent = stages[0][2]
    assert first_request_intent["messages"] == backend.calls[0]["messages"]
    assert first_request_intent["model_kwargs"] == backend.calls[0]["model_kwargs"]
    assert [stage for stage, _, _ in step_stages] == [
        "action_intent",
        "raw_execution",
        "normalized_transition",
        "step_termination",
        "action_intent",
        "raw_execution",
        "normalized_transition",
        "step_termination",
    ]
    assert step_stages[0][2]["state_before_sha256"] == step_stages[2][2]["state_before_sha256"]
    assert step_stages[1][2]["output"] == step_stages[2][2]["output"]

    audit = recorder.build_audit(policy_version="qwen3-4b-starting@revision")
    verify_manifest_digest(audit)
    assert audit["canonical_prompts_match"] is True
    assert audit["sampled_anchors_match"] is True
    assert audit["single_flattened_rollout_ready"] is True
    assert audit["transitions"][0]["environment_token_ids"] == [5, 6]
    assert audit["flattened_sequence"]["input_ids"] == [1, 2, 3, 4, 5, 6, 7]
    assert audit["flattened_sequence"]["policy_origin_mask"] == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    assert audit["retokenized_sampled_tokens_used"] is False


def test_token_native_audit_preserves_drift_evidence_and_strict_assembly_fails():
    calls = [
        VLLMChatTokenData("first", "chatcmpl-first", (1, 2), (3, 4), (-0.1, -0.2), "stop"),
        VLLMChatTokenData("second", "chatcmpl-second", (1, 2, 3, 99, 5), (6,), (-0.3,), "stop"),
    ]
    expected_prompts = [calls[0].prompt_token_ids, calls[1].prompt_token_ids]

    audit = build_appworld_token_native_audit(
        calls,
        expected_prompt_token_ids=expected_prompts,
        policy_version="policy-v1",
    )

    assert audit["canonical_prompts_match"] is True
    assert audit["sampled_anchors_match"] is False
    assert audit["single_flattened_rollout_ready"] is False
    assert audit["flattened_sequence"] is None
    assert audit["transitions"][0]["sampled_anchor_first_difference"] == {
        "position": 3,
        "expected_token_id": 4,
        "actual_token_id": 99,
        "kind": "token_mismatch",
    }
    with pytest.raises(ValueError, match="sampled-token anchor mismatch"):
        assemble_verified_appworld_token_sequence(
            calls,
            expected_prompt_token_ids=expected_prompts,
            policy_version="policy-v1",
        )


def test_qwen3_prompt_renderer_preserves_messages_and_thinking_identity():
    class FakeTokenizer:
        def __init__(self):
            self.arguments = None
            self.chat_template = (
                "{% if message.role == 'user' and "
                "not(message.content.startswith('<tool_response>') "
                "and message.content.endswith('</tool_response>')) %}kept{% endif %} "
                "{% if loop.last or (not loop.last and reasoning_content) %}assistant{% endif %}"
            )

        def apply_chat_template(self, messages, **kwargs):
            self.arguments = {"messages": messages, **kwargs}
            return {"input_ids": [[1, 2, 3]]}

    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "reasoning_content": "think", "content": "action"},
        {"role": "user", "content": "Output:\nresult"},
    ]

    actual = qwen3_appworld_prompt_token_ids(
        tokenizer,
        messages,
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    )

    assert actual == (1, 2, 3)
    assert tokenizer.arguments == {
        "messages": messages,
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "chat_template": qwen3_appworld_replay_chat_template(tokenizer),
    }
    assert "not(message.content.startswith('Output:\\n'))" in tokenizer.arguments["chat_template"]
    assert (
        "reasoning_content or (enable_thinking is defined and enable_thinking is false)"
        in tokenizer.arguments["chat_template"]
    )


def test_qwen3_replay_template_identity_is_fail_closed_and_stable():
    condition = (
        "and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>'))"
    )
    historical = "if loop.last or (not loop.last and reasoning_content)"
    tokenizer = SimpleNamespace(chat_template=f"prefix {condition} middle {historical} suffix")

    replay_template = qwen3_appworld_replay_chat_template(tokenizer)
    identity = qwen3_appworld_replay_chat_template_identity(tokenizer)

    assert replay_template == (
        f"prefix {condition} and not(message.content.startswith('Output:\\n')) middle "
        "if loop.last or (not loop.last and "
        "(reasoning_content or (enable_thinking is defined and enable_thinking is false))) suffix"
    )
    assert identity == {
        "policy_id": "lmflow.appworld-qwen3-reasoning-replay/v2",
        "source_chat_template_sha256": "f46fa44f6f55681061f774239bbeffeb5a1f1ae1ae486594ea5fde531700adbf",
        "replay_chat_template_sha256": "a0a1bc73eef49db0f25169a05d81fa27e9b84a1c4670704b21b9b48b04862e09",
    }

    with pytest.raises(ValueError, match="last-query boundary is unsupported"):
        qwen3_appworld_replay_chat_template(SimpleNamespace(chat_template="unrecognized"))
    with pytest.raises(ValueError, match="historical reasoning boundary is unsupported"):
        qwen3_appworld_replay_chat_template(SimpleNamespace(chat_template=f"prefix {condition} suffix"))
    with pytest.raises(ValueError, match="non-empty chat template"):
        qwen3_appworld_replay_chat_template(SimpleNamespace(chat_template=None))


def test_qwen3_replay_policy_preserves_length_stopped_reasoning_before_output_observation():
    condition = (
        "and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>'))"
    )
    historical = "if loop.last or (not loop.last and reasoning_content)"

    class MinimalQwen3Tokenizer:
        def __init__(self):
            self.chat_template = f"prefix {condition} middle {historical} suffix"

        def apply_chat_template(self, messages, *, chat_template, add_generation_prompt, **kwargs):
            preserve_output_reasoning = "not(message.content.startswith('Output:\\n'))" in chat_template
            last_query_index = len(messages) - 1
            for index in range(len(messages) - 1, -1, -1):
                message = messages[index]
                if message["role"] != "user":
                    continue
                if preserve_output_reasoning and message["content"].startswith("Output:\n"):
                    continue
                last_query_index = index
                break

            rendered = ""
            for index, message in enumerate(messages):
                if message["role"] == "user":
                    rendered += f"<u>{message['content']}</u>"
                elif message["role"] == "assistant":
                    reasoning = message.get("reasoning_content", "")
                    content = message.get("content", "")
                    if index > last_query_index and reasoning:
                        rendered += f"<a><think>\n{reasoning.strip()}\n</think>\n\n{content.lstrip()}</a>"
                    else:
                        rendered += f"<a>{content}</a>"
            if add_generation_prompt:
                rendered += "<a>"
            return list(rendered.encode("utf-8"))

    tokenizer = MinimalQwen3Tokenizer()
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "reasoning_content": "unfinished reasoning", "content": ""},
        {"role": "user", "content": "Output:\n```\nNo code available to execute.\n```\n\n"},
    ]
    model_kwargs = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
    first_prompt = qwen3_appworld_prompt_token_ids(tokenizer, messages[:1], model_kwargs)
    sampled_length_stopped_output = tuple(b"<think>\nunfinished reasoning")
    replay_prompt = qwen3_appworld_prompt_token_ids(tokenizer, messages, model_kwargs)
    stock_prompt = tuple(
        tokenizer.apply_chat_template(
            messages,
            chat_template=tokenizer.chat_template,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=True,
        )
    )
    expected_sampled_prefix = first_prompt + sampled_length_stopped_output

    assert replay_prompt[: len(expected_sampled_prefix)] == expected_sampled_prefix
    assert stock_prompt[: len(expected_sampled_prefix)] != expected_sampled_prefix
    deterministic_suffix = tuple(b"\n</think>")
    assert replay_prompt[len(expected_sampled_prefix) :][: len(deterministic_suffix)] == deterministic_suffix

    stop_content = "```python\ncomplete()\n```"
    third_messages = messages + [
        {
            "role": "assistant",
            "reasoning_content": "second reasoning",
            "content": stop_content,
        },
        {"role": "user", "content": "Output:\n```\nExecution successful.\n```\n\n"},
    ]
    sampled_stop_output = tuple(f"<think>\nsecond reasoning\n</think>\n\n{stop_content}</a>".encode())
    third_prompt = qwen3_appworld_prompt_token_ids(tokenizer, third_messages, model_kwargs)
    expected_stop_prefix = replay_prompt + sampled_stop_output

    assert third_prompt[: len(expected_stop_prefix)] == expected_stop_prefix


def test_token_native_recorder_emits_raw_evidence_before_token_assertion():
    class InvalidTokenBackend(FakeTokenNativeBackend):
        def complete(self, **kwargs):
            response = super().complete(**kwargs)
            response["raw_response"]["choices"][0]["logprobs"]["content"][0]["token"] = "decoded-text"
            return response

    backend = InvalidTokenBackend()
    stages = []
    recorder = AppWorldTokenNativeCompletionRecorder(
        backend,
        request_id_prefix="invalid-token-evidence",
        prompt_token_ids_renderer=lambda messages, model_kwargs: backend.prompts[0],
        max_model_len=32768,
        evidence_sink=lambda stage, call_index, evidence: stages.append(stage),
    )

    with pytest.raises(ValueError, match="cannot prove sampled-token identity"):
        recorder.complete(
            messages=[{"role": "user", "content": "task"}],
            tools=[],
            model_name="model",
            model_kwargs={
                "max_completion_tokens": 16,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        )

    assert stages == [
        "request_intent",
        "raw_response",
        "normalized_response",
        "token_evidence_error",
        "request_termination",
    ]
    assert recorder.calls == ()


def test_token_native_recorder_seals_backend_error_before_reraising():
    class BackendHTTPError(RuntimeError):
        status_code = 400
        request_id = "provider-request-id"

    class FailingBackend(FakeTokenNativeBackend):
        def complete(self, **kwargs):
            self.calls.append(kwargs)
            raise BackendHTTPError("non-finite log-prob response")

    backend = FailingBackend()
    backend.prompts[0] = (1, 2, 3)
    stages = []
    recorder = AppWorldTokenNativeCompletionRecorder(
        backend,
        request_id_prefix="backend-error-evidence",
        prompt_token_ids_renderer=lambda messages, model_kwargs: backend.prompts[0],
        max_model_len=32768,
        evidence_sink=lambda stage, call_index, evidence: stages.append((stage, evidence)),
    )

    with pytest.raises(BackendHTTPError, match="non-finite log-prob response"):
        recorder.complete(
            messages=[{"role": "user", "content": "task"}],
            tools=[],
            model_name="model",
            model_kwargs={
                "max_completion_tokens": 16,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        )

    assert [stage for stage, _ in stages] == [
        "request_intent",
        "backend_error",
        "request_termination",
    ]
    assert stages[1][1] == {
        "request_id": "backend-error-evidence-call-000",
        "provider_request_id": "provider-request-id",
        "type": "BackendHTTPError",
        "message": "non-finite log-prob response",
        "status": "backend_error",
        "status_code": 400,
    }
    assert stages[2][1]["status"] == "backend_error"
    assert stages[2][1]["finish_reason"] == "backend_error"
    assert recorder.calls == ()


def test_token_native_recorder_seals_normalization_error_after_raw_response():
    class InvalidMessageBackend(FakeTokenNativeBackend):
        def complete(self, **kwargs):
            response = super().complete(**kwargs)
            response["message"]["role"] = "user"
            return response

    backend = InvalidMessageBackend()
    stages = []
    recorder = AppWorldTokenNativeCompletionRecorder(
        backend,
        request_id_prefix="normalize-error-evidence",
        prompt_token_ids_renderer=lambda messages, model_kwargs: backend.prompts[0],
        max_model_len=32768,
        evidence_sink=lambda stage, call_index, evidence: stages.append((stage, evidence)),
    )

    with pytest.raises(ValueError, match="role must be 'assistant'"):
        recorder.complete(
            messages=[{"role": "user", "content": "task"}],
            tools=[],
            model_name="model",
            model_kwargs={
                "max_completion_tokens": 16,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        )

    assert [stage for stage, _ in stages] == [
        "request_intent",
        "raw_response",
        "normalized_response_error",
        "request_termination",
    ]
    assert stages[2][1]["status"] == "invalid_normalized_response"
    assert stages[3][1]["status"] == "invalid_normalized_response"
    assert recorder.calls == ()


def test_token_native_recorder_caps_output_before_sending_request():
    backend = FakeTokenNativeBackend()
    backend.prompts[0] = tuple(range(9))
    stages = []
    recorder = AppWorldTokenNativeCompletionRecorder(
        backend,
        request_id_prefix="context-cap",
        prompt_token_ids_renderer=lambda messages, model_kwargs: backend.prompts[0],
        max_model_len=10,
        evidence_sink=lambda stage, call_index, evidence: stages.append((stage, evidence)),
    )

    recorder.complete(
        messages=[{"role": "user", "content": "task"}],
        tools=[],
        model_name="model",
        model_kwargs={
            "max_completion_tokens": 3,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    )

    assert backend.calls[0]["model_kwargs"]["max_completion_tokens"] == 1
    request_intent = stages[0][1]
    assert request_intent["request_will_be_sent"] is True
    assert request_intent["context_budget"] == {
        "policy_id": "lmflow.appworld-dynamic-output-cap/v1",
        "max_model_len": 10,
        "prompt_tokens": 9,
        "budget_field": "max_completion_tokens",
        "requested_output_tokens": 3,
        "available_output_tokens": 1,
        "effective_output_tokens": 1,
        "cap_applied": True,
        "history_truncated": False,
        "request_will_be_sent": True,
    }
    assert stages[-1][1]["context_budget"] == request_intent["context_budget"]
    assert recorder.context_budgets == (request_intent["context_budget"],)


@pytest.mark.parametrize("prompt_length", [10, 11])
def test_token_native_recorder_rejects_exhausted_context_before_backend_call(prompt_length):
    backend = FakeTokenNativeBackend()
    backend.prompts[0] = tuple(range(prompt_length))
    stages = []
    recorder = AppWorldTokenNativeCompletionRecorder(
        backend,
        request_id_prefix="context-exhausted",
        prompt_token_ids_renderer=lambda messages, model_kwargs: backend.prompts[0],
        max_model_len=10,
        evidence_sink=lambda stage, call_index, evidence: stages.append((stage, evidence)),
    )

    with pytest.raises(AppWorldContextBudgetExhaustedError, match="leaves no completion budget"):
        recorder.complete(
            messages=[{"role": "user", "content": "task"}],
            tools=[],
            model_name="model",
            model_kwargs={
                "max_completion_tokens": 3,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        )

    assert backend.calls == []
    assert [stage for stage, _ in stages] == [
        "request_intent",
        "context_budget_error",
        "request_termination",
    ]
    assert stages[0][1]["request_will_be_sent"] is False
    assert stages[0][1]["context_budget"]["prompt_tokens"] == prompt_length
    assert stages[0][1]["context_budget"]["effective_output_tokens"] == 0
    assert stages[1][1]["type"] == "AppWorldContextBudgetExhaustedError"
    assert stages[-1][1]["status"] == "rejected_before_request"
    assert recorder.context_budgets[0]["request_will_be_sent"] is False
