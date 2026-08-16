import os

import pytest

from lmflow.agentic.appworld_episode import replay_appworld_episode, run_appworld_episode
from lmflow.agentic.appworld_protocol import APPWORLD_TINY_TASK_IDS, load_pinned_appworld_data_pilot_dataset
from lmflow.agentic.evaluate_appworld import verify_appworld_install


class OneStepBackend:
    def complete(self, *, messages, tools, model_name, model_kwargs):
        return {
            "message": {
                "role": "assistant",
                "content": "```python\nprint(apis.api_docs.show_app_descriptions())\n```",
            },
            "finish_reason": "stop",
            "raw_response": {"usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}},
        }


@pytest.mark.optional_backend
def test_pinned_appworld_tiny_load_reset_and_official_evaluator():
    appworld_root = os.environ.get("APPWORLD_ROOT")
    appworld_source = os.environ.get("APPWORLD_SOURCE")
    if not appworld_root or not appworld_source:
        pytest.skip("APPWORLD_ROOT and APPWORLD_SOURCE select the pinned optional integration environment")
    # freezegun scans already imported modules when AppWorld freezes task time.
    # vLLM exposes lazy processor attributes whose optional multimedia imports
    # must not be triggered by that scan in the unified agentic environment.
    import vllm.transformers_utils.processors  # noqa: F401

    result = verify_appworld_install(
        appworld_root=appworld_root,
        appworld_source=appworld_source,
    )
    assert result["reset_equal"] is True
    assert result["dataset_instance_count"] == 1
    assert result["official_evaluation_stats"]["num_tests"] > 0
    pilot_dataset, pilot_manifest = load_pinned_appworld_data_pilot_dataset(appworld_root=appworld_root)
    assert len(pilot_dataset) == 9
    assert pilot_manifest["scenario_disjoint"] is True
    assert pilot_manifest["source"]["split"] == "train"


@pytest.mark.optional_backend
def test_real_appworld_episode_uses_unfrozen_latency_and_official_evaluator():
    appworld_root = os.environ.get("APPWORLD_ROOT")
    appworld_source = os.environ.get("APPWORLD_SOURCE")
    if not appworld_root or not appworld_source:
        pytest.skip("APPWORLD_ROOT and APPWORLD_SOURCE select the pinned optional integration environment")
    result = run_appworld_episode(
        OneStepBackend(),
        task_id=APPWORLD_TINY_TASK_IDS[0],
        model_name="integration-backend",
        model_revision="fixed",
        trajectory_id="appworld-real-one-step-integration",
        appworld_root=appworld_root,
        appworld_source=appworld_source,
        experiment_name="lmflow-appworld-real-one-step-integration",
        model_kwargs={"temperature": 0},
        max_steps=1,
    )
    metrics = result.artifact["metrics"]
    assert metrics["steps"] == 1
    assert metrics["valid_tool_calls"] == 1
    assert metrics["api_call_attempts"] == 1
    assert metrics["latency_seconds"]["initialization"] < 60
    assert metrics["latency_seconds"]["environment"] < 60
    assert result.artifact["official_evaluation"]["num_tests"] > 0
    replay = replay_appworld_episode(
        result.artifact,
        appworld_root=appworld_root,
        experiment_name="lmflow-appworld-real-one-step-integration-replay",
    )
    assert replay["replay_match"] is True
    assert replay["collateral_invariant_passed"] is False
