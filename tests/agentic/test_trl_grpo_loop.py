import copy
from types import SimpleNamespace

import pytest
import torch

from lmflow.agentic.contracts import TaskSpec
from lmflow.agentic.trl_grpo_loop import _SynchronousRolloutBridge
from lmflow.utils.protocol import DataProto


def _setup(rollout):
    model = torch.nn.Linear(1, 1).train()
    trainer = SimpleNamespace(model=model, state=SimpleNamespace(global_step=0))

    def publish(trainer, version):
        trainer.model.eval()  # Publication must not leak this mode into training.
        return dict(policy_version=version, global_step=0, weight_digest="weights-v0", source="test")

    bridge = _SynchronousRolloutBridge(
        [TaskSpec("task", [])],
        SimpleNamespace(max_steps=2, num_generations=2),
        rollout,
        publish,
        None,
        "run",
    )
    return bridge, trainer


@pytest.mark.parametrize("corruption", ["version", "publication", "identity"])
def test_rejects_stale_publication_and_mutated_request_identity(corruption):
    def rollout(requests, trainer):
        if corruption == "identity":
            requests.non_tensor_batch["rollout_ids"][0] = 999999
        meta = copy.deepcopy(requests.meta_info)
        if corruption == "version":
            meta["policy_version"] = "stale"
        if corruption == "publication":
            meta["policy_publication"]["weight_digest"] = "not-loaded"
        return DataProto.from_dict(non_tensors=requests.non_tensor_batch, meta_info=meta)

    bridge, trainer = _setup(rollout)
    with pytest.raises(ValueError, match="stale|publication|rollout_ids"):
        bridge.rollout_func(bridge.handles * 2, trainer)
    assert trainer.model.training
    assert bridge.active is None and bridge.completed_steps == 0


def test_failed_sampling_restores_training_mode_without_consumption():
    def rollout(requests, trainer):
        trainer.model.eval()
        raise RuntimeError("sampling failed")

    bridge, trainer = _setup(rollout)
    with pytest.raises(RuntimeError, match="sampling failed"):
        bridge.rollout_func(bridge.handles * 2, trainer)
    assert trainer.model.training
    assert not bridge.history and bridge.completed_steps == 0


def test_skipped_optimizer_step_does_not_advance_or_publish_policy():
    bridge, trainer = _setup(lambda *args: None)
    trainer.accelerator = SimpleNamespace(optimizer_step_was_skipped=True)
    bridge.trainer = trainer
    bridge.active = SimpleNamespace(_reward_consumed=True)
    with pytest.raises(RuntimeError, match="skipped"):
        bridge.callback().on_step_end(None, SimpleNamespace(global_step=1), None)
    assert bridge.completed_steps == 0 and bridge.final_publication is None
