"""AppWorld-local fresh K-group producer for the synchronous native TRL Pipeline.

Serving/publication belongs to the caller. This module owns neither a service
process nor a Trainer loop. AppWorld evaluation evidence stays in local audit
files; only terminal test counts and the scalar reward enter the training batch.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lmflow.agentic.appworld_episode import replay_appworld_episode, run_appworld_episode
from lmflow.agentic.appworld_protocol import (
    APPWORLD_DATA_VERSION,
    APPWORLD_REVISION,
    canonical_appworld_sliced_instance_id,
    canonical_json_sha256,
)
from lmflow.agentic.appworld_token_native import AppWorldTokenNativeCompletionRecorder
from lmflow.agentic.contracts import TaskSpec
from lmflow.agentic.scaffolds.appworld_react_code.scaffold import AppWorldPrompt
from lmflow.agentic.vllm_token_native import _pack_token_sequences, assemble_vllm_chat_token_data
from lmflow.utils.protocol import DataProto

APPWORLD_GRPO_FORMAT = "lmflow.appworld-grpo/v1"


def appworld_grpo_tasks(dataset) -> list[TaskSpec]:
    """Adapt pinned Train dataset rows, copying only model-visible task facts."""
    tasks = []
    for row in dataset.to_dict()["instances"]:
        if row.get("source_split") != "train":
            raise ValueError("AppWorld GRPO requires official Train rows")
        identity = canonical_appworld_sliced_instance_id(row["task_id"], source_split="train")
        if row.get("instance_id") != identity:
            raise ValueError("AppWorld row has a non-canonical identity")
        tasks.append(
            TaskSpec(
                identity,
                [],  # The fresh world's public task metadata is rendered by the episode runner.
                environment={"task_id": row["task_id"], "source_split": "train"},
                metadata={"task_spec_sha256": row["task_spec_sha256"]},
            )
        )
    return tasks


def official_test_fraction(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only the official terminal scalar; never expose verifier text."""
    if artifact.get("runner_error") or artifact.get("evaluator_error"):
        raise ValueError("infrastructure failure cannot become an AppWorld reward")
    evaluation = artifact.get("official_evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("missing official evaluation")
    count, passes, failures = (evaluation.get(key) for key in ("num_tests", "passes", "failures"))
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or not isinstance(passes, list)
        or not isinstance(failures, list)
        or len(passes) + len(failures) != count
    ):
        raise ValueError("invalid official terminal test counts")
    return {
        "reward": len(passes) / count,
        "passed": len(passes),
        "total": count,
        "official_success": evaluation.get("success") is True,
    }


def _write_json(path: Path, value: Any) -> None:
    """Publish once inside an exclusively claimed batch directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    # link is an atomic, no-replace publication on the Linux artifact filesystem.
    os.link(temporary, path)
    temporary.unlink()


class AppWorldTokenNativeRollout:
    """Run complete fresh groups, seal evidence, and return the existing DataProto.

    ``backend_for_policy(receipt, trainer)`` returns ``(backend, loaded_receipt)``.
    The receipt must come from the actual sampling target after publication/load;
    copying an intended receipt without loading weights violates this contract.
    ``select_calls(calls)`` optionally selects whole sampled calls for optimization.
    It must have an explicit policy identity. It never changes sampled-origin masks
    or tokens, and does not inherit offline-SFT valid-action filtering.
    """

    def __init__(
        self,
        backend_for_policy: Callable,
        *,
        appworld_root: str | Path,
        appworld_source: str | Path,
        artifact_root: str | Path,
        prompt: AppWorldPrompt,
        prompt_token_ids_renderer: Callable,
        model_name: str,
        model_revision: str,
        tokenizer_revision: str,
        token_protocol_identity: Mapping[str, Any],
        model_kwargs: Mapping[str, Any],
        max_model_len: int,
        max_steps: int,
        pad_token_id: int,
        select_calls: Callable | None = None,
        selection_policy: str = "all-sampled-assistant/v1",
        world_factory: Callable | None = None,
    ):
        if not isinstance(prompt, AppWorldPrompt):
            raise TypeError("an explicit AppWorldPrompt is required")
        if select_calls is not None and selection_policy == "all-sampled-assistant/v1":
            raise ValueError("custom call selection requires its own selection_policy")
        if not selection_policy:
            raise ValueError("selection_policy must be non-empty")
        # Native behavior-old GRPO currently scores the unwarped policy at T=1.
        if model_kwargs.get("temperature") != 1.0 or model_kwargs.get("top_p", 1.0) != 1.0:
            raise ValueError("native AppWorld GRPO requires temperature=1 and top_p=1")
        forbidden = {"top_k", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty", "logit_bias"}
        if forbidden & (set(model_kwargs) | set(model_kwargs.get("extra_body", {}))):
            raise ValueError("sampling warpers are unsupported by the behavior-old bridge")
        if {"temperature", "top_p", "seed", "max_tokens", "max_completion_tokens"} & set(
            model_kwargs.get("extra_body", {})
        ):
            raise ValueError("sampling controls must not be duplicated in extra_body")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 50:
            raise ValueError("max_steps must be between 1 and 50")
        if isinstance(max_model_len, bool) or not isinstance(max_model_len, int) or max_model_len < 1:
            raise ValueError("max_model_len must be positive")
        if not all((model_name, model_revision, tokenizer_revision, token_protocol_identity)):
            raise ValueError("model, tokenizer and token protocol identities are required")
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
            raise ValueError("pad_token_id must be a non-negative integer")
        self.backend_for_policy = backend_for_policy
        self.root, self.source = Path(appworld_root), Path(appworld_source)
        self.artifact_root = Path(artifact_root)
        self.prompt = prompt
        self.renderer = prompt_token_ids_renderer
        self.model_name, self.model_revision = model_name, model_revision
        self.tokenizer_revision = tokenizer_revision
        self.token_protocol = copy.deepcopy(dict(token_protocol_identity))
        self.model_kwargs = copy.deepcopy(dict(model_kwargs))
        self.max_model_len, self.max_steps, self.pad_token_id = max_model_len, max_steps, pad_token_id
        self.select_calls, self.selection_policy = select_calls, selection_policy
        self.world_factory = world_factory

    def _jobs(self, requests, trainer):
        arrays = requests.non_tensor_batch
        receipt = copy.deepcopy(requests.meta_info["policy_publication"])
        if (
            receipt.get("policy_version") != requests.meta_info["policy_version"]
            or receipt.get("global_step") != trainer.state.global_step
            or not receipt.get("weight_digest")
            or not receipt.get("source")
        ):
            raise ValueError("stale or incomplete policy publication")
        jobs, groups = [], defaultdict(list)
        for task, task_id, group_id, rollout_id in zip(
            *(arrays[key] for key in ("tasks", "task_ids", "group_ids", "rollout_ids")), strict=True
        ):
            if task.environment.get("source_split") != "train":
                raise ValueError("only Train tasks may be sampled")
            native_id = task.environment["task_id"]
            if task_id != task.task_id or task_id != canonical_appworld_sliced_instance_id(
                native_id, source_split="train"
            ):
                raise ValueError("task identity mismatch")
            if task.messages or task.tools:
                raise ValueError("AppWorld messages must come from fresh public environment metadata")
            spec = self.root / "data" / "tasks" / native_id / "specs.json"
            if hashlib.sha256(spec.read_bytes()).hexdigest() != task.metadata["task_spec_sha256"]:
                raise ValueError("task spec digest mismatch")
            group_id, rollout_id = int(group_id), int(rollout_id)
            jobs.append(
                dict(
                    task_id=native_id,
                    instance_id=task_id,
                    group_id=group_id,
                    rollout_id=rollout_id,
                    task_spec_sha256=task.metadata["task_spec_sha256"],
                )
            )
            groups[group_id].append(task_id)
        k = trainer.args.num_generations
        if not jobs or k < 2 or any(len(ids) != k or len(set(ids)) != 1 for ids in groups.values()):
            raise ValueError("each task requires one complete K-group")
        if len({job["rollout_id"] for job in jobs}) != len(jobs):
            raise ValueError("rollout identities must be unique")
        if len({ids[0] for ids in groups.values()}) != len(groups):
            raise ValueError("each task must appear in exactly one group")
        return jobs, receipt

    def __call__(self, requests: DataProto, trainer) -> DataProto:
        jobs, receipt = self._jobs(requests, trainer)
        identity = {
            "format_version": APPWORLD_GRPO_FORMAT,
            "appworld_revision": APPWORLD_REVISION,
            "appworld_data_version": APPWORLD_DATA_VERSION,
            "policy_publication": receipt,
            "jobs": jobs,
            "model": {"name": self.model_name, "revision": self.model_revision},
            "tokenizer_revision": self.tokenizer_revision,
            "token_protocol": self.token_protocol,
            "prompt": {"id": self.prompt.identity, "sha256": self.prompt.sha256},
            "sampling": self.model_kwargs,
            "max_model_len": self.max_model_len,
            "max_steps": self.max_steps,
            "selection_policy": self.selection_policy,
        }
        directory = self.artifact_root / canonical_json_sha256(identity)
        directory.mkdir(parents=True, exist_ok=False)
        _write_json(directory / "identity.json", identity)
        rows = []
        try:
            backend, loaded = self.backend_for_policy(copy.deepcopy(receipt), trainer)
            _write_json(directory / "loaded-policy.json", dict(loaded))
            if loaded != receipt:
                raise ValueError("sampling backend loaded a different policy publication")
            for index, job in enumerate(jobs):
                rows.append(self._run_one(backend, job, receipt, directory / f"candidate-{index:04d}", trainer))
            result = self._batch(rows, requests)
            _write_json(
                directory / "complete.json", {"candidates": len(rows), "rows": [row["summary"] for row in rows]}
            )
            return result
        except Exception as error:
            _write_json(
                directory / "failure.json",
                {
                    "type": type(error).__name__,
                    "completed_candidates": len(rows),
                    "batch_returned": False,
                    "retry_allowed": False,
                },
            )
            raise

    def _run_one(self, backend, job, receipt, directory, trainer):
        directory.mkdir()
        trajectory_id = "appworld-grpo-" + canonical_json_sha256({"policy": receipt, **job})[:32]
        recorder = AppWorldTokenNativeCompletionRecorder(
            backend,
            request_id_prefix=trajectory_id,
            prompt_token_ids_renderer=self.renderer,
            max_model_len=self.max_model_len,
            evidence_sink=lambda stage, i, value: _write_json(directory / "requests" / f"{i:03d}-{stage}.json", value),
        )
        # A distinct deterministic seed per real candidate; no copied trajectories.
        sampling = copy.deepcopy(self.model_kwargs)
        sampling["seed"] = (int(sampling.get("seed", 0)) + job["rollout_id"]) % (2**63 - 1)
        episode = run_appworld_episode(
            recorder,
            task_id=job["task_id"],
            model_name=self.model_name,
            model_revision=self.model_revision,
            trajectory_id=trajectory_id,
            appworld_root=self.root,
            appworld_source=self.source,
            experiment_name=trajectory_id,
            source_split="train",
            model_kwargs=sampling,
            max_steps=self.max_steps,
            world_factory=self.world_factory,
            prompt=self.prompt,
            step_evidence_sink=lambda stage, i, value: _write_json(
                directory / "steps" / f"{i:03d}-{stage}.json", value
            ),
        )
        _write_json(directory / "episode.json", episode.artifact)
        reward = official_test_fraction(episode.artifact)
        audit = recorder.build_audit(policy_version=receipt["policy_version"])
        _write_json(directory / "tokens.json", audit)
        if not audit["canonical_prompts_match"] or not audit["sampled_anchors_match"]:
            raise ValueError("AppWorld token continuity failed")
        sequence = assemble_vllm_chat_token_data(recorder.calls)
        origin = sequence.loss_mask.clone()
        selected = [True] * len(recorder.calls) if self.select_calls is None else self.select_calls(recorder.calls)
        # Keep all exact conditioning and all original sampled logprobs, including
        # audit-only calls. Call selection only changes the optimization mask.
        selected_sequence = assemble_vllm_chat_token_data(recorder.calls, optimize_calls=selected)
        prompt_length = sequence.call_spans[0]["output_start"]
        suffix_length = len(sequence.input_ids) - prompt_length
        _write_json(
            directory / "selection.json",
            {
                "policy": self.selection_policy,
                "calls": selected,
                "prompt_length": prompt_length,
                "completion_length": suffix_length,
                "policy_origin_tokens": int(origin.sum()),
                "optimization_tokens": int(selected_sequence.loss_mask.sum()),
            },
        )
        if suffix_length > trainer.args.max_completion_length:
            raise ValueError("flattened completion including environment tokens exceeds training budget")
        max_prompt = getattr(trainer.args, "max_prompt_length", None)
        if max_prompt is not None and prompt_length > max_prompt:
            raise ValueError("actual initial prompt exceeds training budget")
        replay = replay_appworld_episode(
            episode.artifact,
            appworld_root=self.root,
            experiment_name=trajectory_id + "-replay",
            world_factory=self.world_factory,
        )
        _write_json(directory / "replay.json", replay)
        if not replay["replay_match"] or replay["replay_error"]:
            raise ValueError("AppWorld fresh replay failed; group is incomplete")
        summary = {
            **job,
            **reward,
            "termination_reason": episode.artifact["metrics"]["termination_reason"],
            "call_count": len(recorder.calls),
            "completion_length": suffix_length,
            "artifact_sha256": episode.artifact["manifest_sha256"],
            "replay_sha256": replay["manifest_sha256"],
            "call_spans": list(sequence.call_spans),
            "optimization_calls": selected,
        }
        _write_json(directory / "complete.json", summary)
        return {"sequence": selected_sequence, "origin": origin, "summary": summary, "prompt_length": prompt_length}

    def _batch(self, rows, requests):
        tensors = _pack_token_sequences(
            [row["sequence"] for row in rows], pad_token_id=self.pad_token_id, float_dtype=torch.get_default_dtype()
        )
        tensors["rewards"] = torch.tensor([row["summary"]["reward"] for row in rows], dtype=torch.float32)
        tensors["policy_origin_mask"] = torch.zeros_like(tensors["loss_mask"])
        for i, row in enumerate(rows):
            tensors["policy_origin_mask"][i, : len(row["sequence"].input_ids)] = row["origin"]
        summaries = np.empty(len(rows), dtype=object)
        summaries[:] = [row["summary"] for row in rows]
        return DataProto.from_dict(
            tensors=tensors,
            non_tensors={
                **{key: requests.non_tensor_batch[key].copy() for key in ("task_ids", "group_ids", "rollout_ids")},
                "rollout_metadata": summaries,
            },
            meta_info={
                **copy.deepcopy(requests.meta_info),
                "rollout_format": APPWORLD_GRPO_FORMAT,
                "selection_policy": self.selection_policy,
                "logprob_provenance": {
                    "behavior": {
                        "source": "vllm.actual_sampled_tokens",
                        "policy_version": requests.meta_info["policy_version"],
                    }
                },
            },
        )
