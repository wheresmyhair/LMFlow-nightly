"""Synthetic CPU protocol tests; these are not AppWorld benchmark results."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lmflow.agentic.appworld_grpo import AppWorldTokenNativeRollout, appworld_grpo_tasks, official_test_fraction
from lmflow.agentic.appworld_protocol import canonical_appworld_sliced_instance_id
from lmflow.agentic.contracts import build_task_batch
from lmflow.agentic.scaffolds.appworld_react_code.scaffold import AppWorldPrompt
from lmflow.agentic.trl_grpo_trainer import _SealedRolloutBridge
from lmflow.datasets.dataset import Dataset


def render(messages, kwargs):
    """Toy token protocol with two observation tokens between sampled actions."""
    ids = [1, 2]
    for message in messages:
        if message["role"] == "assistant":
            token = 6 if "complete()" in message["content"] else (7 if "fail_again()" in message["content"] else 5)
            ids += [token, 8, 9]
    return ids


class Backend:
    def __init__(self, fail_at=None, corrupt=False, early_finish=False):
        self.calls = []
        self.fail_at = fail_at
        self.corrupt = corrupt
        self.early_finish = early_finish

    def complete(self, *, messages, tools, model_name, model_kwargs):
        self.calls.append(copy.deepcopy(messages))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("synthetic infrastructure failure")
        # Even seeds finish on call two; odd seeds produce a genuine failed task.
        recover = any(m["role"] == "assistant" for m in messages) and model_kwargs["seed"] % 2 == 0
        recover = recover or (self.early_finish and model_kwargs["seed"] % 2 == 1)
        token = 6 if recover else (7 if model_kwargs["seed"] % 2 else 5)
        code = {5: "fail()", 6: "complete()", 7: "fail_again()"}[token]
        content = "```python\n" + code + "\n```"
        prompt = render(messages, model_kwargs)
        if self.corrupt:
            prompt[0] = 11
        return {
            "message": {"role": "assistant", "content": content, "reasoning_content": "synthetic reasoning"},
            "finish_reason": "stop",
            "raw_response": {
                "id": "chatcmpl-" + model_kwargs["extra_body"]["request_id"],
                "prompt_token_ids": prompt,
                "choices": [
                    {
                        "token_ids": [token],
                        "finish_reason": "stop",
                        "logprobs": {"content": [{"token": f"token_id:{token}", "logprob": -1.0}]},
                    }
                ],
            },
        }


class World:
    def __init__(self, directory, *, replay_fault=False):
        self.task = SimpleNamespace(instruction="public instruction", supervisor={}, app_descriptions={})
        self.output_directory = str(directory)
        self.output_db_home_path_on_disk = str(directory / "dbs")
        Path(self.output_db_home_path_on_disk).mkdir(parents=True)
        self.requester = SimpleNamespace(request_tracker=SimpleNamespace(requests=[]))
        self.completed = False
        self.replay_fault = replay_fault

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, code):
        if code == "complete()":
            self.completed = True
            Path(self.output_db_home_path_on_disk, "state").write_text("changed")
            return "done"
        return "Execution failed. Traceback:\n" + ("different" if self.replay_fault else "bad action")

    def task_completed(self):
        return self.completed

    def evaluate(self, **kwargs):
        return SimpleNamespace(
            to_dict=lambda **kwargs: {
                "num_tests": 2,
                "success": self.completed,
                "passes": [{"requirement": "HIDDEN_VERIFIER"}] * (2 if self.completed else 1),
                "failures": [] if self.completed else [{"trace": "HIDDEN_VERIFIER"}],
            }
        )


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setenv("APPWORLD_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.scaffold_identity",
        lambda source: {"id": "synthetic-scaffold", "prompt_sha256": "0" * 64},
    )

    # The custom prompt must bypass the old one-shot loader, not globally patch it.
    def reject_reference(source):
        raise AssertionError("custom prompt unexpectedly loaded the reference one-shot")

    monkeypatch.setattr("lmflow.agentic.appworld_episode.load_reference_prompt", reject_reference)
    task_id = "82e2fac_1"
    spec = tmp_path / "data" / "tasks" / task_id / "specs.json"
    spec.parent.mkdir(parents=True)
    spec.write_text("{}")
    dataset = Dataset.create_from_dict(
        {
            "type": "text_only",
            "instances": [
                {
                    "text": "public instruction",
                    "task_id": task_id,
                    "source_split": "train",
                    "instance_id": canonical_appworld_sliced_instance_id(task_id, source_split="train"),
                    "task_spec_sha256": hashlib.sha256(b"{}").hexdigest(),
                }
            ],
        }
    )
    task = appworld_grpo_tasks(dataset)[0]
    requests = build_task_batch([task, task])
    requests.non_tensor_batch.update(group_ids=np.array([0, 0]), rollout_ids=np.array([0, 1]))
    receipt = dict(policy_version="test@0", global_step=0, weight_digest="weights-0", source="synthetic")
    requests.meta_info.update(policy_version="test@0", policy_publication=receipt)
    worlds = []

    def factory(**kwargs):
        world = World(tmp_path / "worlds" / kwargs["experiment_name"])
        worlds.append(world)
        return world

    prompt_text = "USER:\n{{ instruction }}"
    backend = Backend()
    config = dict(
        appworld_root=tmp_path,
        appworld_source=tmp_path,
        artifact_root=tmp_path / "artifacts",
        prompt=AppWorldPrompt(prompt_text, "synthetic-prompt/v1", hashlib.sha256(prompt_text.encode()).hexdigest()),
        prompt_token_ids_renderer=render,
        model_name="synthetic",
        model_revision="synthetic-r1",
        tokenizer_revision="synthetic-t1",
        token_protocol_identity={"id": "synthetic/v1"},
        model_kwargs={"temperature": 1.0, "seed": 0, "max_tokens": 2},
        max_model_len=32,
        max_steps=2,
        pad_token_id=0,
        world_factory=factory,
    )
    trainer = SimpleNamespace(
        args=SimpleNamespace(num_generations=2, max_completion_length=4), state=SimpleNamespace(global_step=0)
    )
    return SimpleNamespace(
        config=config,
        backend=backend,
        requests=requests,
        trainer=trainer,
        worlds=worlds,
        dataset=dataset,
        task=task,
        root=tmp_path,
    )


def producer(s, **kwargs):
    return AppWorldTokenNativeRollout(lambda receipt, trainer: (s.backend, receipt), **{**s.config, **kwargs})


def test_fresh_group_scalar_reward_masks_and_replay(setup):
    s = setup
    batch = producer(s)(s.requests, s.trainer)
    assert len(s.worlds) == 4 and len({id(w) for w in s.worlds}) == 4
    assert len(s.backend.calls) == 4
    assert batch.batch["rewards"].tolist() == [1.0, 0.5]
    assert batch.batch["input_ids"].tolist() == [[1, 2, 5, 8, 9, 6], [1, 2, 7, 8, 9, 7]]
    assert batch.batch["loss_mask"].tolist() == [[0, 0, 1, 0, 0, 1]] * 2
    assert torch.equal(batch.batch["loss_mask"], batch.batch["policy_origin_mask"])
    assert batch.batch["prompt_lengths"].tolist() == [2, 2]
    assert "HIDDEN_VERIFIER" not in str(batch.non_tensor_batch)
    assert "HIDDEN_VERIFIER" not in str(s.backend.calls)
    assert batch.non_tensor_batch["rollout_metadata"][1]["termination_reason"] == "max_steps"
    bridge = _SealedRolloutBridge(batch)
    assert bridge.num_generations == 2
    episode = next((s.root / "artifacts").glob("*/candidate-0000/episode.json"))
    assert json.loads(episode.read_text())["agent"]["prompt_identity"] == "synthetic-prompt/v1"
    with pytest.raises(FileExistsError):
        producer(s)(s.requests, s.trainer)
    assert len(s.backend.calls) == 4


def test_zero_optimization_member_preserves_origin_reward_and_group(setup):
    counter = iter(([True, True], [False, False]))
    batch = producer(setup, select_calls=lambda calls: next(counter), selection_policy="test-audit-member/v1")(
        setup.requests, setup.trainer
    )
    assert batch.batch["loss_mask"][1].sum() == 0
    assert batch.batch["policy_origin_mask"][1].sum() == 2
    assert batch.batch["old_log_probs"][1, 2] == -1
    assert _SealedRolloutBridge(batch).num_generations == 2


def test_variable_episode_lengths_are_right_padded(setup):
    setup.backend.early_finish = True
    batch = producer(setup)(setup.requests, setup.trainer)
    assert batch.batch["input_ids"][1].tolist() == [1, 2, 6, 0, 0, 0]
    assert batch.batch["attention_mask"][1].tolist() == [1, 1, 1, 0, 0, 0]
    assert batch.batch["old_log_probs"][1].tolist() == [0, 0, -1, 0, 0, 0]
    assert _SealedRolloutBridge(batch).num_generations == 2


@pytest.mark.parametrize("failure", ["backend", "replay", "tokens", "length", "receipt"])
def test_failure_keeps_partial_evidence_without_returning_group(setup, failure):
    s = setup
    if failure == "backend":
        s.backend.fail_at = 3
    elif failure == "replay":
        original = s.config["world_factory"]

        def factory(**kwargs):
            world = original(**kwargs)
            world.replay_fault = kwargs["experiment_name"].endswith("-replay")
            return world

        s.config["world_factory"] = factory
    elif failure == "tokens":
        s.backend.corrupt = True
    elif failure == "length":
        s.trainer.args.max_completion_length = 3  # Sampled=2, but full suffix=4.
    rollout = producer(s)
    if failure == "receipt":
        rollout.backend_for_policy = lambda receipt, trainer: (s.backend, {**receipt, "weight_digest": "stale"})
    with pytest.raises(ValueError):
        rollout(s.requests, s.trainer)
    batch_dir = next((s.root / "artifacts").iterdir())
    assert (batch_dir / "failure.json").is_file() and not (batch_dir / "complete.json").exists()
    if failure == "receipt":
        assert not s.backend.calls
    else:
        assert list(batch_dir.glob("candidate-*/requests/*-raw_response.json"))
        assert list(batch_dir.glob("candidate-*/episode.json"))
    if failure == "backend":
        assert json.loads((batch_dir / "failure.json").read_text())["completed_candidates"] == 1


def test_incomplete_group_rejected_before_publication_or_environment(setup):
    setup.trainer.args.num_generations = 4
    with pytest.raises(ValueError, match="complete K-group"):
        producer(setup)(setup.requests, setup.trainer)
    assert not setup.backend.calls and not setup.worlds


def test_new_policy_samples_fresh_candidates_and_old_claim_is_preserved(setup):
    s = setup
    rollout = producer(s)
    first = rollout(s.requests, s.trainer)
    s.requests.meta_info.update(
        policy_version="test@1",
        policy_publication={
            "policy_version": "test@1",
            "global_step": 1,
            "weight_digest": "weights-1",
            "source": "synthetic",
        },
    )
    s.requests.non_tensor_batch.update(group_ids=np.array([1, 1]), rollout_ids=np.array([2, 3]))
    s.trainer.state.global_step = 1
    second = rollout(s.requests, s.trainer)
    assert len(s.backend.calls) == 8 and len(s.worlds) == 8
    assert first.meta_info["policy_publication"]["weight_digest"] == "weights-0"
    assert second.meta_info["policy_publication"]["weight_digest"] == "weights-1"
    assert len(list((s.root / "artifacts").glob("*/complete.json"))) == 2


def test_reward_rejects_missing_or_infrastructure_outcome():
    with pytest.raises(ValueError, match="missing official"):
        official_test_fraction({})
    with pytest.raises(ValueError, match="infrastructure"):
        official_test_fraction({"runner_error": {"type": "TimeoutError"}})


def test_prompt_requires_exact_digest(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_bytes(b"USER:\nunchanged\r\n")
    prompt = AppWorldPrompt.from_file(path, identity="test/v1", sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert prompt.text.endswith("\r\n")
    with pytest.raises(ValueError, match="digest"):
        AppWorldPrompt.from_file(path, identity="test/v1", sha256="0" * 64)


@pytest.mark.optional_backend
@pytest.mark.gpu
def test_gpu_native_lifecycle_with_synthetic_appworld_actions(setup):
    """Real tiny Trainer updates, fake environment/actions; no benchmark claim."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed_version = version("trl")
    except PackageNotFoundError:
        pytest.skip("requires trl==1.15.0")
    if installed_version != "1.15.0":
        pytest.skip(f"requires trl==1.15.0, found {installed_version}")
    if not torch.cuda.is_available():
        pytest.skip("native TRL 1.15 fused scoring requires CUDA")

    from peft import LoraConfig
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import GRPOConfig

    from lmflow.pipeline.grpo_pipeline import GRPOPipeline

    s = setup
    torch.manual_seed(43)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(13)}, unk_token="t2")),
        pad_token="t0",
        eos_token="t1",
        unk_token="t2",
    )
    network = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=13,
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
            pad_token_id=0,
            eos_token_id=1,
        )
    )
    model = SimpleNamespace(get_backend_model=lambda: network, get_tokenizer=lambda: tokenizer)
    published, loaded = [], []

    def publish(trainer, version):
        digest = hashlib.sha256()
        for name, parameter in trainer.model.named_parameters():
            if parameter.requires_grad:
                digest.update(name.encode())
                digest.update(parameter.detach().float().cpu().numpy().tobytes())
        receipt = dict(
            policy_version=version,
            global_step=trainer.state.global_step,
            weight_digest=digest.hexdigest(),
            source="synthetic-actions-live-tiny-trainer",
        )
        published.append(receipt)
        return receipt

    def backend_for_policy(receipt, trainer):
        loaded.append(copy.deepcopy(receipt))
        # Scripted actions/probabilities test wiring only, not on-policy evidence.
        return s.backend, receipt

    rollout = AppWorldTokenNativeRollout(backend_for_policy, **s.config)
    args = GRPOConfig(
        output_dir=str(s.root / "trainer"),
        max_steps=2,
        num_generations=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        generation_batch_size=2,
        max_completion_length=4,
        learning_rate=0.01,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        beta=0.0,
        use_bias_correction_kl=False,
        loss_type="grpo",
        scale_rewards="group",
        num_iterations=1,
        vllm_importance_sampling_correction=False,
        shuffle_dataset=False,
        use_cpu=False,
        bf16=False,
        dataloader_pin_memory=False,
        seed=43,
        report_to="none",
        logging_strategy="no",
        save_strategy="no",
        disable_tqdm=True,
    )
    pipeline = GRPOPipeline(
        args,
        task_adapter=appworld_grpo_tasks,
        rollout_fn=rollout,
        publish_policy=publish,
        policy_prefix="synthetic",
        peft_config=LoraConfig(
            task_type="CAUSAL_LM", r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["c_attn"], bias="none"
        ),
    )
    try:
        assert pipeline(model, s.dataset) is model
        assert pipeline.trainer.state.global_step == 2
        assert pipeline.trainer.lr_scheduler.last_epoch == 2
        assert {int(value["step"]) for value in pipeline.trainer.optimizer.state.values()} == {2}
        assert [receipt["policy_version"] for receipt in loaded] == ["synthetic@0", "synthetic@1"]
        assert len({receipt["weight_digest"] for receipt in published}) == 3
        assert len(s.backend.calls) == 8 and len(s.worlds) == 8
        assert all(item["status"] == "updated" for item in pipeline.trainer.lmflow_sync_bridge.history)
        assert torch.isfinite(torch.tensor(pipeline.train_result.training_loss))
    finally:
        torch.set_num_threads(original_threads)
