"""Lifecycle and private-contract checks for the sealed TRL GRPO bridge."""

import copy
import hashlib
import inspect
import math
from collections import Counter
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pytest
import torch

from lmflow.agentic.policy import grpo_loss_from_model
from lmflow.agentic.trl_grpo_trainer import build_one_step_trl_grpo_trainer
from lmflow.utils.protocol import DataProto

pytestmark = pytest.mark.optional_backend

_TRL_VERSION = "1.9.2"
_GENERATE_SCORE_SOURCE_SHA256 = "da3b7eb07b6398e7ae500646a582d159bf9abfccc8fe70134c70ebb857d90755"
_COMPUTE_LOSS_SOURCE_SHA256 = "9721cd3affc33b37b8089d7a41463dd864535861bfb42cec7c50984d25d3f3da"
_VOCAB = {
    "<pad>": 0,
    "<eos>": 1,
    "<unk>": 2,
    "<bos>": 3,
    "prompt_zero": 4,
    "prompt_one": 5,
    "call_zero": 6,
    "observation": 7,
    "answer_zero_good": 8,
    "answer_zero_bad": 9,
    "call_one": 10,
    "answer_one_good": 11,
    "answer_one_bad": 12,
}


def _load_backend():
    try:
        installed_version = version("trl")
    except PackageNotFoundError:
        pytest.skip(f"requires trl=={_TRL_VERSION}")
    if installed_version != _TRL_VERSION:
        pytest.skip(f"requires trl=={_TRL_VERSION}, found {installed_version}")

    from peft import LoraConfig
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    return (
        LoraConfig,
        Tokenizer,
        WordLevel,
        Whitespace,
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
        TrainerCallback,
        GRPOConfig,
        GRPOTrainer,
    )


def _make_tokenizer_and_model():
    (
        _,
        tokenizer_class,
        word_level,
        whitespace,
        config_class,
        model_class,
        fast_tokenizer,
        _,
        _,
        _,
    ) = _load_backend()
    backend = tokenizer_class(word_level(vocab=_VOCAB, unk_token="<unk>"))
    backend.pre_tokenizer = whitespace()
    tokenizer = fast_tokenizer(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    config = config_class(
        vocab_size=len(_VOCAB),
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=_VOCAB["<bos>"],
        eos_token_id=_VOCAB["<eos>"],
        pad_token_id=_VOCAB["<pad>"],
        use_cache=False,
    )
    return tokenizer, model_class(config).double()


def _completion_logprobs(model, prompt_ids, completion_ids):
    full_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids)).logits
        start = len(prompt_ids) - 1
        completion_logits = logits[:, start : start + len(completion_ids)]
        targets = torch.tensor([completion_ids], dtype=torch.long)
        return completion_logits.log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze().tolist()


def _sealed_rollouts(model, *, logprob_shift=0.25):
    rows = [
        ([3, 4], [6, 7, 8, 1], 1.0, "task-zero", 0, 0),
        ([3, 4], [6, 7, 9, 1], 0.0, "task-zero", 0, 1),
        ([3, 5], [10, 7, 12, 1], 0.0, "task-one", 1, 2),
        ([3, 5], [10, 7, 11, 1], 1.0, "task-one", 1, 3),
    ]
    input_ids = torch.tensor([prompt + completion for prompt, completion, *_ in rows])
    old_log_probs = torch.zeros(input_ids.shape, dtype=torch.float32)
    for index, (prompt, completion, *_) in enumerate(rows):
        old_log_probs[index, len(prompt) :] = torch.tensor(
            [value + logprob_shift for value in _completion_logprobs(model, prompt, completion)]
        )
    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "loss_mask": torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0, 1.0]] * 4),
            "old_log_probs": old_log_probs,
            "prompt_lengths": torch.tensor([2, 2, 2, 2]),
            "rewards": torch.tensor([row[2] for row in rows]),
        },
        non_tensors={
            "task_ids": np.asarray([row[3] for row in rows]),
            "group_ids": np.asarray([row[4] for row in rows]),
            "rollout_ids": np.asarray([row[5] for row in rows]),
        },
        meta_info={
            "policy_version": "tiny-policy@initial",
            "logprob_provenance": {
                "behavior": {
                    "source": "test.sampled-token-logprobs",
                    "policy_version": "tiny-policy@initial",
                }
            },
        },
    )


def _make_args(tmp_path):
    *_, config_class, _ = _load_backend()
    return config_class(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        generation_batch_size=4,
        num_generations=2,
        max_completion_length=4,
        max_steps=1,
        learning_rate=0.01,
        max_grad_norm=0.0,
        lr_scheduler_type="constant",
        warmup_steps=0,
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        beta=0.0,
        loss_type="grpo",
        scale_rewards="group",
        importance_sampling_level="token",
        num_iterations=1,
        vllm_importance_sampling_correction=False,
        shuffle_dataset=False,
        seed=20260831,
        data_seed=20260831,
        use_cpu=True,
        dataloader_pin_memory=False,
        logging_strategy="no",
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )


def test_locked_trl_private_source_contract():
    *_, trainer_class = _load_backend()

    assert list(inspect.signature(trainer_class._generate_and_score_completions).parameters) == [
        "self",
        "inputs",
    ]
    source_hashes = {
        name: hashlib.sha256(inspect.getsource(getattr(trainer_class, name)).encode()).hexdigest()
        for name in ("_generate_and_score_completions", "_compute_loss")
    }
    assert source_hashes == {
        "_generate_and_score_completions": _GENERATE_SCORE_SOURCE_SHA256,
        "_compute_loss": _COMPUTE_LOSS_SOURCE_SHA256,
    }


def test_standard_train_lifecycle_consumes_behavior_old_logprobs_and_updates_only_lora(tmp_path):
    lora_config_class, *_, callback_class, _, _ = _load_backend()
    torch.manual_seed(20260831)
    tokenizer, model = _make_tokenizer_and_model()
    sealed_rollouts = _sealed_rollouts(model)

    class LifecycleCallback(callback_class):
        def __init__(self):
            self.events = Counter()
            self.final_gradients = {}

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            self.events["pre_optimizer_step"] += 1
            self.final_gradients = {
                name: value.grad.detach().cpu().clone()
                for name, value in model.named_parameters()
                if value.requires_grad and value.grad is not None
            }

        def on_optimizer_step(self, args, state, control, **kwargs):
            self.events["optimizer_step"] += 1

        def on_step_end(self, args, state, control, **kwargs):
            self.events["step_end"] += 1

    callback = LifecycleCallback()
    peft_config = lora_config_class(
        task_type="CAUSAL_LM",
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        bias="none",
    )
    trainer = build_one_step_trl_grpo_trainer(
        model,
        tokenizer,
        _make_args(tmp_path),
        sealed_rollouts,
        old_logprobs_source="behavior",
        peft_config=peft_config,
        callbacks=[callback],
    )
    generated_batches = []
    loss_inputs = []
    original_generate = trainer._generate_and_score_completions
    original_compute_loss = trainer._compute_loss

    def audited_generate(inputs):
        output = original_generate(inputs)
        generated_batches.append(
            {key: value.detach().cpu().clone() for key, value in output.items() if isinstance(value, torch.Tensor)}
        )
        return output

    def audited_compute_loss(model, inputs):
        loss_inputs.append(
            {
                "sampling": inputs["sampling_per_token_logps"].detach().cpu().clone(),
                "old": inputs["old_per_token_logps"].detach().cpu().clone(),
                "has_reference": "ref_per_token_logps" in inputs,
                "gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing", False)),
            }
        )
        return original_compute_loss(model, inputs)

    trainer._generate_and_score_completions = audited_generate
    trainer._compute_loss = audited_compute_loss
    parameters_before = {name: value.detach().clone() for name, value in trainer.model.named_parameters()}
    trainable_names = {name for name, value in trainer.model.named_parameters() if value.requires_grad}
    expected_advantages = torch.tensor([0.7070068, -0.7070068, -0.7070068, 0.7070068])
    oracle_model = copy.deepcopy(trainer.model)
    oracle_data = DataProto.from_dict(
        tensors={
            "input_ids": sealed_rollouts.batch["input_ids"].clone(),
            "attention_mask": sealed_rollouts.batch["attention_mask"].clone(),
            "loss_mask": sealed_rollouts.batch["loss_mask"].clone(),
            "old_log_probs": sealed_rollouts.batch["old_log_probs"].clone(),
            "advantages": expected_advantages.clone(),
        }
    )
    oracle_model.zero_grad(set_to_none=True)
    oracle_loss = grpo_loss_from_model(oracle_model, oracle_data)
    oracle_loss.backward()
    oracle_gradients = {
        name: value.grad.detach().cpu().clone()
        for name, value in oracle_model.named_parameters()
        if value.requires_grad and value.grad is not None
    }

    result = trainer.train()

    parameters_after = {name: value.detach() for name, value in trainer.model.named_parameters()}
    assert trainer.state.global_step == 1
    assert math.isfinite(result.training_loss)
    assert result.training_loss == pytest.approx(float(oracle_loss.detach()), abs=1e-6, rel=1e-6)
    assert callback.events == Counter({"pre_optimizer_step": 1, "optimizer_step": 1, "step_end": 1})
    assert len(generated_batches) == 1
    generated = generated_batches[0]
    assert "sampling_per_token_logps" in generated
    assert "old_per_token_logps" in generated
    assert "ref_per_token_logps" not in generated
    torch.testing.assert_close(generated["old_per_token_logps"], generated["sampling_per_token_logps"])
    torch.testing.assert_close(generated["sampling_per_token_logps"], sealed_rollouts.batch["old_log_probs"][:, 2:])
    torch.testing.assert_close(generated["completion_ids"], sealed_rollouts.batch["input_ids"][:, 2:])
    torch.testing.assert_close(generated["tool_mask"], sealed_rollouts.batch["loss_mask"][:, 2:])
    torch.testing.assert_close(generated["advantages"], expected_advantages, atol=1e-6, rtol=1e-6)
    assert len(loss_inputs) == 4
    assert all(not item["has_reference"] for item in loss_inputs)
    assert all(item["gradient_checkpointing"] for item in loss_inputs)
    for item in loss_inputs:
        torch.testing.assert_close(item["old"], item["sampling"])
    assert trainer.lmflow_old_logprobs_source == "behavior"
    assert trainer.lmflow_sealed_rollout_bridge._reward_consumed is True
    assert trainer.lmflow_logprob_provenance == {
        "behavior": {
            "source": "test.sampled-token-logprobs",
            "policy_version": "tiny-policy@initial",
        },
        "trainer_old": {
            "source": "behavior",
            "input_field": "DataProto.batch['old_log_probs']",
            "trl_field": "old_per_token_logps",
            "compatibility_contract": "trl==1.9.2:post-generate-score-injection",
        },
        "reference": {"enabled": False, "source": None, "reason": "beta=0"},
    }
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)
    assert set(callback.final_gradients) == set(oracle_gradients) == trainable_names
    for name in trainable_names:
        torch.testing.assert_close(callback.final_gradients[name], oracle_gradients[name], atol=1e-6, rtol=1e-6)
    assert any(not torch.equal(parameters_before[name], parameters_after[name]) for name in trainable_names)
    assert all(
        torch.equal(parameters_before[name], parameters_after[name])
        for name in parameters_before
        if name not in trainable_names
    )
    assert trainer.optimizer.state
    assert trainer.lr_scheduler.last_epoch == 1


def test_builder_rejects_non_grpo_config_before_trainer_construction():
    _, model = _make_tokenizer_and_model()
    sealed_rollouts = _sealed_rollouts(model)

    with pytest.raises(TypeError, match="GRPOConfig"):
        build_one_step_trl_grpo_trainer(
            model,
            processing_class=object(),
            args=object(),
            sealed_rollouts=sealed_rollouts,
            old_logprobs_source="behavior",
        )


@pytest.mark.parametrize("zero_mask_member", [False, True])
def test_continuous_train_samples_updated_policy_and_keeps_native_optimizer(tmp_path, zero_mask_member):
    from lmflow.agentic.contracts import TaskSpec
    from lmflow.agentic.trl_grpo_loop import build_synchronous_trl_grpo_trainer

    lora_config_class, *_, callback_class, _, _ = _load_backend()
    torch.manual_seed(43)
    tokenizer, model = _make_tokenizer_and_model()
    args = _make_args(tmp_path)
    args.max_steps = 2
    args.num_generations = 4
    args.max_completion_length = 5
    args.lr_scheduler_type = "linear"
    publications, sampled, consumed, updates = [], [], [], []

    def digest(model):
        value = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                value.update(name.encode())
                value.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return value.hexdigest()

    def publish(trainer, version):
        receipt = dict(
            policy_version=version,
            global_step=trainer.state.global_step,
            weight_digest=digest(trainer.model),
            source="live-hf-policy",
        )
        publications.append(receipt)
        return receipt

    def generate(model, prefix):
        ids = torch.tensor([prefix])
        output = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=True,
            max_new_tokens=2,
            min_new_tokens=2,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            eos_token_id=None,
            pad_token_id=0,
            use_cache=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        tokens = output.sequences[0, len(prefix) :].tolist()
        probabilities = [
            float(score[0].log_softmax(-1)[token]) for score, token in zip(output.scores, tokens, strict=True)
        ]
        return tokens, probabilities

    def rollout(requests, trainer):
        live = trainer.accelerator.unwrap_model(trainer.model)
        live.eval()
        before = digest(live)
        rows, probabilities, rewards = [], [], []
        with torch.no_grad():
            for _ in range(len(requests)):
                first, lp1 = generate(live, [3, 4])
                # An environment observation remains conditioning, with zero loss.
                second, lp2 = generate(live, [3, 4] + first + [7])
                rows.append([3, 4] + first + [7] + second)
                probabilities.append([0.0, 0.0] + lp1 + [0.0] + lp2)
                rewards.append(float(second[-1] % 2))
        masks = [[0, 0, 1, 1, 0, 1, 1] for _ in rows]
        if zero_mask_member:
            masks[0] = [0] * 7
            rewards = [0.0, 1.0, 1.0, 1.0]
        sampled.append(
            dict(
                step=trainer.state.global_step,
                digest=before,
                optimizer=id(trainer.optimizer),
                scheduler=id(trainer.lr_scheduler),
                ids=copy.deepcopy(rows),
                logprobs=copy.deepcopy(probabilities),
                masks=copy.deepcopy(masks),
            )
        )
        return DataProto.from_dict(
            tensors=dict(
                input_ids=torch.tensor(rows),
                attention_mask=torch.ones(4, 7, dtype=torch.long),
                loss_mask=torch.tensor(masks),
                old_log_probs=torch.tensor(probabilities),
                prompt_lengths=torch.tensor([2] * 4),
                rewards=torch.tensor(rewards),
            ),
            non_tensors=copy.deepcopy(requests.non_tensor_batch),
            meta_info={
                **copy.deepcopy(requests.meta_info),
                "logprob_provenance": {
                    "behavior": dict(
                        source="hf.generate.output_scores", policy_version=requests.meta_info["policy_version"]
                    )
                },
            },
        )

    class Audit(callback_class):
        def on_step_end(self, args, state, control, model=None, optimizer=None, lr_scheduler=None, **kwargs):
            updates.append(
                dict(
                    step=state.global_step,
                    digest=digest(model),
                    optimizer=id(optimizer),
                    scheduler=id(lr_scheduler),
                    optimizer_steps=[int(s["step"]) for s in optimizer.state.values()],
                    scheduler_step=lr_scheduler.last_epoch,
                )
            )

    trainer = build_synchronous_trl_grpo_trainer(
        model,
        tokenizer,
        args,
        [TaskSpec("tiny-tool", [])],
        rollout_fn=rollout,
        publish_policy=publish,
        policy_prefix="tiny",
        old_logprobs_source="behavior",
        peft_config=lora_config_class(
            task_type="CAUSAL_LM", r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["c_attn"], bias="none"
        ),
        callbacks=[Audit()],
    )
    original_loss = trainer._compute_loss

    def observe_loss(model, inputs):
        consumed.append(
            dict(
                step=trainer.state.global_step,
                completion=inputs["completion_ids"].detach().cpu().clone(),
                old=inputs["old_per_token_logps"].detach().cpu().clone(),
                sampled=inputs["sampling_per_token_logps"].detach().cpu().clone(),
                mask=inputs["tool_mask"].detach().cpu().clone(),
                training=model.training,
                gc=model.is_gradient_checkpointing,
            )
        )
        loss = original_loss(model, inputs)
        consumed[-1]["loss"] = loss.detach().item()
        consumed[-1]["advantage"] = inputs["advantages"].detach().cpu().clone()
        return loss

    trainer._compute_loss = observe_loss  # Test-only observation; native loss is unchanged.
    zero_row_gradient_norms = []

    def observe_gradient(gradient):
        if not consumed[-1]["mask"].any():
            zero_row_gradient_norms.append(gradient.norm().item())

    hooks = [p.register_hook(observe_gradient) for p in trainer.model.parameters() if p.requires_grad]
    try:
        result = trainer.train()
    finally:
        for hook in hooks:
            hook.remove()
    assert trainer.state.global_step == 2 and math.isfinite(result.training_loss)
    assert [row["step"] for row in sampled] == [0, 1]
    assert [row["global_step"] for row in publications] == [0, 1, 2]
    assert sampled[1]["digest"] == updates[0]["digest"] == publications[1]["weight_digest"]
    assert sampled[0]["digest"] != sampled[1]["digest"] != publications[2]["weight_digest"]
    assert len({row["optimizer"] for row in sampled + updates}) == 1
    assert len({row["scheduler"] for row in sampled + updates}) == 1
    assert [row["scheduler_step"] for row in updates] == [1, 2]
    assert all(set(row["optimizer_steps"]) == {row["step"]} for row in updates)
    assert [row["step"] for row in consumed] == [0] * 4 + [1] * 4
    for step in (0, 1):
        actual = consumed[step * 4 : (step + 1) * 4]
        assert Counter(tuple(row["completion"][0].tolist()) for row in actual) == Counter(
            tuple(ids[2:]) for ids in sampled[step]["ids"]
        )
        for row in actual:
            assert row["training"] and row["gc"]
            torch.testing.assert_close(row["old"], row["sampled"])
            match = next(i for i, ids in enumerate(sampled[step]["ids"]) if ids[2:] == row["completion"][0].tolist())
            assert row["mask"].tolist() == [sampled[step]["masks"][match][2:]]
            torch.testing.assert_close(row["sampled"][0], torch.tensor(sampled[step]["logprobs"][match][2:]))
        if zero_mask_member:
            zero_rows = [row for row in actual if not row["mask"].any()]
            assert len(zero_rows) == 1 and zero_rows[0]["loss"] == 0.0
            advantages = sorted(row["advantage"].item() for row in actual)
            assert advantages == pytest.approx([-1.5 / 1.0002] + [0.5 / 1.0002] * 3)
    if zero_mask_member:
        assert zero_row_gradient_norms and all(norm == 0.0 for norm in zero_row_gradient_norms)
    bridge = trainer.lmflow_sync_bridge
    assert bridge.completed_steps == 2 and bridge.active is None
    assert all(row["status"] == "updated" for row in bridge.history)
    assert bridge.final_publication == publications[-1]


def test_dataset_model_pipeline_example_and_native_adapter_reload(tmp_path):
    import runpy
    from pathlib import Path

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    root = Path(__file__).resolve().parents[3]
    example = runpy.run_path(str(root / "examples/grpo_tiny.py"))
    model, pipeline = example["run"](tmp_path)
    assert model.get_backend_model() is pipeline.trainer.model
    assert pipeline.trainer.state.global_step == 2
    assert [record["global_step"] for record in pipeline.trainer.lmflow_sync_bridge.history] == [0, 1]
    pipeline.trainer.save_model(str(tmp_path / "adapter"))
    restored = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(tmp_path / "initial"), tmp_path / "adapter"
    )
    ids = torch.tensor([[4, 5]])
    model.get_backend_model().eval()
    restored.eval()
    with torch.no_grad():
        expected = model.get_backend_model()(input_ids=ids, use_cache=False).logits
        actual = restored(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
