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
