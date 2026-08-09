"""Integration checks for the locked TRL policy-trainer adapter."""

import copy
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pytest
import torch

from lmflow.agentic.grpo_recipe import run_grpo_step
from lmflow.agentic.policy import causal_token_log_probs, grpo_loss_from_model
from lmflow.agentic.trl_policy_trainer import TRLPolicyTrainer
from lmflow.utils.protocol import DataProto

pytestmark = pytest.mark.optional_backend

_TRL_VERSION = "1.9.2"


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
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import GRPOConfig

    return Tokenizer, WordLevel, LoraConfig, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, GRPOConfig


def _make_tokenizer_and_model():
    (
        tokenizer_class,
        word_level,
        _,
        config_class,
        model_class,
        fast_tokenizer,
        _,
    ) = _load_backend()
    tokenizer_backend = tokenizer_class(
        word_level({f"token-{index}": index for index in range(16)}, unk_token="token-0")
    )
    tokenizer = fast_tokenizer(
        tokenizer_object=tokenizer_backend,
        unk_token="token-0",
        pad_token="token-1",
        eos_token="token-2",
    )
    model_config = config_class(
        vocab_size=16,
        n_positions=16,
        n_embd=8,
        n_layer=1,
        n_head=1,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
    )
    return tokenizer, model_class(model_config).double()


def _make_data(model):
    input_ids = torch.tensor([[3, 4, 5, 6], [3, 7, 8, 9]])
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        old_log_probs = causal_token_log_probs(logits, input_ids)
    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "advantages": advantages,
            "old_log_probs": old_log_probs,
        }
    )


def _make_args(tmp_path, *, learning_rate=0.05):
    *_, grpo_config = _load_backend()
    return grpo_config(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        learning_rate=learning_rate,
        optim="sgd",
        max_grad_norm=0.0,
        gradient_checkpointing=False,
        loss_type="grpo",
        beta=0.0,
        use_cpu=True,
        report_to="none",
        disable_tqdm=True,
    )


def test_trl_policy_trainer_step_matches_lmflow_objective(tmp_path):
    torch.manual_seed(23)
    tokenizer, base_model = _make_tokenizer_and_model()
    native_model = copy.deepcopy(base_model)
    trl_model = copy.deepcopy(base_model)
    data = _make_data(base_model)

    learning_rate = 0.05
    native_optimizer = torch.optim.SGD(native_model.parameters(), lr=learning_rate)
    native_loss = grpo_loss_from_model(native_model, data)
    native_loss.backward()
    native_optimizer.step()

    args = _make_args(tmp_path, learning_rate=learning_rate)
    adapter = TRLPolicyTrainer(trl_model, tokenizer, args)

    metrics = adapter.train_step(data)

    assert metrics["loss"] == pytest.approx(native_loss.detach().item(), rel=1e-5, abs=1e-6)
    assert metrics["selected_tokens"] == 4.0
    native_parameters = list(native_model.parameters())
    trl_parameters = list(adapter.unwrap_model().parameters())
    assert len(native_parameters) == len(trl_parameters)
    for native_parameter, trl_parameter in zip(native_parameters, trl_parameters):
        torch.testing.assert_close(native_parameter, trl_parameter, rtol=1e-5, atol=1e-6)


def test_trl_policy_trainer_updates_only_lora_parameters(tmp_path):
    *_, lora_config_class, _, _, _, _ = _load_backend()
    torch.manual_seed(29)
    tokenizer, base_model = _make_tokenizer_and_model()
    data = _make_data(base_model)
    peft_config = lora_config_class(
        r=2,
        lora_alpha=4,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
        fan_in_fan_out=True,
    )
    adapter = TRLPolicyTrainer(
        copy.deepcopy(base_model),
        tokenizer,
        _make_args(tmp_path),
        peft_config=peft_config,
    )
    del data.batch["advantages"]
    data.batch["rewards"] = torch.tensor([1.0, 3.0], dtype=torch.float64)
    data.non_tensor_batch["group_ids"] = np.asarray(["group", "group"], dtype=object)
    model = adapter.unwrap_model()
    parameters_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    metrics = run_grpo_step(adapter, data)

    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)
    assert metrics["train/tokens"] == 4.0
    assert metrics["rollout/trajectories"] == 2.0
    assert metrics["rollout/groups"] == 1.0
    torch.testing.assert_close(
        data.batch["advantages"],
        torch.tensor([-0.7071, 0.7071], dtype=torch.float64),
        rtol=1e-4,
        atol=1e-4,
    )
    assert any(
        not torch.equal(parameter.detach(), parameters_before[name])
        for name, parameter in model.named_parameters()
        if name in trainable_names
    )
    assert all(
        torch.equal(parameter.detach(), parameters_before[name])
        for name, parameter in model.named_parameters()
        if name not in trainable_names
    )
