"""Integration checks for the locked TRL DPO adapter."""

import copy
import math
from importlib.metadata import PackageNotFoundError, version

import pytest
import torch

from lmflow.agentic import TRLDPOTrainer
from lmflow.datasets.dataset import Dataset
from lmflow.utils.conversation_template.qwen import QWEN3_TEMPLATE

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
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import DPOConfig

    return (
        Tokenizer,
        ByteLevelDecoder,
        BPE,
        ByteLevel,
        LoraConfig,
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
        DPOConfig,
    )


def _make_tokenizer_and_model():
    (
        tokenizer_class,
        decoder_class,
        bpe_class,
        pre_tokenizer_class,
        _,
        config_class,
        model_class,
        fast_tokenizer_class,
        _,
    ) = _load_backend()
    vocab = {token: index for index, token in enumerate(pre_tokenizer_class.alphabet())}
    vocab["<pad>"] = len(vocab)
    backend = tokenizer_class(bpe_class(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizer_class(add_prefix_space=False)
    backend.decoder = decoder_class()
    tokenizer = fast_tokenizer_class(
        tokenizer_object=backend,
        pad_token="<pad>",
        model_max_length=1024,
    )
    tokenizer.chat_template = QWEN3_TEMPLATE
    config = config_class(
        vocab_size=len(tokenizer),
        n_positions=1024,
        n_ctx=1024,
        n_embd=16,
        n_layer=1,
        n_head=2,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer, model_class(config)


def _make_dataset():
    prompt = [
        {"role": "user", "content": "Inspect the failure and propose the smallest fix."},
        {
            "role": "assistant",
            "content": "",
            "loss": False,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"pytest -q"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "bash",
            "content": "test_value FAILED",
        },
    ]
    tool = {
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

    def conversation(answer):
        return {
            "system": "You are a coding agent.",
            "tools": [tool],
            "messages": copy.deepcopy(prompt) + [{"role": "assistant", "content": answer}],
        }

    return Dataset.create_from_dict(
        {
            "type": "paired_conversation",
            "instances": [
                {
                    "chosen": conversation("GOOD_COMPLETION: fix the value calculation."),
                    "rejected": conversation("BAD_COMPLETION: change an unrelated comment."),
                }
            ],
        }
    )


def _make_args(tmp_path):
    *_, config_class = _load_backend()
    return config_class(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        max_steps=1,
        learning_rate=0.1,
        optim="sgd",
        max_grad_norm=0.0,
        gradient_checkpointing=False,
        beta=0.1,
        loss_type="sigmoid",
        max_length=1024,
        use_cpu=True,
        bf16=False,
        fp16=False,
        report_to="none",
        logging_strategy="no",
        save_strategy="no",
        eval_strategy="no",
        disable_tqdm=True,
        dataloader_pin_memory=False,
        seed=31,
    )


def test_trl_dpo_trainer_tokenizes_qwen3_and_updates_only_lora(tmp_path):
    *_, lora_config_class, _, _, _, _ = _load_backend()
    torch.manual_seed(31)
    tokenizer, model = _make_tokenizer_and_model()
    train_dataset = _make_dataset()
    source_tools = copy.deepcopy(train_dataset.get_backend_dataset()[0]["chosen"]["tools"])
    peft_config = lora_config_class(
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
        fan_in_fan_out=True,
    )
    adapter = TRLDPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=_make_args(tmp_path),
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    prepared = adapter._trainer.train_dataset[0]
    assert prepared["tools"] == source_tools
    expected_prompt_ids = tokenizer.apply_chat_template(
        prepared["prompt"],
        tools=source_tools,
        tokenize=True,
        add_generation_prompt=True,
    )["input_ids"]
    expected_chosen_ids = tokenizer.apply_chat_template(
        prepared["prompt"] + prepared["chosen"],
        tools=source_tools,
        tokenize=True,
    )["input_ids"]
    expected_rejected_ids = tokenizer.apply_chat_template(
        prepared["prompt"] + prepared["rejected"],
        tools=source_tools,
        tokenize=True,
    )["input_ids"]
    assert prepared["prompt_ids"] == expected_prompt_ids
    assert prepared["prompt_ids"] + prepared["chosen_ids"] == expected_chosen_ids
    assert prepared["prompt_ids"] + prepared["rejected_ids"] == expected_rejected_ids
    prompt_text = tokenizer.decode(prepared["prompt_ids"])
    chosen_text = tokenizer.decode(prepared["chosen_ids"])
    rejected_text = tokenizer.decode(prepared["rejected_ids"])
    assert "Inspect the failure" in prompt_text
    assert "test_value FAILED" in prompt_text
    assert "GOOD_COMPLETION" in chosen_text
    assert "BAD_COMPLETION" in rejected_text
    assert prepared["chosen_ids"]
    assert prepared["rejected_ids"]

    backend_model = adapter.unwrap_model()
    parameters_before = {name: parameter.detach().clone() for name, parameter in backend_model.named_parameters()}
    trainable_names = {name for name, parameter in backend_model.named_parameters() if parameter.requires_grad}
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)
    assert all("lora_" not in name for name in parameters_before if name not in trainable_names)

    metrics = adapter.train()

    assert math.isfinite(metrics["train_loss"])
    assert metrics["global_step"] == 1.0
    assert any(
        not torch.equal(parameter.detach(), parameters_before[name])
        for name, parameter in backend_model.named_parameters()
        if name in trainable_names
    )
    assert all(
        torch.equal(parameter.detach(), parameters_before[name])
        for name, parameter in backend_model.named_parameters()
        if name not in trainable_names
    )
