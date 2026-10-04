"""Offline CPU example: Dataset -> two-call rollout -> two native GRPO updates.

Run with ``PYTHONPATH=src python examples/grpo_tiny.py`` in the Agentic environment.
No download, external API, GPU or benchmark score is involved. Replace the task
adapter, rollout/reward functions and publication callback for a real environment.
"""

import copy
import hashlib
import tempfile
from pathlib import Path

import torch
from peft import LoraConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import GRPOConfig

from lmflow.agentic.contracts import TaskSpec
from lmflow.args import DatasetArguments, ModelArguments
from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.grpo_pipeline import GRPOPipeline
from lmflow.utils.protocol import DataProto


def publish_policy(trainer, version):
    digest = hashlib.sha256()
    for name, parameter in trainer.model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    # Sampling below uses this exact in-process model. External serving must
    # publish/load the weights and confirm its own loaded receipt instead.
    return dict(
        policy_version=version,
        global_step=trainer.state.global_step,
        weight_digest=digest.hexdigest(),
        source="live-hf-model",
    )


def rollout(requests, trainer):
    model = trainer.accelerator.unwrap_model(trainer.model)
    model.eval()
    rows, masks, probabilities = [], [], []
    with torch.no_grad():
        for task in requests.non_tensor_batch["tasks"]:
            prompt = trainer.processing_class.encode(task.messages[0]["content"], add_special_tokens=False)
            ids, mask, logprobs = list(prompt), [0] * len(prompt), [0.0] * len(prompt)
            for call in range(2):
                tokens = torch.tensor([ids], device=model.device)
                output = model.generate(
                    input_ids=tokens,
                    attention_mask=torch.ones_like(tokens),
                    do_sample=True,
                    max_new_tokens=2,
                    min_new_tokens=2,
                    eos_token_id=None,
                    pad_token_id=0,
                    top_k=0,
                    top_p=1.0,
                    temperature=1.0,
                    use_cache=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                sampled = output.sequences[0, len(ids) :].tolist()
                logprobs.extend(
                    float(score[0].log_softmax(-1)[token]) for score, token in zip(output.scores, sampled, strict=True)
                )
                ids.extend(sampled)
                mask.extend([1] * len(sampled))
                if call == 0:
                    ids.append(7)  # Toy environment observation; not a sampled model token.
                    mask.append(0)
                    logprobs.append(0.0)
            rows.append(ids)
            masks.append(mask)
            probabilities.append(logprobs)
    return DataProto.from_dict(
        tensors=dict(
            input_ids=torch.tensor(rows),
            attention_mask=torch.ones(len(rows), len(rows[0]), dtype=torch.long),
            loss_mask=torch.tensor(masks),
            old_log_probs=torch.tensor(probabilities),
            prompt_lengths=torch.tensor([2] * len(rows)),
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


def reward(data):
    """Toy outcome, intentionally unrelated to any benchmark or tool bonus."""
    return (data.batch["input_ids"][:, -1] % 2).float()


def run(directory):
    torch.manual_seed(43)
    torch.set_num_threads(2)
    root = Path(directory)
    initial = root / "initial"
    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "<bos>": 3,
        "prompt": 4,
        "one": 5,
        "action": 6,
        "observation": 7,
        "a": 8,
        "b": 9,
        "c": 10,
        "d": 11,
        "e": 12,
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>", unk_token="<unk>", pad_token="<pad>"
    )
    tokenizer.save_pretrained(initial)
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=13,
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            bos_token_id=3,
            eos_token_id=1,
            pad_token_id=0,
            use_cache=False,
        )
    ).save_pretrained(initial)
    dataset = Dataset(DatasetArguments(dataset_path=None)).from_dict(
        {"type": "text_only", "instances": [{"text": "prompt one"}]}
    )
    model = AutoModel.get_model(ModelArguments(model_name_or_path=str(initial), torch_dtype="float32"), device="cpu")
    args = GRPOConfig(
        output_dir=str(root / "trainer"),
        max_steps=2,
        num_generations=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        generation_batch_size=4,
        max_completion_length=5,
        learning_rate=0.01,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        beta=0.0,
        loss_type="grpo",
        scale_rewards="group",
        num_iterations=1,
        vllm_importance_sampling_correction=False,
        shuffle_dataset=False,
        use_cpu=True,
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
        task_adapter=lambda data: [
            TaskSpec(f"toy-{i}", [{"role": "user", "content": row["text"]}])
            for i, row in enumerate(data.to_dict()["instances"])
        ],
        rollout_fn=rollout,
        reward_fn=reward,
        publish_policy=publish_policy,
        policy_prefix="toy",
        peft_config=LoraConfig(
            task_type="CAUSAL_LM", r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["c_attn"], bias="none"
        ),
    )
    model = pipeline(model, dataset)
    return model, pipeline


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        _, pipeline = run(directory)
        print(
            {
                "global_step": pipeline.trainer.state.global_step,
                "policy": pipeline.trainer.lmflow_sync_bridge.final_publication,
                "training_loss": pipeline.train_result.training_loss,
            }
        )
