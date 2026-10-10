# Agentic Python environment

The environment targets Linux x86_64, CUDA 13.0, and NVIDIA driver 580 or newer.
It is a complete execution environment, not an additive extra for
`requirements/base.txt`. LMFlow is installed editable with `--no-deps` after
the lock is synchronized.

## Install

The agentic environment uses PyTorch==2.11.0 required by vLLM==0.25.1. It also uses
NumPy==2.3.5 because vLLM's Numba dependency requires NumPy>=2.5.

```bash
uv venv --python 3.12 .venvs/agentic
uv pip sync --python .venvs/agentic/bin/python --require-hashes \
  requirements/agentic/lock/agentic-py312-cu130-linux-x86_64.txt
uv pip install --python .venvs/agentic/bin/python --no-deps -e .
```

To use standard pip after creating a Python 3.12 virtual environment:

```bash
python -m pip install --require-hashes \
  -r requirements/agentic/lock/agentic-py312-cu130-linux-x86_64.txt
python -m pip install --no-deps -e .
```

`uv pip sync` removes packages absent from the lock. Always use a
dedicated environment path; do not synchronize an unrelated existing
environment.

## Refresh locks

The uv version used to generate the locks is pinned in `UV_VERSION`. uv is the
primary maintainer path, but it is not an LMFlow runtime dependency. Each lock
records its exact `uv pip compile` command in the header and remains installable
with standard pip.

Lock regeneration is a coordinated change. Review the complete dependency diff
and rerun training, FSDP2, checkpoint, and vLLM smoke tests.

`bitsandbytes`, `flash-attn`, and `cpm_kernels` remain outside the default
profile until their combinations receive separate compatibility checks.

## TRL 1.15 upgrade boundary

The Agentic lock pins TRL 1.15.0. The migration retains all other package
versions, including Transformers 5.14.1, PyTorch 2.11.0, Triton 3.6.0, PEFT
0.20.0, Accelerate 1.14.0 and vLLM 0.25.1. Refresh just this package with
`uv pip compile --upgrade-package trl` and the existing hash-generation options;
review the resulting lock diff before synchronizing an isolated environment.

- Agentic `GRPOPipeline` and the sealed GRPO builders use native
  `GRPOTrainer.train()`, public `rollout_func`, and the existing small
  behavior-old log-probability bridge. The fixed GRPO objective and masks are
  preserved. Set `use_bias_correction_kl=False` explicitly when constructing
  `GRPOConfig`; its upstream default changed in 1.15.
- Agentic `TRLDPOTrainer` uses native sigmoid DPO with `DPOConfig` and
  `processing_class`. Paired-conversation projection remains on CPU; native
  DPO and GRPO scoring now use the upstream fused GPU/Triton path. LoRA on
  `lm_head` is not supported by upstream.
- `Finetuner` continues to use Transformers Trainer. Installing this lock does
  not enable TRL SFTTrainer or selective activation checkpointing for LMFlow SFT.
- Historical `TRLPolicyTrainer` remains locked to 1.9.2 and rejects this
  environment. It is not a fallback for the native lifecycle. Its optional
  historical differential tests may skip; those skips are not upgrade evidence.
- `setup.py`'s `trl` extra remains the separate legacy 0.11 profile.
  `DPOAligner` and `DPOv2Trainer` use old constructor arguments such as
  `tokenizer`, `beta`, and length limits outside `DPOConfig`. They are not
  supported by this Agentic lock. Do not install `lmflow[trl]` into it.

CPU CI continues to exclude `gpu` and `optional_backend`. Native upgrade
validation uses a local tiny CUDA model: sealed IDs/masks/rewards/behavior-old,
loss/gradient accumulation, continuous fresh sampling across two updates,
LoRA-only changes, adapter reload, and sigmoid DPO. This does not establish
large-model performance, distributed/FSDP2, external vLLM publication, or task
quality under 1.15. Those require separate bounded validation.

## AppWorld source and data

The lock includes the runtime and build dependencies declared by AppWorld
commit `a072b7a86e7c1d5b1d7175659d750ebb9b79f10a`. The AppWorld distribution
itself is installed in a second, deterministic step because its protected
source bundles are Git LFS objects and a VCS requirement cannot satisfy the
environment's `--require-hashes` policy.

After synchronizing the same agentic environment, run:

```bash
scripts/agentic/bootstrap_appworld.sh \
  --python .venvs/agentic/bin/python \
  --root "${XDG_CACHE_HOME:-${HOME}/.cache}/lmflow-agent/appworld-root-0.2.0-a072b7a"
```

The script checks the exact Git revision and all four LFS bundle digests,
installs AppWorld with `--no-deps --no-build-isolation`, unpacks its protected
application code, and downloads data version `0.2.0`. It prints the resulting
`APPWORLD_SOURCE` and `APPWORLD_ROOT` paths. Validate that installation with:

```bash
python -m lmflow.agentic.evaluate_appworld verify \
  --appworld-source "$APPWORLD_SOURCE" \
  --appworld-root "$APPWORLD_ROOT"
```

The stable `0.1.3.post1` package requires Pydantic 1, while the pinned source
revision supports Python 3.12 and Pydantic 2. `appworld-agents` is deliberately
excluded because its current OpenAI cap would downgrade the unified agentic
environment. LMFlow uses the verified official Simplified ReAct Code prompt
and loop semantics through a benchmark-local completion adapter.

AppWorld's protected tasks, databases, ground truth, verifier code, and raw
task artifacts must remain outside Git. Public redistribution must follow the
restrictions in the downloaded AppWorld data license; model training and local
evaluation do not make those files repository inputs.
