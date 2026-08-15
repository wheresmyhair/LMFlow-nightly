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
