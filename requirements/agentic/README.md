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
