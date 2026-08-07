# Agentic Python environments

LMFlow-Agent keeps training and vLLM rollout in separate Python 3.12
environments. Both profiles target Linux x86_64, CUDA 13.0, and NVIDIA driver
580 or newer.

These profiles are complete execution environments, not additive extras for
`requirements/base.txt`. The bootstrap installs LMFlow editable with
`--no-deps` after synchronizing the selected lock, so do not install the base
requirements into an Agentic environment separately.

The uv version used to generate the locks is pinned in `UV_VERSION`. uv is the
primary development path for LMFlow-Agent, but it is not an LMFlow runtime
dependency. The generated lock files remain installable with standard pip.

## Create environments

Install the uv version recorded in `UV_VERSION`, then run one of the following
profiles from the repository root.

Training:

```bash
uv venv --python 3.12 .venvs/agentic-train
uv pip sync --python .venvs/agentic-train/bin/python --require-hashes \
  requirements/agentic/lock/train-py312-cu130-linux-x86_64.txt
uv pip install --python .venvs/agentic-train/bin/python --no-deps -e .
```

Rollout:

```bash
uv venv --python 3.12 .venvs/agentic-rollout
uv pip sync --python .venvs/agentic-rollout/bin/python --require-hashes \
  requirements/agentic/lock/rollout-vllm-py312-cu130-linux-x86_64.txt
uv pip install --python .venvs/agentic-rollout/bin/python --no-deps -e .
```

`uv pip sync` removes packages that are absent from the selected lock. Use a
dedicated environment path for each profile; do not synchronize an unrelated
existing environment.

To use pip instead of uv after creating a Python 3.12 virtual environment:

```bash
python -m pip install --require-hashes \
  -r requirements/agentic/lock/train-py312-cu130-linux-x86_64.txt
python -m pip install --no-deps -e .
```

## Refresh locks

Lock regeneration is a coordinated maintainer operation. Each generated lock
records its exact `uv pip compile` command in the file header. Run that command
with the uv version from `UV_VERSION`, review the complete dependency diff, and
rerun the train/rollout compatibility matrix before committing an update.

`bitsandbytes`, `flash-attn`, `cpm_kernels`, and vLLM are deliberately absent
from the base training profile. Quantization and optional kernels require
separate compatibility checks before receiving their own profile.
