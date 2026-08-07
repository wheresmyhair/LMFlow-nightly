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

From the repository root:

```bash
scripts/agentic/bootstrap_env.sh train
scripts/agentic/bootstrap_env.sh rollout
```

The default locations are `.venvs/agentic-train` and
`.venvs/agentic-rollout`. Override them with `--env-dir`.
Bootstrap-created environments carry a profile ownership marker; the script
refuses to synchronize an unrelated existing virtual environment because sync
removes packages that are absent from the selected lock.

To use pip instead of uv after creating a Python 3.12 virtual environment:

```bash
python -m pip install --require-hashes \
  -r requirements/agentic/lock/train-py312-cu130-linux-x86_64.txt
python -m pip install --no-deps -e .
```

## Refresh locks

```bash
scripts/agentic/lock_envs.sh
```

Routine regeneration preserves existing pins. Use
`scripts/agentic/lock_envs.sh --upgrade` only for a coordinated dependency
upgrade and rerun the complete train/rollout compatibility matrix.

`bitsandbytes`, `flash-attn`, `cpm_kernels`, and vLLM are deliberately absent
from the base training profile. Quantization and optional kernels require
separate compatibility checks before receiving their own profile.
