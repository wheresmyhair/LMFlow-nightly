# Dependency manifests

All repository-level Python dependency manifests live under this directory.

- `base.txt` is the canonical dependency list for the LMFlow Python package.
  `setup.py` reads it for `pip install .` and editable installs.
- `agentic/` contains the self-contained training and rollout environment for
  LMFlow-Agent. It must not be layered on top of `base.txt`.

Most users should install LMFlow through `pip install -e .`. Agentic developers
should follow `agentic/README.md`.
