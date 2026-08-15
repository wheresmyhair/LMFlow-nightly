# Third-party notices

## mini-swe-agent

LMFlow vendors selected scaffold-core files from
[`SWE-agent/mini-swe-agent`](https://github.com/SWE-agent/mini-swe-agent),
commit `a83fcae82d2a08f0ee0c688f9d137b3566c097f8`, under the MIT License.

Copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez

The complete license text is available at
`LICENSES/mini-swe-agent-MIT.txt`. The vendoring manifest and audited local
adaptations are documented in
`src/lmflow/agentic/scaffolds/mini_swe_agent/UPSTREAM.md`.

## AppWorld

The benchmark-specific adapter follows the Simplified ReAct Code scaffold from
[`StonyBrookNLP/appworld`](https://github.com/StonyBrookNLP/appworld), commit
`a072b7a86e7c1d5b1d7175659d750ebb9b79f10a`, under the Apache License 2.0.
LMFlow's root `LICENSE` contains that license text. Exact reference file hashes
and adapter differences are documented in
`src/lmflow/agentic/scaffolds/appworld_react_code/UPSTREAM.md`.

No AppWorld protected tasks, databases, ground truth, or verifier sources are
vendored. They are installed into an external cache by the explicit bootstrap
workflow and remain subject to AppWorld's data license.
