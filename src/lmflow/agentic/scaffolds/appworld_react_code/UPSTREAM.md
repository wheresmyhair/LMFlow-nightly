# AppWorld Simplified ReAct Code scaffold provenance

This benchmark adapter follows AppWorld's official `SimplifiedReActCodeAgent`
at commit `a072b7a86e7c1d5b1d7175659d750ebb9b79f10a` (`0.2.0.dev0`). AppWorld is
Apache-2.0 licensed; LMFlow's root `LICENSE` contains the applicable license
text. No AppWorld protected task, database, ground-truth, or verifier material
is stored in this repository.

Reference sources:

| Upstream source | SHA-256 |
|---|---|
| `experiments/code/simplified/agent.py` | `e48934f66a23d00e32babbf8759343d21c277cc722e874b1d25e2a9bfc8fe017` |
| `experiments/code/simplified/react_code_agent.py` | `4b80c27e61b5d04859c447fa47e525025f2c10044a8799ffad650f42fbfc5963` |
| `experiments/prompts/react_code_agent/instructions.txt` | `c41f7852217d46586047a38f68e88b827cb9dcbe624e9651f9db8301547a534b` |
| `experiments/configs/_generator/templates/simplified_react_code_agent.jsonnet.j2` | `efcdebcf10f4ccd534619a7ff1f5a6e0fcf3d038cd0904bd0ac005a7bc673166` |
| `experiments/configs/_generator/models/alibaba.py` | `357a30bc9febad027db2d01f8dde1e80aa1b9e1b34587420762357f8c0babed7` |

LMFlow keeps the official one-code-block-per-step loop, prompt rendering,
first-complete-code-block parsing, observation framing, 50-step limit,
AppWorld random seed 100, and the official Qwen3-8B no-reasoning sampling
profile. The local adapter replaces `appworld-agents`' model client and logger
with LMFlow's existing completion backend and benchmark-specific provenance,
failure, state-digest, and projection records. The prompt is read from a
verified checkout instead of copied into LMFlow, so the `appworld-agents`
package and its incompatible OpenAI dependency cap are not installed.
