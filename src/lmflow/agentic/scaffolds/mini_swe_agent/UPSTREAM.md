# mini-swe-agent vendoring manifest

This package vendors the scaffold core from
[`SWE-agent/mini-swe-agent`](https://github.com/SWE-agent/mini-swe-agent) at
commit `a83fcae82d2a08f0ee0c688f9d137b3566c097f8` (`2.4.6`). The upstream code is
MIT licensed; see `LICENSES/mini-swe-agent-MIT.txt` and
`THIRD_PARTY_NOTICES.md` at the repository root.

Vendored upstream sources:

| Upstream source | SHA-256 | LMFlow destination |
|---|---|---|
| `src/minisweagent/agents/default.py` | `e8ef8aa365942d739c2ec5cb0879f60f377d2dc2de8ec670aaedf3bafb45a4c2` | `_vendor/agent.py` |
| `src/minisweagent/exceptions.py` | `0590393c56bee873c79a691dcb4f15cb39c85f1658598b84bfb295bfde56921d` | `_vendor/exceptions.py` |
| `src/minisweagent/utils/serialize.py` | `3035b27cfeffdf91117c72bc8131340bd9c5f67a3010baeb4ecd6b0c7406f462` | `_vendor/serialize.py` |
| `src/minisweagent/models/utils/actions_toolcall.py` | `47b666412e4f838508b50009cf34c7f63fe15e9e40affc257a17c0829896526f` | `_vendor/toolcalls.py` |
| `src/minisweagent/models/utils/openai_multimodal.py` | `c498ec2e97c0bf45c212a10de9f5d1b2d24d4de46a05fd0eaea975c1abe3f81f` | `_vendor/multimodal.py` |
| `src/minisweagent/environments/local.py` | `01dd33ae6be897458611911cc0eee1389092ea9fc6788c7eb24773e7c6342b1c` | `_vendor/local.py` |

Audited differences from upstream are limited to:

- LMFlow namespace imports;
- postponed annotations required by LMFlow's Python 3.9 package baseline;
- a side-effect-free protocol module in place of upstream startup/config loading;
- docstrings, exported names, and local variable names that do not alter control flow.

The provider and subprocess implementations are LMFlow adapters outside the
vendored directory. Initial acceptance ran the complete upstream tests for the
selected agent, tool-call, serialization, and local-environment sources: 98
tests passed. LMFlow CI retains only the critical contracts and deterministic
golden trajectories; upstream acceptance is repeated when the pinned commit
changes.
