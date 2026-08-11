# mini-swe-agent rollout integration

LMFlow runs the pinned mini-swe-agent core behind two narrow adapters:

- `OpenAICompatibleCompletionBackend` calls a synchronous OpenAI-compatible
  Chat Completions endpoint, such as a vLLM server.
- `ProcessSandboxEnvironment` exposes the per-episode checkout through
  `ProcessSandbox` while preserving mini-swe-agent's Bash action and
  observation semantics.

`run_mini_swe_agent_episode()` composes those adapters with an
`EpisodeWorkspace`. It owns one rollout attempt from checkout creation through
patch export and cleanup, then atomically publishes `trajectory.json` and
`model.patch` in a new artifact directory. The
[SWE-bench adapter](agentic_swe_bench.md) prepares benchmark tasks, exports
official prediction records, and runs explicit verifier commands in fresh workspaces.
Source-repository acquisition, model-server lifecycle, scheduling, and official
SWE-bench grading remain caller responsibilities.

## Completion boundary

The backend requires an explicit `base_url` when it creates its own client. An
omitted API key becomes `EMPTY`, which is suitable for an unauthenticated local
vLLM server. Pass a preconfigured `openai.OpenAI` client when authentication,
custom HTTP transport, proxying, or external client ownership is required.
Configure credentials and sensitive headers on the backend or client. Do not
place secrets in `model_kwargs`, because the model configuration is included in
the episode trajectory; request-local `extra_headers` are rejected for this
reason.

The backend accepts one non-streaming choice per request. `model_kwargs` may
contain ordinary Chat Completions options and provider extensions such as
`extra_body`, but cannot override `model`, `messages`, or `tools`. The complete
provider response, including usage and provider-specific fields such as
`reasoning_content`, is stored in the raw trajectory.

```python
from lmflow.agentic.scaffolds.mini_swe_agent import (
    LMFlowMiniSWEAgentModel,
    OpenAICompatibleCompletionBackend,
)

backend = OpenAICompatibleCompletionBackend(
    base_url="http://127.0.0.1:8000/v1",
    timeout_seconds=120,
)
model = LMFlowMiniSWEAgentModel(
    backend,
    model_name="Qwen/Qwen3-8B",
    model_kwargs={
        "temperature": 0.0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
)
```

The serving endpoint must implement OpenAI-style function calling for the Bash
tool. Endpoint-specific model loading, chat-template, and tool-parser flags are
part of the frozen experiment configuration and are managed outside this
adapter.

This API is the scaffold control plane used for data generation. It reports
zero monetary cost because an OpenAI-compatible response does not define a
portable pricing contract. It also does not provide the exact prompt/output
token ids or sampled-token log-probabilities required by online RL. The
token-native vLLM rollout adapter is a separate data-plane capability.

## SFT export boundary

The raw `trajectory.json` and `model.patch` pair remains the authoritative
episode artifact. `mini_swe_agent_artifact_to_atif()` validates the pinned
mini-swe-agent format and projects its model-visible, text-only history into
ATIF v1.7. `mini_swe_agent_artifact_to_conversation()` then reuses LMFlow's ATIF
adapter to produce one conversation SFT example.

The projection keeps accepted assistant messages, reasoning, Bash tool calls,
and the rendered tool observations that were sent to later model requests.
Visible format-correction messages are also retained as user context. Raw
provider responses, sandbox diagnostics, the terminal `exit` control message,
and the patch stay in the source artifact rather than becoming prompt tokens.
Tool arguments are parsed as strict JSON and later serialized canonically by
the conversation adapter; consumers that need byte-exact provider output must
read the raw trajectory.

A successful submission commonly terminates while executing the final
model-generated Bash call. That call has no tool observation because no later
model request receives one. The ATIF conversation adapter permits this only for
the final trainable model step. Missing observations before any later
model-visible message still fail closed.

The initial exporter accepts only artifacts produced by LMFlow's pinned
mini-swe-agent model/environment adapters with multimodal expansion disabled.
Unsupported provider message fields, malformed or mismatched actions,
non-strict tool arguments, incomplete nonterminal observations, and version or
provenance mismatches are rejected. Batch discovery, dataset selection,
verification labels, and partial-publication policy remain separate data
pipeline responsibilities.
