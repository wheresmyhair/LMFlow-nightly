"""Compose AppWorld with the native synchronous GRPO Pipeline.

Import ``build_pipeline`` from your experiment script after loading a Model and a
pinned Train Dataset. Provide actual serving publication/load callbacks; this
example does not start a GPU process or claim an endpoint has updated its weights.
See docs/agentic_appworld_grpo.md for the complete callback and budget contracts.
"""

from lmflow.agentic.appworld_grpo import AppWorldTokenNativeRollout, appworld_grpo_tasks
from lmflow.agentic.appworld_token_native import (
    qwen3_appworld_prompt_token_ids,
    qwen3_appworld_replay_chat_template_identity,
)
from lmflow.agentic.scaffolds.appworld_react_code.scaffold import AppWorldPrompt
from lmflow.pipeline.grpo_pipeline import GRPOPipeline


def build_pipeline(
    args,
    model,
    *,
    publish_policy,
    backend_for_policy,
    policy_prefix,
    appworld_root,
    appworld_source,
    artifact_root,
    prompt_file,
    prompt_identity,
    prompt_sha256,
    model_name,
    model_revision,
    tokenizer_revision,
    max_model_len,
    max_steps,
    max_tokens_per_call,
    seed,
    peft_config=None,
    callbacks=None,
):
    """Return a Pipeline; the caller then uses ``model = pipeline(model, dataset)``."""
    tokenizer = model.get_tokenizer()
    rollout = AppWorldTokenNativeRollout(
        backend_for_policy,
        appworld_root=appworld_root,
        appworld_source=appworld_source,
        artifact_root=artifact_root,
        prompt=AppWorldPrompt.from_file(prompt_file, identity=prompt_identity, sha256=prompt_sha256),
        prompt_token_ids_renderer=lambda messages, kwargs: qwen3_appworld_prompt_token_ids(tokenizer, messages, kwargs),
        model_name=model_name,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        token_protocol_identity=qwen3_appworld_replay_chat_template_identity(tokenizer),
        model_kwargs={
            "temperature": 1.0,
            "top_p": 1.0,
            "seed": seed,
            "max_completion_tokens": max_tokens_per_call,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        max_model_len=max_model_len,
        max_steps=max_steps,
        pad_token_id=tokenizer.pad_token_id,
    )
    return GRPOPipeline(
        args,
        task_adapter=appworld_grpo_tasks,
        rollout_fn=rollout,
        publish_policy=publish_policy,
        policy_prefix=policy_prefix,
        peft_config=peft_config,
        callbacks=callbacks,
    )
