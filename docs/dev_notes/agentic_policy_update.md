# Minimal causal-LM policy update

The first policy-update adapter connects a causal language model to the
existing GRPO objective without introducing a Trainer abstraction.

`causal_token_log_probs()` applies the causal shift, gathers the log-probability
of each target `input_ids` token, and returns a tensor with the same
`[batch, sequence]` shape. Column zero is filled with zero because no model
logit predicts the first token. Keeping full-sequence alignment lets
`loss_mask`, rollout `old_log_probs`, and model-computed log-probs share one
layout while prompt, environment, and padding tokens remain available as
context.

`grpo_loss_from_model()` reads `input_ids`, `attention_mask`, and `loss_mask`
from `DataProto`, runs the model, gathers aligned token log-probs, and delegates
the algorithm math to `grpo_policy_loss()`. It rejects a selected first token or
tokens hidden by `attention_mask`, because both indicate a broken training
alignment.

The adapter deliberately leaves optimizer ownership, gradient accumulation,
mixed precision, Accelerate/FSDP wrapping, checkpointing, and metric reduction
to the future `PolicyTrainer`. Required CPU tests use a tiny causal LM for a
real backward and optimizer update. The optional `trl==1.9.2` differential test
starts from identical model weights and compares scalar loss, every model
parameter gradient, and the post-SGD parameters.

The current reference helper computes a full-vocabulary `log_softmax` before
gathering selected token values. A later trainer optimization may replace this
with a chunked or fused selective-log-softmax implementation, but it must retain
the same alignment and pass the differential tests before becoming the default.
