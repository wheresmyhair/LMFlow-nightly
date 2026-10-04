"""Minimal synchronous GRPO Pipeline over LMFlow Dataset and Model objects."""

from lmflow.agentic.trl_grpo_loop import build_synchronous_trl_grpo_trainer
from lmflow.pipeline.base_pipeline import BasePipeline


class GRPOPipeline(BasePipeline):
    """Run one native TRL lifecycle with replaceable task/rollout/reward functions.

    ``task_adapter(dataset)`` returns existing TaskSpec objects. ``rollout_fn``
    and ``publish_policy`` follow the backend-internal synchronous bridge contract.
    Rewards can be audited in the rollout result or supplied by ``reward_fn``.
    The input Model retains the trained backend and is returned to the caller.
    Native training metrics and publication history are available on ``trainer``.
    """

    def __init__(
        self,
        args,
        *,
        task_adapter,
        rollout_fn,
        publish_policy,
        policy_prefix,
        old_logprobs_source="behavior",
        reward_fn=None,
        peft_config=None,
        callbacks=None,
    ):
        self.args = args
        self.task_adapter = task_adapter
        self.rollout_fn = rollout_fn
        self.publish_policy = publish_policy
        self.policy_prefix = policy_prefix
        self.old_logprobs_source = old_logprobs_source
        self.reward_fn = reward_fn
        self.peft_config = peft_config
        self.callbacks = callbacks
        self.trainer = None
        self.train_result = None

    def __call__(self, model, dataset):
        if self.trainer is not None:
            raise RuntimeError("create a new Pipeline for a new run; automatic training retry is not supported")
        self.trainer = build_synchronous_trl_grpo_trainer(
            model.get_backend_model(),
            model.get_tokenizer(),
            self.args,
            list(self.task_adapter(dataset)),
            rollout_fn=self.rollout_fn,
            publish_policy=self.publish_policy,
            policy_prefix=self.policy_prefix,
            old_logprobs_source=self.old_logprobs_source,
            reward_fn=self.reward_fn,
            peft_config=self.peft_config,
            callbacks=self.callbacks,
        )
        model.backend_model = self.trainer.model
        self.train_result = self.trainer.train()
        return model
