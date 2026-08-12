"""Default LMFlow evaluation pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lmflow.args import DatasetArguments, EvaluatorArguments, ModelArguments
from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.base_pipeline import BasePipeline
from lmflow.pipeline.evaluation.legacy import LegacyEvaluatorMixin
from lmflow.pipeline.evaluation.orchestration import evaluate_recipe
from lmflow.pipeline.evaluation.recipe import (
    EvaluationRecipe,
    LegacyEvaluationRecipe,
    ModelRunner,
)
from lmflow.pipeline.evaluation.result import EvaluationResult
from lmflow.pipeline.evaluation.runtime import EvaluationRuntime, LocalEvaluationRuntime
from lmflow.utils.envs import is_accelerate_env


class Evaluator(LegacyEvaluatorMixin, BasePipeline):
    """Evaluate a model and dataset through one configurable Pipeline entry."""

    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DatasetArguments,
        evaluator_args: EvaluatorArguments,
        *,
        recipe: EvaluationRecipe | None = None,
        runner: ModelRunner | None = None,
        runtime: EvaluationRuntime | None = None,
    ) -> None:
        self.data_args = data_args
        self.evaluator_args = evaluator_args
        self.model_args = model_args
        self.recipe = recipe
        self.runner = runner
        self.runtime = runtime
        self.accelerator = None
        self.config = None
        self.model_hidden_size = None
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.block_size = evaluator_args.evaluate_block_size
        self._legacy_initialized = False
        self._wandb = None

    def __call__(self, model, dataset: Dataset, **kwargs):
        """Evaluate through the standard ``pipeline(model, dataset)`` form."""

        return self.evaluate(model, dataset, **kwargs)

    def evaluate(
        self,
        model,
        dataset: Dataset,
        metric: str = "accuracy",
        verbose: bool = True,
        *,
        recipe: EvaluationRecipe | None = None,
        runner: ModelRunner | None = None,
        runtime: EvaluationRuntime | None = None,
    ):
        """Evaluate ``model`` on ``dataset`` using the configured recipe.

        Per-call recipe, runner, and runtime values are retained as advanced
        overrides. Existing scalar metrics remain available when no recipe is
        configured.
        """

        selected_recipe = recipe if recipe is not None else self.recipe
        selected_runner = runner if runner is not None else self.runner
        selected_runtime = runtime if runtime is not None else self.runtime
        if selected_recipe is not None:
            if selected_runner is None:
                raise ValueError("runner is required when recipe is provided")
            return self._evaluate_recipe(
                model=model,
                dataset=dataset,
                recipe=selected_recipe,
                runner=selected_runner,
                runtime=selected_runtime or LocalEvaluationRuntime(),
            )
        if selected_runner is not None:
            raise ValueError("recipe is required when runner is provided")
        if selected_runtime is not None:
            raise ValueError("recipe is required when runtime is provided")

        legacy_recipe = LegacyEvaluationRecipe(metric=metric)
        self._initialize_legacy_runtime()
        if legacy_recipe.metric in {"acc", "accuracy"}:
            if is_accelerate_env():
                value = self._evaluate_acc_with_accelerate(model, dataset, verbose=verbose)
            else:
                value = self._evaluate_acc_with_deepspeed(model, dataset, verbose=verbose)
            print(f"Evaluating final accuracy: {value}")
            return value
        if legacy_recipe.metric in {"ppl", "perplexity"}:
            value = self._evaluate_ppl(model, dataset, verbose=verbose)
            print(f"Evaluating final perplexity: {value}")
            return value

        value = self._evaluate_nll(model, dataset, verbose=verbose)
        print(f"Evaluating final negative log likelihood: {value}")
        return value

    def _evaluate_recipe(
        self,
        *,
        model: Any,
        dataset: Dataset,
        recipe: EvaluationRecipe,
        runner: ModelRunner,
        runtime: EvaluationRuntime,
    ) -> EvaluationResult:
        """Run a recipe without initializing the legacy distributed runtime."""

        model_provenance = {
            "model_name_or_path": self.model_args.model_name_or_path,
            "model_revision": self.model_args.model_revision,
            "tokenizer_name": self.model_args.tokenizer_name,
            "lora_model_path": self.model_args.lora_model_path,
        }
        dataset_path = self.data_args.dataset_path
        dataset_provenance = {
            "dataset_type": dataset.get_type(),
            "dataset_name": self.data_args.dataset_name,
            "dataset_config_name": self.data_args.dataset_config_name,
            "dataset_path": None if dataset_path is None else str(Path(dataset_path)),
            "dataset_fingerprint": self._dataset_fingerprint(dataset),
        }
        return evaluate_recipe(
            recipe,
            runner,
            runtime,
            model,
            dataset,
            model_provenance=model_provenance,
            dataset_provenance=dataset_provenance,
        )

    @staticmethod
    def _dataset_fingerprint(dataset: Dataset) -> str | None:
        try:
            fingerprint = dataset.get_fingerprint()
        except (AttributeError, TypeError, ValueError):
            return None
        return fingerprint if isinstance(fingerprint, str) else None


__all__ = ["Evaluator"]
