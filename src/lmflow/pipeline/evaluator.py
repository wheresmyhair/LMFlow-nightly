"""Compatibility import for the evaluation Pipeline."""

import warnings

from lmflow.pipeline.evaluation.evaluator import Evaluator

warnings.warn(
    "lmflow.pipeline.evaluator is deprecated; import Evaluator from lmflow.pipeline.evaluation instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Evaluator"]
