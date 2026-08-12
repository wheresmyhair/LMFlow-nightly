"""LMFlow's generic evaluation Pipeline."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lmflow.pipeline.evaluation.evaluator import Evaluator


def __getattr__(name: str) -> Any:
    if name == "Evaluator":
        from lmflow.pipeline.evaluation.evaluator import Evaluator

        return Evaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Evaluator"]
