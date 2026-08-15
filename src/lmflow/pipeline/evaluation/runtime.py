"""Execution runtimes used by the Evaluator pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Protocol, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class EvaluationRuntime(Protocol):
    """Execute independent evaluation work without owning dataset semantics."""

    def map(
        self,
        function: Callable[[InputT], OutputT],
        items: Iterable[InputT],
    ) -> tuple[OutputT, ...]: ...


class LocalEvaluationRuntime:
    """Synchronous reference runtime with optional bounded thread concurrency."""

    def __init__(self, *, max_concurrency: int = 1) -> None:
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int):
            raise TypeError("max_concurrency must be an integer")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self.max_concurrency = max_concurrency

    def map(
        self,
        function: Callable[[InputT], OutputT],
        items: Iterable[InputT],
    ) -> tuple[OutputT, ...]:
        if self.max_concurrency == 1:
            return tuple(function(item) for item in items)

        indexed_items = enumerate(items)
        results: dict[int, OutputT] = {}
        with ThreadPoolExecutor(
            max_workers=self.max_concurrency,
            thread_name_prefix="lmflow-evaluator",
        ) as executor:
            in_flight: dict[Future[OutputT], int] = {}

            def submit_next() -> bool:
                try:
                    index, item = next(indexed_items)
                except StopIteration:
                    return False
                in_flight[executor.submit(function, item)] = index
                return True

            for _ in range(self.max_concurrency):
                if not submit_next():
                    break
            while in_flight:
                completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in completed:
                    index = in_flight.pop(future)
                    results[index] = future.result()
                    submit_next()

        return tuple(results[index] for index in range(len(results)))

    def provenance(self) -> dict[str, int | str]:
        return {
            "execution": "local",
            "max_concurrency": self.max_concurrency,
        }


__all__ = ["EvaluationRuntime", "LocalEvaluationRuntime"]
