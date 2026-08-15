"""Offline contract tests for local evaluation execution."""

import threading

import pytest

from lmflow.pipeline.evaluation.runtime import LocalEvaluationRuntime


def test_local_runtime_preserves_input_order_with_bounded_execution():
    runtime = LocalEvaluationRuntime(max_concurrency=2)

    result = runtime.map(lambda value: value * value, iter([3, 1, 2]))

    assert result == (9, 1, 4)
    assert runtime.provenance() == {"execution": "local", "max_concurrency": 2}


def test_local_runtime_does_not_eagerly_consume_the_entire_input_stream():
    runtime = LocalEvaluationRuntime(max_concurrency=2)
    release = threading.Event()
    started = threading.Event()
    consumed = []

    def items():
        for value in range(10):
            consumed.append(value)
            yield value

    def work(value):
        if len(consumed) >= 2:
            started.set()
        release.wait(timeout=1)
        return value

    thread = threading.Thread(target=lambda: runtime.map(work, items()))
    thread.start()
    assert started.wait(timeout=1)
    assert len(consumed) == 2
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_local_runtime_rejects_non_integer_concurrency(value):
    with pytest.raises(TypeError, match="must be an integer"):
        LocalEvaluationRuntime(max_concurrency=value)


def test_local_runtime_requires_positive_concurrency():
    with pytest.raises(ValueError, match="must be positive"):
        LocalEvaluationRuntime(max_concurrency=0)
