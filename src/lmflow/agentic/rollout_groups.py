"""Controller-local assembly of complete rollout groups."""

import math
import time
from collections import deque
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Optional

from lmflow.utils.protocol import DataProto


@dataclass
class _PendingGroup:
    rollout_ids: tuple[Hashable, ...]
    started_at: float
    rows: dict[Hashable, DataProto] = field(default_factory=dict)
    first_received_at: Optional[float] = None


def _require_hashable(value: Any, label: str) -> Hashable:
    if not isinstance(value, Hashable):
        raise TypeError(f"{label} must be hashable, got {type(value).__name__}")
    return value


class RolloutGroupAssembler:
    """Buffer rollout rows until their registered groups are complete."""

    def __init__(
        self,
        *,
        policy_version: Hashable,
        timeout_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.policy_version = _require_hashable(policy_version, "policy_version")
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be finite and positive, got {timeout_seconds}")

        self._clock = clock
        self._pending: dict[Hashable, _PendingGroup] = {}
        self._ready: deque[tuple[Hashable, DataProto]] = deque()
        self._closed_group_ids: set[Hashable] = set()
        self._registered_rollout_ids: set[Hashable] = set()
        self._received_rollout_ids: set[Hashable] = set()

        self._completed_groups = 0
        self._timed_out_groups = 0
        self._cancelled_groups = 0
        self._duplicate_results_rejected = 0
        self._policy_mismatch_results_rejected = 0
        self._completion_latency_total = 0.0
        self._straggler_wait_total = 0.0

    def _timestamp(self, value: Optional[float]) -> float:
        timestamp = float(self._clock() if value is None else value)
        if not math.isfinite(timestamp):
            raise ValueError(f"timestamp must be finite, got {timestamp}")
        return timestamp

    def register_group(
        self,
        group_id: Hashable,
        rollout_ids: Iterable[Hashable],
        *,
        started_at: Optional[float] = None,
    ) -> None:
        """Register the exact logical rollout requests expected for one group."""
        group_id = _require_hashable(group_id, "group_id")
        if group_id in self._pending or group_id in self._closed_group_ids:
            raise ValueError(f"group_id {group_id!r} has already been registered")

        expected_ids = tuple(rollout_ids)
        if not expected_ids:
            raise ValueError("rollout_ids must contain at least one identity")

        seen: set[Hashable] = set()
        for index, rollout_id in enumerate(expected_ids):
            rollout_id = _require_hashable(rollout_id, f"rollout_ids[{index}]")
            if rollout_id in seen:
                raise ValueError(f"duplicate rollout_id {rollout_id!r} within group {group_id!r}")
            if rollout_id in self._registered_rollout_ids:
                raise ValueError(f"rollout_id {rollout_id!r} has already been registered")
            seen.add(rollout_id)

        self._pending[group_id] = _PendingGroup(
            rollout_ids=expected_ids,
            started_at=self._timestamp(started_at),
        )
        self._registered_rollout_ids.update(expected_ids)

    def add(self, data: DataProto, *, received_at: Optional[float] = None) -> tuple[Hashable, ...]:
        """Add rollout results atomically and enqueue every newly complete group."""
        if len(data) == 0:
            raise ValueError("rollout result batch must contain at least one row")

        try:
            actual_policy_version = data.meta_info["policy_version"]
        except KeyError as exc:
            raise KeyError("rollout results require DataProto.meta_info['policy_version']") from exc

        if actual_policy_version != self.policy_version:
            self._policy_mismatch_results_rejected += len(data)
            raise ValueError(
                f"policy_version mismatch: expected {self.policy_version!r}, got {actual_policy_version!r}"
            )

        try:
            group_ids = data.non_tensor_batch["group_ids"]
            rollout_ids = data.non_tensor_batch["rollout_ids"]
        except KeyError as exc:
            raise KeyError("rollout results require non_tensor_batch['group_ids'] and ['rollout_ids']") from exc

        if group_ids.ndim != 1 or rollout_ids.ndim != 1:
            raise ValueError("group_ids and rollout_ids must both have shape (batch,)")

        timestamp = self._timestamp(received_at)
        incoming_ids: set[Hashable] = set()
        incoming_rows: dict[Hashable, dict[Hashable, DataProto]] = {}

        for index, (group_id, rollout_id) in enumerate(zip(group_ids, rollout_ids)):
            group_id = _require_hashable(group_id, f"group_ids[{index}]")
            rollout_id = _require_hashable(rollout_id, f"rollout_ids[{index}]")

            if rollout_id in incoming_ids or rollout_id in self._received_rollout_ids:
                self._duplicate_results_rejected += 1
                raise ValueError(f"duplicate rollout result {rollout_id!r}")
            if group_id in self._closed_group_ids:
                raise ValueError(f"group_id {group_id!r} is already closed")
            if group_id not in self._pending:
                raise KeyError(f"group_id {group_id!r} has not been registered")

            group = self._pending[group_id]
            if timestamp < group.started_at:
                raise ValueError(
                    f"received_at {timestamp} precedes started_at {group.started_at} for group {group_id!r}"
                )
            if rollout_id not in group.rollout_ids:
                raise ValueError(f"rollout_id {rollout_id!r} is not registered for group {group_id!r}")

            incoming_ids.add(rollout_id)
            incoming_rows.setdefault(group_id, {})[rollout_id] = data.select_idxs([index])

        completed_batches: dict[Hashable, DataProto] = {}
        for group_id, rows in incoming_rows.items():
            group = self._pending[group_id]
            candidate_rows = {**group.rows, **rows}
            if len(candidate_rows) == len(group.rollout_ids):
                completed_batches[group_id] = DataProto.concat(
                    [candidate_rows[rollout_id] for rollout_id in group.rollout_ids]
                )

        for group_id, rows in incoming_rows.items():
            group = self._pending[group_id]
            group.rows.update(rows)
            if group.first_received_at is None:
                group.first_received_at = timestamp
        self._received_rollout_ids.update(incoming_ids)

        completed_group_ids = tuple(group_id for group_id in self._pending if group_id in completed_batches)
        for group_id in completed_group_ids:
            group = self._pending.pop(group_id)
            self._closed_group_ids.add(group_id)
            self._ready.append((group_id, completed_batches[group_id]))
            self._completed_groups += 1
            self._completion_latency_total += timestamp - group.started_at
            self._straggler_wait_total += timestamp - group.first_received_at

        return completed_group_ids

    def pop_ready(self, max_groups: Optional[int] = None) -> Optional[DataProto]:
        """Consume up to ``max_groups`` complete groups as one DataProto batch."""
        if max_groups is not None:
            if isinstance(max_groups, bool) or not isinstance(max_groups, int):
                raise TypeError(f"max_groups must be an integer, got {type(max_groups).__name__}")
            if max_groups <= 0:
                raise ValueError(f"max_groups must be positive, got {max_groups}")

        count = len(self._ready) if max_groups is None else min(max_groups, len(self._ready))
        if count == 0:
            return None

        ready_groups = [batch for _, batch in islice(self._ready, 0, count)]
        output = ready_groups[0] if len(ready_groups) == 1 else DataProto.concat(ready_groups)
        for _ in range(count):
            self._ready.popleft()
        return output

    def expire(self, *, now: Optional[float] = None) -> tuple[Hashable, ...]:
        """Close and report every pending group whose deadline has elapsed."""
        timestamp = self._timestamp(now)
        for group_id, group in self._pending.items():
            if timestamp < group.started_at:
                raise ValueError(f"now {timestamp} precedes started_at {group.started_at} for group {group_id!r}")

        expired = tuple(
            group_id
            for group_id, group in self._pending.items()
            if timestamp - group.started_at >= self.timeout_seconds
        )
        for group_id in expired:
            del self._pending[group_id]
            self._closed_group_ids.add(group_id)
            self._timed_out_groups += 1
        return expired

    def cancel_group(self, group_id: Hashable) -> None:
        """Close one pending group so none of its late results can be consumed."""
        group_id = _require_hashable(group_id, "group_id")
        if group_id not in self._pending:
            raise KeyError(f"group_id {group_id!r} is not pending")
        del self._pending[group_id]
        self._closed_group_ids.add(group_id)
        self._cancelled_groups += 1

    def metrics(self) -> dict[str, float]:
        """Return local assembler counters and latency means in seconds."""
        completed = self._completed_groups
        return {
            "rollout/groups_completed": float(completed),
            "rollout/groups_timed_out": float(self._timed_out_groups),
            "rollout/groups_cancelled": float(self._cancelled_groups),
            "rollout/groups_pending": float(len(self._pending)),
            "rollout/groups_ready": float(len(self._ready)),
            "rollout/trajectories_pending": float(sum(len(group.rows) for group in self._pending.values())),
            "rollout/duplicate_results_rejected": float(self._duplicate_results_rejected),
            "rollout/policy_mismatch_results_rejected": float(self._policy_mismatch_results_rejected),
            "rollout/group_completion_latency_mean": self._completion_latency_total / completed if completed else 0.0,
            "rollout/group_straggler_wait_mean": self._straggler_wait_total / completed if completed else 0.0,
        }


__all__ = ["RolloutGroupAssembler"]
