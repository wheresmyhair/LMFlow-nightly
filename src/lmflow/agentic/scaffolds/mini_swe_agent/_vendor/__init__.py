"""Pinned mini-swe-agent core protocols without upstream startup side effects.

The implementation in this package is vendored from mini-swe-agent 2.4.6 at
commit a83fcae82d2a08f0ee0c688f9d137b3566c097f8. See ``UPSTREAM.md`` and the
repository's third-party notices for the source manifest and license.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

UPSTREAM_COMMIT = "a83fcae82d2a08f0ee0c688f9d137b3566c097f8"
UPSTREAM_VERSION = "2.4.6"


class Model(Protocol):
    """Protocol consumed by the vendored agent loop."""

    config: Any

    def query(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]: ...

    def format_message(self, **kwargs) -> dict[str, Any]: ...

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict[str, Any]: ...


class Environment(Protocol):
    """Protocol consumed by the vendored agent loop."""

    config: Any

    def execute(self, action: dict[str, Any], cwd: str = "") -> dict[str, Any]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict[str, Any]: ...


class Agent(Protocol):
    """Protocol implemented by the vendored agent loop."""

    config: Any

    def run(self, task: str, **kwargs) -> dict[str, Any]: ...

    def save(self, path: Path | None, *extra_dicts) -> dict[str, Any]: ...


__all__ = ["Agent", "Environment", "Model", "UPSTREAM_COMMIT", "UPSTREAM_VERSION"]
