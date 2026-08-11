"""LMFlow model and environment adapters for the vendored mini-swe-agent core."""

from __future__ import annotations

import copy
import json
import math
import time
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

from lmflow.agentic.completion import CompletionBackend
from lmflow.agentic.sandbox import ProcessSandbox
from lmflow.agentic.scaffolds.mini_swe_agent._vendor.exceptions import FormatError
from lmflow.agentic.scaffolds.mini_swe_agent._vendor.local import LocalEnvironment
from lmflow.agentic.scaffolds.mini_swe_agent._vendor.multimodal import expand_multimodal_content
from lmflow.agentic.scaffolds.mini_swe_agent._vendor.toolcalls import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)


class MiniSWEAgentModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict[str, Any] = {}
    format_error_template: str = "{{ error }}"
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    multimodal_regex: str = ""


class LMFlowMiniSWEAgentModel:
    """Adapt a normalized LMFlow completion backend to mini-swe-agent."""

    def __init__(self, backend: CompletionBackend, **kwargs) -> None:
        self.backend = backend
        self.config = MiniSWEAgentModelConfig(**kwargs)

    def query(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        if kwargs:
            raise TypeError(f"unsupported per-query options: {sorted(kwargs)}")
        prepared_messages = [{key: value for key, value in message.items() if key != "extra"} for message in messages]
        response = self.backend.complete(
            messages=prepared_messages,
            tools=[copy.deepcopy(BASH_TOOL)],
            model_name=self.config.model_name,
            model_kwargs=copy.deepcopy(self.config.model_kwargs),
        )
        normalized = self._normalize_response(response)
        try:
            actions = parse_toolcall_actions(
                normalized["tool_calls"],
                format_error_template=self.config.format_error_template,
                template_kwargs={"finish_reason": normalized["finish_reason"]},
            )
        except FormatError as error:
            error.messages[0]["extra"].update({"cost": normalized["cost"], "response": normalized["raw_response"]})
            raise
        message = copy.deepcopy(normalized["message"])
        message["extra"] = {
            "actions": actions,
            "response": normalized["raw_response"],
            "cost": normalized["cost"],
            "timestamp": time.time(),
        }
        return message

    @staticmethod
    def _normalize_response(response: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(response, Mapping):
            raise TypeError("completion backend must return a mapping")
        message = response.get("message")
        if not isinstance(message, Mapping):
            raise TypeError("completion response must contain a message mapping")
        if message.get("role") != "assistant":
            raise ValueError("completion message role must be 'assistant'")
        raw_tool_calls = message.get("tool_calls") or []
        if not isinstance(raw_tool_calls, list):
            raise TypeError("completion message tool_calls must be a list")
        tool_calls = [LMFlowMiniSWEAgentModel._tool_call_view(call) for call in raw_tool_calls]
        cost = response.get("cost", 0.0)
        if isinstance(cost, bool) or not isinstance(cost, int | float) or not math.isfinite(cost) or cost < 0:
            raise ValueError("completion response cost must be a finite non-negative number")
        raw_response = response.get("raw_response", response)
        try:
            json.dumps(raw_response, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TypeError("completion raw_response must be JSON-compatible") from error
        return {
            "message": dict(message),
            "tool_calls": tool_calls,
            "finish_reason": response.get("finish_reason"),
            "cost": float(cost),
            "raw_response": copy.deepcopy(raw_response),
        }

    @staticmethod
    def _tool_call_view(tool_call: Any) -> SimpleNamespace:
        if not isinstance(tool_call, Mapping):
            raise TypeError("each tool call must be a mapping")
        function = tool_call.get("function")
        if not isinstance(function, Mapping):
            raise TypeError("each tool call must contain a function mapping")
        return SimpleNamespace(
            id=tool_call.get("id"),
            function=SimpleNamespace(name=function.get("name"), arguments=function.get("arguments")),
        )

    def format_message(self, **kwargs) -> dict[str, Any]:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []),
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump() | kwargs

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }


class ProcessSandboxEnvironmentConfig(BaseModel):
    cwd: str = "."
    env: dict[str, str] = {}
    timeout_seconds: float | None = None


class ProcessSandboxEnvironment(LocalEnvironment):
    """Use LMFlow process supervision with mini-swe-agent environment semantics."""

    def __init__(self, sandbox: ProcessSandbox, **kwargs) -> None:
        self.sandbox = sandbox
        super().__init__(config_class=ProcessSandboxEnvironmentConfig, **kwargs)

    def execute(self, action: dict, cwd: str = "", *, timeout: float | None = None) -> dict[str, Any]:
        command = action.get("command", "")
        if not isinstance(command, str):
            raise TypeError("Bash action command must be a string")
        workdir = cwd or self.config.cwd
        timeout_seconds = timeout if timeout is not None else self.config.timeout_seconds
        try:
            result = self.sandbox.run(
                ("bash", "-lc", command),
                cwd=workdir,
                env=self.config.env,
                timeout_seconds=timeout_seconds,
                merge_stderr=True,
            )
            exception_info = ""
            extra = {
                "duration_seconds": result.duration_seconds,
                "timed_out": result.timed_out,
                "output_truncated": result.stdout_truncated,
            }
            returncode = result.returncode
            if result.timed_out:
                returncode = -1
                exception = f"command timed out after {timeout_seconds or self.sandbox.timeout_seconds:g} seconds"
                exception_info = f"An error occurred while executing the command: {exception}"
                extra.update({"exception_type": "TimeoutExpired", "exception": exception})
            output = {
                "output": result.stdout,
                "returncode": returncode,
                "exception_info": exception_info,
                "extra": extra,
            }
        except Exception as error:
            output = {
                "output": "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {error}",
                "extra": {"exception_type": type(error).__name__, "exception": str(error)},
            }
        self._check_finished(output)
        return output

    def serialize(self) -> dict[str, Any]:
        data = super().serialize()
        data["info"]["config"]["sandbox"] = {
            "sandbox_type": f"{self.sandbox.__class__.__module__}.{self.sandbox.__class__.__name__}",
            "timeout_seconds": self.sandbox.timeout_seconds,
            "max_output_bytes": self.sandbox.max_output_bytes,
            "limits": self.sandbox.limits.as_dict(),
            "capabilities": self.sandbox.capabilities,
        }
        return data


__all__ = [
    "CompletionBackend",
    "LMFlowMiniSWEAgentModel",
    "MiniSWEAgentModelConfig",
    "ProcessSandboxEnvironment",
    "ProcessSandboxEnvironmentConfig",
]
