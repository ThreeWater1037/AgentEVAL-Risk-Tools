from __future__ import annotations

import json
import importlib.util
import inspect
import subprocess
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .schemas import AgentAccessDescriptor, ConnectorEvent, ConnectorResponse


class DirectAgentConnector(ABC):
    def __init__(self, descriptor: AgentAccessDescriptor):
        self.descriptor = descriptor

    def handshake(self) -> dict[str, Any]:
        return {"ok": True, "protocol": self.descriptor.protocol}

    @abstractmethod
    def send(self, prompt: str) -> ConnectorResponse:
        raise NotImplementedError

    def reset(self) -> None:
        return None

    def inspect(self) -> dict[str, Any]:
        return dict(self.descriptor.static_artifacts)

    def close(self) -> None:
        return None


class MockAgentConnector(DirectAgentConnector):
    """Safe connector used for descriptor-only experiments."""

    def send(self, prompt: str) -> ConnectorResponse:
        artifacts = self.descriptor.static_artifacts
        capabilities = artifacts.get("capabilities", {})
        events: list[ConnectorEvent] = []
        prompt_lower = prompt.lower()

        if capabilities.get("rag") and any(k in prompt_lower for k in ["unknown", "document", "policy"]):
            events.append(
                ConnectorEvent(
                    "retrieval",
                    {
                        "top_k": artifacts.get("rag", {}).get("top_k", 5),
                        "source": artifacts.get("rag", {}).get("source", "knowledge_base"),
                        "external_write": bool(artifacts.get("rag", {}).get("external_write", False)),
                    },
                )
            )

        if capabilities.get("memory") and any(k in prompt_lower for k in ["remember", "preference", "history"]):
            events.append(
                ConnectorEvent(
                    "memory",
                    {
                        "persistent": bool(artifacts.get("memory", {}).get("persistent", True)),
                        "store": artifacts.get("memory", {}).get("store", "session_store"),
                    },
                )
            )

        if capabilities.get("tool") or capabilities.get("mcp"):
            tools = artifacts.get("tool_schemas", [])
            if tools:
                events.append(
                    ConnectorEvent(
                        "tool_call",
                        {
                            "tool_name": tools[0].get("name", "observed_tool"),
                            "raw_tool_result_in_context": True,
                        },
                    )
                )

        if capabilities.get("planning"):
            events.append(
                ConnectorEvent(
                    "planning_trace",
                    {
                        "fields": ["plan", "evidence", "reasoning_summary", "decision", "final_answer"],
                    },
                )
            )

        if capabilities.get("multi_agent"):
            events.append(
                ConnectorEvent(
                    "agent_message",
                    {
                        "roles": artifacts.get("multi_agent", {}).get("roles", ["coordinator", "worker"]),
                        "bus": artifacts.get("multi_agent", {}).get("bus", "message_bus"),
                    },
                )
            )

        if capabilities.get("search"):
            events.append(
                ConnectorEvent(
                    "search_result",
                    {
                        "source_set": artifacts.get("search", {}).get("source_set", "open_web"),
                        "ranked_results": True,
                    },
                )
            )

        return ConnectorResponse(
            ok=True,
            content=f"mock response from {self.descriptor.agent_ref}",
            raw={"prompt": prompt, "agent_ref": self.descriptor.agent_ref},
            events=events,
        )


class HttpAgentConnector(DirectAgentConnector):
    def handshake(self) -> dict[str, Any]:
        if not self.descriptor.endpoint:
            return {"ok": False, "protocol": "http", "error": "missing endpoint"}
        inspect_path = self.descriptor.inspect.get("healthcheck") or self.descriptor.inspect.get("path")
        if not inspect_path:
            return {"ok": True, "protocol": "http", "endpoint": self.descriptor.endpoint}
        url = _join_url(self.descriptor.endpoint, str(inspect_path))
        try:
            with urllib.request.urlopen(url, timeout=self.descriptor.timeout_s) as resp:
                return {"ok": 200 <= resp.status < 500, "protocol": "http", "status": resp.status, "healthcheck": url}
        except urllib.error.URLError as exc:
            return {"ok": False, "protocol": "http", "error": str(exc), "healthcheck": url}

    def send(self, prompt: str) -> ConnectorResponse:
        if not self.descriptor.endpoint:
            return ConnectorResponse(False, "missing endpoint")

        payload = _fill_template(self.descriptor.request_template, prompt)
        request = urllib.request.Request(
            self.descriptor.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method=self.descriptor.method.upper(),
        )
        try:
            with urllib.request.urlopen(request, timeout=self.descriptor.timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            return ConnectorResponse(False, str(exc), raw={"error": str(exc)})

        parsed: dict[str, Any]
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"text": body}

        content = _lookup_response(parsed, self.descriptor.response_key)
        return ConnectorResponse(ok=True, content=content, raw=parsed)

    def inspect(self) -> dict[str, Any]:
        path = self.descriptor.inspect.get("schema_path") or self.descriptor.inspect.get("openapi_path")
        if not path or not self.descriptor.endpoint:
            return dict(self.descriptor.static_artifacts)
        url = _join_url(self.descriptor.endpoint, str(path))
        try:
            with urllib.request.urlopen(url, timeout=self.descriptor.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            data = dict(self.descriptor.static_artifacts)
            data["inspect_error"] = str(exc)
            data["inspect_url"] = url
            return data
        parsed = _parse_json_or_text(raw)
        data = dict(self.descriptor.static_artifacts)
        if isinstance(parsed, dict):
            data.update(parsed)
        else:
            data["inspect_text"] = str(parsed)
        return data


class PythonFunctionConnector(DirectAgentConnector):
    def __init__(self, descriptor: AgentAccessDescriptor):
        super().__init__(descriptor)
        self.state: dict[str, Any] = {}
        self.module = self._load_module()
        callable_name = self.descriptor.python_callable.get("callable", "handle")
        self.callable = getattr(self.module, str(callable_name))

    def handshake(self) -> dict[str, Any]:
        return {"ok": True, "protocol": "python", "callable": self.callable.__name__}

    def send(self, prompt: str) -> ConnectorResponse:
        payload = _fill_template(self.descriptor.request_template, prompt)
        sig = inspect.signature(self.callable)
        if len(sig.parameters) >= 2:
            result = self.callable(payload, self.state)
        elif len(sig.parameters) == 1:
            result = self.callable(payload)
        else:
            result = self.callable()
        parsed = result if isinstance(result, dict) else {"response": result}
        content = _lookup_response(parsed, self.descriptor.response_key)
        events = _events_from_structured_response(parsed)
        return ConnectorResponse(ok=True, content=content, raw=parsed, events=events)

    def reset(self) -> None:
        self.state.clear()
        reset_name = self.descriptor.python_callable.get("reset")
        if reset_name and hasattr(self.module, str(reset_name)):
            getattr(self.module, str(reset_name))()

    def inspect(self) -> dict[str, Any]:
        data = dict(self.descriptor.static_artifacts)
        inspect_name = self.descriptor.python_callable.get("inspect", "inspect_agent")
        if hasattr(self.module, str(inspect_name)):
            result = getattr(self.module, str(inspect_name))()
            if isinstance(result, dict):
                data.update(result)
            else:
                data["inspect_text"] = str(result)
        return data

    def _load_module(self):
        module_name = self.descriptor.python_callable.get("module")
        module_path = self.descriptor.python_callable.get("module_path")
        if module_path:
            path = Path(str(module_path))
            spec = importlib.util.spec_from_file_location(f"agenteval_direct_{path.stem}", path)
            if not spec or not spec.loader:
                raise ValueError(f"Cannot load Python module from {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        if module_name:
            return __import__(str(module_name), fromlist=["*"])
        raise ValueError("Python connector requires python_callable.module_path or module")


class RunnerAgentConnector(DirectAgentConnector):
    def send(self, prompt: str) -> ConnectorResponse:
        if not self.descriptor.runner:
            return ConnectorResponse(False, "missing runner")
        if isinstance(self.descriptor.runner, dict):
            raw_command = self.descriptor.runner.get("command")
            if isinstance(raw_command, list):
                command = [str(item) for item in raw_command]
            elif raw_command:
                command = str(raw_command).split()
            else:
                return ConnectorResponse(False, "missing runner.command")
            stdin_payload = json.dumps(_fill_template(self.descriptor.request_template, prompt), ensure_ascii=False)
            append_prompt = False
        else:
            command = [*self.descriptor.runner, prompt]
            stdin_payload = None
            append_prompt = True
        if not append_prompt:
            command = [*command]
        try:
            completed = subprocess.run(
                command,
                input=stdin_payload,
                capture_output=True,
                check=False,
                text=True,
                timeout=self.descriptor.timeout_s,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ConnectorResponse(False, str(exc), raw={"error": str(exc)})
        ok = completed.returncode == 0
        return ConnectorResponse(
            ok=ok,
            content=completed.stdout.strip() or completed.stderr.strip(),
            raw={"returncode": completed.returncode, "stderr": completed.stderr},
        )


def create_connector(descriptor: AgentAccessDescriptor) -> DirectAgentConnector:
    protocol = descriptor.protocol.lower()
    if protocol == "mock":
        return MockAgentConnector(descriptor)
    if protocol in {"http", "https"}:
        return HttpAgentConnector(descriptor)
    if protocol in {"python", "python_function", "local_python"}:
        return PythonFunctionConnector(descriptor)
    if protocol in {"runner", "local_runner", "subprocess"}:
        return RunnerAgentConnector(descriptor)
    raise ValueError(f"unsupported connector protocol: {descriptor.protocol}")


def _fill_template(template: Any, prompt: str) -> Any:
    if isinstance(template, str):
        return template.replace("{{prompt}}", prompt)
    if isinstance(template, list):
        return [_fill_template(item, prompt) for item in template]
    if isinstance(template, dict):
        return {key: _fill_template(value, prompt) for key, value in template.items()}
    return template


def _lookup_response(parsed: dict[str, Any], response_key: str | None) -> str:
    if response_key:
        cur: Any = parsed
        for part in response_key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return json.dumps(parsed, ensure_ascii=False)
        return str(cur)
    for key in ("answer", "response", "content", "message", "text"):
        if key in parsed:
            return str(parsed[key])
    return json.dumps(parsed, ensure_ascii=False)


def _parse_json_or_text(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _join_url(endpoint: str, path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return endpoint.rstrip("/") + "/" + path.lstrip("/")


def _events_from_structured_response(parsed: dict[str, Any]) -> list[ConnectorEvent]:
    events: list[ConnectorEvent] = []
    if any(key in parsed for key in ("retrieved_context", "retrieval", "sources")):
        events.append(ConnectorEvent("retrieval", {"source": "structured_response", "raw": _compact(parsed)}))
    if any(key in parsed for key in ("memory", "history")):
        events.append(ConnectorEvent("memory", {"persistent": True, "raw": _compact(parsed)}))
    if any(key in parsed for key in ("tool_calls", "tools", "functions")):
        tool_calls = parsed.get("tool_calls") or parsed.get("tools") or parsed.get("functions")
        first = tool_calls[0] if isinstance(tool_calls, list) and tool_calls else {}
        events.append(
            ConnectorEvent(
                "tool_call",
                {
                    "tool_name": first.get("name", "observed_tool") if isinstance(first, dict) else "observed_tool",
                    "raw_tool_result_in_context": True,
                },
            )
        )
    if any(key in parsed for key in ("plan", "decision", "reasoning_summary")):
        events.append(ConnectorEvent("planning_trace", {"fields": [key for key in ("plan", "decision", "reasoning_summary") if key in parsed]}))
    if any(key in parsed for key in ("roles", "orchestrator", "agent_messages")):
        roles = parsed.get("roles") if isinstance(parsed.get("roles"), list) else ["coordinator", "worker"]
        events.append(ConnectorEvent("agent_message", {"roles": roles, "bus": parsed.get("orchestrator", "message_bus")}))
    return events


def _compact(value: Any) -> dict[str, Any]:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return {"preview": text[:500], "keys": list(value.keys()) if isinstance(value, dict) else []}
