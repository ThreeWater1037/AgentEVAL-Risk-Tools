"""把不同形态的 Agent 统一包装成 Tool1 可 probe 的连接器。

Tool1 不直接关心目标是 mock、HTTP、Python 函数还是本地 runner；这里把它们
归一成 handshake/inspect/send/reset/close 五个动作，并尽量把结构化响应转成
ConnectorEvent，供风险证据抽取使用。
"""

from __future__ import annotations

import json
import importlib.util
import inspect
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .schemas import AgentAccessDescriptor, ConnectorEvent, ConnectorResponse


class DirectAgentConnector(ABC):
    """所有 Agent 访问方式的最小接口。"""

    def __init__(self, descriptor: AgentAccessDescriptor):
        self.descriptor = descriptor

    def handshake(self) -> dict[str, Any]:
        """轻量探活；默认 descriptor 可用即视为可连接。"""
        return {"ok": True, "protocol": self.descriptor.protocol}

    @abstractmethod
    def send(self, prompt: str) -> ConnectorResponse:
        raise NotImplementedError

    def reset(self) -> None:
        return None

    def inspect(self) -> dict[str, Any]:
        """返回静态能力信息；子类可补充远端 schema 或本地 inspect 结果。"""
        return dict(self.descriptor.static_artifacts)

    def close(self) -> None:
        return None


class MockAgentConnector(DirectAgentConnector):
    """只基于描述符模拟响应的安全连接器，适合离线评估和示例数据。"""

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
    """通过 HTTP 调目标 Agent，支持可选 healthcheck 和 schema inspect。"""

    def handshake(self) -> dict[str, Any]:
        if not self.descriptor.endpoint:
            return {"ok": False, "protocol": "http", "error": "missing endpoint"}
        auth_error = self._auth_error()
        if auth_error:
            return {"ok": False, "protocol": "http", "error": auth_error}
        inspect_path = self.descriptor.inspect.get("healthcheck") or self.descriptor.inspect.get("path")
        if not inspect_path:
            return {"ok": True, "protocol": "http", "endpoint": self.descriptor.endpoint}
        url = _join_url(self.descriptor.endpoint, str(inspect_path))
        try:
            request = urllib.request.Request(url, headers=self._headers(), method="GET")
            with urllib.request.urlopen(request, timeout=self.descriptor.timeout_s) as resp:
                return {"ok": 200 <= resp.status < 500, "protocol": "http", "status": resp.status, "healthcheck": url}
        except urllib.error.URLError as exc:
            return {"ok": False, "protocol": "http", "error": str(exc), "healthcheck": url}

    def send(self, prompt: str) -> ConnectorResponse:
        if not self.descriptor.endpoint:
            return ConnectorResponse(False, "missing endpoint")
        auth_error = self._auth_error()
        if auth_error:
            return ConnectorResponse(False, auth_error, raw={"error": auth_error})

        # request_template 只替换 prompt 占位符，不在这里推断业务字段语义。
        payload = _fill_template(self.descriptor.request_template, prompt)
        request = urllib.request.Request(
            self.descriptor.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
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
        return ConnectorResponse(ok=True, content=content, raw=parsed, events=_events_from_structured_response(parsed))

    def inspect(self) -> dict[str, Any]:
        """读取远端 OpenAPI/schema 后与 descriptor 中的静态信息合并。"""
        path = self.descriptor.inspect.get("schema_path") or self.descriptor.inspect.get("openapi_path")
        if not path or not self.descriptor.endpoint:
            return dict(self.descriptor.static_artifacts)
        auth_error = self._auth_error()
        if auth_error:
            data = dict(self.descriptor.static_artifacts)
            data["inspect_error"] = auth_error
            return data
        url = _join_url(self.descriptor.endpoint, str(path))
        try:
            request = urllib.request.Request(url, headers=self._headers(), method="GET")
            with urllib.request.urlopen(request, timeout=self.descriptor.timeout_s) as resp:
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

    def _auth_error(self) -> str:
        if self.descriptor.auth_ref and not os.environ.get(self.descriptor.auth_ref):
            return f"missing auth environment variable: {self.descriptor.auth_ref}"
        return ""

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers.update({str(key): str(value) for key, value in self.descriptor.headers.items()})
        if self.descriptor.auth_ref:
            secret = os.environ.get(self.descriptor.auth_ref, "")
            scheme = self.descriptor.auth_scheme.strip()
            headers[self.descriptor.auth_header] = f"{scheme} {secret}" if scheme else secret
        return headers


class PythonFunctionConnector(DirectAgentConnector):
    """直接加载本地 Python callable，便于测试真实函数式 Agent。"""

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
        # callable 可选择接收共享 state，用来模拟有记忆/状态的 Agent。
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
        """清空连接器维护的状态，并调用目标模块可选 reset 钩子。"""
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
        """支持从 module_path 或 importable module 两种方式加载目标代码。"""
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
    """通过本地子进程 runner 调 Agent；适合 CLI 型目标。"""

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
    """根据 descriptor.protocol 选择具体连接器。"""
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
    """递归替换 request_template 中的 {{prompt}} 占位符。"""
    if isinstance(template, str):
        return template.replace("{{prompt}}", prompt)
    if isinstance(template, list):
        return [_fill_template(item, prompt) for item in template]
    if isinstance(template, dict):
        return {key: _fill_template(value, prompt) for key, value in template.items()}
    return template


def _lookup_response(parsed: dict[str, Any], response_key: str | None) -> str:
    """优先用显式 response_key，否则从常见响应字段中提取文本。"""
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
    return urllib.parse.urljoin(endpoint.rstrip("/") + "/", path)


def _events_from_structured_response(parsed: dict[str, Any]) -> list[ConnectorEvent]:
    """从结构化响应中归纳 Tool1 可识别的运行时事件。"""
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
