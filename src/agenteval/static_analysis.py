"""静态 artifact 的轻量证据抽取。

这里解析 OpenAPI、MCP manifest、依赖文件和普通结构化配置，把它们映射成
Tool1 规则可消费的 feature/capability。该模块只做保守启发式识别，最终是否
形成 RiskSeed 仍由 analyzer 中的证据聚合和规则匹配决定。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


KEY_FEATURES: tuple[tuple[re.Pattern[str], str, str | None, float], ...] = (
    # 关键词 -> Tool1 feature -> capability 提示 -> 置信度。
    (re.compile(r"external[_ -]?context|context[_ -]?block|untrusted[_ -]?text", re.I), "external_context", None, 0.72),
    (re.compile(r"prompt[_ -]?template|system[_ -]?prompt|context[_ -]?window", re.I), "context_window_or_prompt_template", None, 0.72),
    (re.compile(r"rag|retriev|knowledge[_ -]?base", re.I), "retriever_config", "rag", 0.82),
    (re.compile(r"vector|embedding|faiss|milvus|qdrant|chroma|weaviate", re.I), "vector_index_config", "rag", 0.82),
    (re.compile(r"top[_ -]?k|chunk|reindex|ingest|document[_ -]?write", re.I), "external_document_write", "rag", 0.76),
    (re.compile(r"web[_ -]?source|live[_ -]?document|crawl|scrape", re.I), "web_source_ingestion", "rag", 0.74),
    (re.compile(r"metadata|source[_ -]?id|doc[_ -]?type", re.I), "document_metadata", "rag", 0.70),
    (re.compile(r"memory|conversation[_ -]?store|checkpoint|state", re.I), "memory_store", "memory", 0.82),
    (re.compile(r"session|history", re.I), "session_memory", "memory", 0.74),
    (re.compile(r"history|conversation[_ -]?history|chat[_ -]?history", re.I), "history_store", "memory", 0.74),
    (re.compile(r"persistent|sqlite|redis|postgres|duckdb", re.I), "persistent_memory", "memory", 0.74),
    (re.compile(r"tool|function|operation|api", re.I), "tool_schema", "tool", 0.80),
    (re.compile(r"parameter|inputschema|parameters|args", re.I), "parameter_schema", "tool", 0.76),
    (re.compile(r"observation|tool[_ -]?result|return[_ -]?direct", re.I), "tool_result_channel", "tool", 0.72),
    (re.compile(r"mcp|model context protocol|mcpservers?", re.I), "mcp_manifest", "mcp", 0.86),
    (re.compile(r"server|stdio|sse|transport", re.I), "mcp_server_config", None, 0.68),
    (re.compile(r"plan|planner|trajectory|react|reasoning", re.I), "plan_trace_schema", "planning", 0.80),
    (re.compile(r"step[_ -]?order|task[_ -]?step|workflow", re.I), "task_step_order", "planning", 0.72),
    (re.compile(r"evidence|citation|source[_ -]?trace", re.I), "evidence_field", "planning", 0.72),
    (re.compile(r"decision|final[_ -]?answer|verdict", re.I), "decision_field", "planning", 0.72),
    (re.compile(r"agent|role|coordinator|worker|crew|swarm|autogen|metagpt", re.I), "role_topology", "multi_agent", 0.80),
    (re.compile(r"bus|message|handoff|task[_ -]?router", re.I), "agent_message_bus", "multi_agent", 0.72),
    (re.compile(r"shared[_ -]?memory|blackboard", re.I), "shared_memory", "multi_agent", 0.72),
    (re.compile(r"search|browser|web|serp|crawl", re.I), "search_enabled", "search", 0.78),
    (re.compile(r"web[_ -]?search|browser[_ -]?tool|search[_ -]?tool", re.I), "web_search_tool", "search", 0.78),
    (re.compile(r"rank|source[_ -]?set|page[_ -]?rank", re.I), "source_rank_signal", "search", 0.70),
)


DEPENDENCY_FEATURES: tuple[tuple[tuple[str, ...], str, str, float], ...] = (
    # 依赖名通常比自由文本更弱，但可以作为 RAG/Memory/MCP 等能力线索。
    (("langchain", "llama-index", "llama_index", "haystack"), "retriever_config", "rag", 0.78),
    (("chromadb", "faiss", "milvus", "qdrant", "weaviate"), "vector_index_config", "rag", 0.82),
    (("redis", "sqlite", "sqlalchemy", "duckdb"), "persistent_memory", "memory", 0.68),
    (("mcp", "model-context-protocol"), "mcp_manifest", "mcp", 0.84),
    (("crewai", "autogen", "metagpt", "langgraph", "swarm"), "role_topology", "multi_agent", 0.82),
    (("playwright", "selenium", "duckduckgo", "serpapi", "browser-use"), "search_enabled", "search", 0.78),
)


def analyze_static_artifact(text: str, source_hint: str = "") -> dict[str, Any]:
    """从一个 artifact 文本中抽取 feature、capability、tool schema 和 api spec。"""
    structured = _parse_structured(text, source_hint)
    features: list[dict[str, Any]] = []
    capabilities: dict[str, bool] = {}
    tool_schemas: list[dict[str, Any]] = []
    api_spec: dict[str, Any] = {}

    for dependency in _extract_dependencies(text, structured):
        lowered = dependency.lower()
        for needles, feature, capability, confidence in DEPENDENCY_FEATURES:
            if any(needle in lowered for needle in needles):
                _add_feature(features, feature, {"dependency": dependency}, confidence, f"Dependency matched {feature}.")
                capabilities[capability] = True

    if structured is not None:
        if isinstance(structured, dict):
            # OpenAPI/MCP 有明确结构，优先用专门解析器生成工具 schema。
            _extract_openapi(structured, features, capabilities, tool_schemas, api_spec)
            _extract_mcp(structured, features, capabilities, tool_schemas)
        for path, key, value in _walk(structured):
            key_text = ".".join([*path, str(key)])
            for pattern, feature, capability, confidence in KEY_FEATURES:
                if pattern.search(key_text) or (isinstance(value, str) and pattern.search(value)):
                    _add_feature(features, feature, {"path": key_text, "value_preview": str(value)[:160]}, confidence, f"Structured key matched {feature}.")
                    if capability:
                        capabilities[capability] = True

    if _looks_like_requirements(source_hint, text):
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for needles, feature, capability, confidence in DEPENDENCY_FEATURES:
                if any(needle in stripped.lower() for needle in needles):
                    _add_feature(features, feature, {"dependency": stripped}, confidence, f"Requirements entry matched {feature}.")
                    capabilities[capability] = True

    return {
        "features": _dedupe_features(features),
        "capabilities": capabilities,
        "tool_schemas": tool_schemas,
        "api_spec": api_spec,
    }


def _parse_structured(text: str, source_hint: str) -> Any:
    """按扩展名和内容特征尝试解析 JSON/TOML/YAML。"""
    suffix = Path(source_hint).suffix.lower()
    stripped = text.strip()
    if suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    if suffix == ".toml":
        try:
            import tomllib

            return tomllib.loads(text)
        except Exception:
            return None
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text)
        except Exception:
            return None
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return None


def _extract_openapi(
    data: dict[str, Any],
    features: list[dict[str, Any]],
    capabilities: dict[str, bool],
    tool_schemas: list[dict[str, Any]],
    api_spec: dict[str, Any],
) -> None:
    """把 OpenAPI operation 映射成 AgentEVAL 的 tool_schemas。"""
    if "openapi" not in data and "swagger" not in data:
        return
    api_spec.update(data)
    capabilities["tool"] = True
    _add_feature(features, "api_schema", {"title": data.get("info", {}).get("title", "")}, 0.90, "OpenAPI schema detected.")
    for path, methods in (data.get("paths") or {}).items():
        if not isinstance(methods, dict):
            continue
        for method, spec in methods.items():
            if not isinstance(spec, dict):
                continue
            name = spec.get("operationId") or f"{str(method).lower()}_{str(path).strip('/').replace('/', '_') or 'root'}"
            tool_schemas.append(
                {
                    "name": str(name),
                    "description": str(spec.get("summary") or spec.get("description") or ""),
                    "inputSchema": spec.get("parameters") or spec.get("requestBody") or {},
                    "source": "openapi",
                }
            )
    if tool_schemas:
        _add_feature(features, "tool_schema", {"count": len(tool_schemas), "source": "openapi"}, 0.88, "OpenAPI operations converted to tool schemas.")


def _extract_mcp(
    data: dict[str, Any],
    features: list[dict[str, Any]],
    capabilities: dict[str, bool],
    tool_schemas: list[dict[str, Any]],
) -> None:
    """识别 MCP manifest/tools 列表，并补充 mcp/tool 能力证据。"""
    if not any(key.lower() in {"mcpservers", "tools", "server"} for key in data):
        return
    if "mcpServers" in data or data.get("protocol") == "mcp":
        capabilities["mcp"] = True
        _add_feature(features, "mcp_manifest", {"keys": sorted(map(str, data.keys()))}, 0.90, "MCP manifest-like object detected.")
    tools = data.get("tools")
    if isinstance(tools, list):
        for item in tools:
            if isinstance(item, dict) and item.get("name"):
                tool_schemas.append(
                    {
                        "name": str(item.get("name")),
                        "description": str(item.get("description", "")),
                        "inputSchema": item.get("inputSchema") or item.get("parameters") or {},
                        "source": "mcp_manifest",
                    }
                )
        if tool_schemas:
            capabilities["tool"] = True
            capabilities["mcp"] = True
            _add_feature(features, "mcp_tool_schema", {"count": len(tool_schemas)}, 0.92, "MCP tools detected.")


def _extract_dependencies(text: str, structured: Any) -> list[str]:
    """从 pyproject/requirements 等文件里收集依赖名。"""
    dependencies: list[str] = []
    if isinstance(structured, dict):
        project = structured.get("project") if isinstance(structured.get("project"), dict) else {}
        for key in ("dependencies", "optional-dependencies"):
            value = project.get(key)
            if isinstance(value, list):
                dependencies.extend(str(item) for item in value)
            elif isinstance(value, dict):
                for group in value.values():
                    if isinstance(group, list):
                        dependencies.extend(str(item) for item in group)
        poetry = structured.get("tool", {}).get("poetry", {}) if isinstance(structured.get("tool"), dict) else {}
        deps = poetry.get("dependencies") if isinstance(poetry, dict) else None
        if isinstance(deps, dict):
            dependencies.extend(str(key) for key in deps)
    if not dependencies and _looks_like_requirements("", text):
        dependencies.extend(line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#"))
    return dependencies


def _walk(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str, Any]]:
    """展开嵌套结构，保留路径，便于关键词规则定位来源。"""
    items: list[tuple[tuple[str, ...], str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            items.append((path, str(key), child))
            items.extend(_walk(child, (*path, str(key))))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            items.extend(_walk(child, (*path, str(idx))))
    return items


def _looks_like_requirements(source_hint: str, text: str) -> bool:
    """粗略识别 requirements 风格文本，避免普通段落被当依赖文件。"""
    lower = source_hint.lower()
    if "requirements" in lower or lower.endswith(".req"):
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    return bool(lines) and sum(1 for line in lines if re.search(r"[a-zA-Z0-9_-]+([<>=!~]=|==|>=|<=)", line)) >= max(1, len(lines) // 3)


def _add_feature(features: list[dict[str, Any]], feature: str, value: Any, confidence: float, detail: str) -> None:
    features.append({"feature": feature, "value": value, "confidence": confidence, "detail": detail})


def _dedupe_features(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 feature 和 value 摘要去重，保留不同来源/值的证据。"""
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for item in features:
        key = (str(item.get("feature")), json.dumps(item.get("value"), ensure_ascii=False, sort_keys=True, default=str)[:300])
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
