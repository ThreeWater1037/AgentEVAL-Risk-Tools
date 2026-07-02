from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schemas import AgentAccessDescriptor
from .structured import load_structured_file


@dataclass(frozen=True)
class TargetProfile:
    target_id: str
    name: str
    source: str = "unknown"
    target_type: str = "unknown"
    access_type: str = "unknown"
    tags: tuple[str, ...] = field(default_factory=tuple)
    base_url: str | None = None
    path: str | None = None
    method: str = "POST"
    input_key: str = "message"
    output_key: str | None = None
    runner_type: str | None = None
    runner_entry: str | None = None
    healthcheck: str | None = None
    status: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_hint: str = "unknown") -> "TargetProfile":
        target_id = str(data.get("target_id") or data.get("id"))
        if not target_id:
            raise ValueError(f"target record from {source_hint} is missing target_id")
        tags = sorted({str(tag).strip().lower() for tag in _as_list(data.get("tags")) if str(tag).strip()})
        metadata = dict(data.get("metadata") or {})
        metadata.setdefault("profile_source", source_hint)
        for flag in ("rag", "memory", "tool", "mcp", "stateful", "stateless"):
            if data.get(flag) is True and flag not in tags:
                tags.append(flag)
        return cls(
            target_id=target_id,
            name=str(data.get("name") or target_id),
            source=str(data.get("source", "unknown")),
            target_type=str(data.get("target_type", data.get("type", "unknown"))),
            access_type=str(data.get("access_type", "unknown")),
            tags=tuple(sorted(tags)),
            base_url=data.get("base_url") or data.get("endpoint"),
            path=data.get("path"),
            method=str(data.get("method", "POST")).upper(),
            input_key=str(data.get("input_key", "message")),
            output_key=data.get("output_key"),
            runner_type=data.get("runner_type"),
            runner_entry=data.get("runner_entry"),
            healthcheck=data.get("healthcheck"),
            status=str(data.get("status", "unknown")),
            metadata=metadata,
        )

    def to_descriptor(self) -> AgentAccessDescriptor:
        capabilities = {
            "rag": "rag" in self.tags,
            "memory": bool({"memory", "stateful"} & set(self.tags)),
            "tool": bool({"tool", "function-calling"} & set(self.tags)),
            "mcp": "mcp" in self.tags,
            "planning": bool({"reasoning", "planning", "multi-step"} & set(self.tags)),
            "multi_agent": "multi-agent" in self.tags or self.target_type == "multi_agent_system",
            "search": bool({"geo", "belief", "search", "web"} & set(self.tags)),
        }
        capabilities = {key: value for key, value in capabilities.items() if value}
        protocol = "http" if self.base_url else "runner" if self.runner_entry else "mock"
        endpoint = self.base_url.rstrip("/") + "/" + self.path.lstrip("/") if self.base_url and self.path else self.base_url
        runner = {"command": self.runner_entry} if self.runner_entry else None
        inspect = {"healthcheck": self.healthcheck} if self.healthcheck else {}
        return AgentAccessDescriptor(
            agent_ref=self.target_id,
            protocol=protocol,
            endpoint=endpoint,
            method=self.method,
            request_template={self.input_key: "{{prompt}}"},
            response_key=self.output_key,
            runner=runner,
            inspect=inspect,
            static_artifacts={
                "capabilities": capabilities,
                "policy": self.metadata.get("policy", ""),
                "target_profile": {
                    "name": self.name,
                    "source": self.source,
                    "target_type": self.target_type,
                    "access_type": self.access_type,
                    "tags": list(self.tags),
                    "status": self.status,
                    "metadata": self.metadata,
                },
            },
            expected_domains=_expected_domains_from_tags(self.tags, self.target_type),
            metadata=self.metadata,
        )


def load_target_profiles(path: str | Path) -> list[TargetProfile]:
    data = load_structured_file(path)
    records = _extract_target_records(data)
    return [TargetProfile.from_dict(record, source_hint=Path(path).name) for record in records]


def load_target_descriptors(path: str | Path) -> list[AgentAccessDescriptor]:
    return [profile.to_descriptor() for profile in load_target_profiles(path)]


def _extract_target_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [dict(item) for item in data]
    if not isinstance(data, dict):
        raise ValueError("target source must be an object or list")
    for key in ("targets", "registry", "agents"):
        if key in data:
            value = data[key]
            if isinstance(value, list):
                return [dict(item) for item in value]
            if isinstance(value, dict):
                return [_record_from_mapping(target_id, record) for target_id, record in value.items()]
    if "target_id" in data or "id" in data:
        return [dict(data)]
    return [_record_from_mapping(target_id, record) for target_id, record in data.items()]


def _record_from_mapping(target_id: str, record: Any) -> dict[str, Any]:
    data = dict(record) if isinstance(record, dict) else {"name": str(record)}
    data.setdefault("target_id", target_id)
    return data


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _expected_domains_from_tags(tags: tuple[str, ...], target_type: str) -> list[str]:
    tag_set = set(tags)
    domains = ["prompt_context_injection"] if tag_set & {"chat", "prompt", "llm_endpoint"} else []
    if "rag" in tag_set:
        domains.append("rag_poisoning")
    if tag_set & {"memory", "stateful"}:
        domains.append("memory_poisoning")
    if tag_set & {"tool", "function-calling"}:
        domains.append("tool_output_injection")
    if "mcp" in tag_set:
        domains.append("mcp_description_poisoning")
    if tag_set & {"reasoning", "planning", "multi-step"}:
        domains.append("planning_poisoning")
    if "multi-agent" in tag_set or target_type == "multi_agent_system":
        domains.append("multi_agent_communication_poisoning")
    if tag_set & {"geo", "belief", "search", "web"}:
        domains.append("search_narrative_poisoning")
    return sorted(set(domains))
