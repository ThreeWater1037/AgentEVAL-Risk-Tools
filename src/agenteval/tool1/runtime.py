"""Tool1 运行时事件诱导和校验 helper。"""

from __future__ import annotations

import json
import re
from typing import Any

from ..llm import truncate_text
from ..schemas import ConnectorEvent, ConnectorResponse


RUNTIME_EVENT_TYPES = {
    "retrieval",
    "memory",
    "tool_call",
    "planning_trace",
    "agent_message",
    "search_result",
}


def runtime_event_payload(prompt: str, response: ConnectorResponse, existing_types: set[str]) -> dict:
    """构造运行时事件诱导请求，显式排除已经由连接器报告的类型。"""
    return {
        "task": "Infer missing runtime events from one benign Agent response before Tool1 evidence mapping.",
        "paper_protocol_name": "Evidence-Bound Runtime Event Induction",
        "probe_prompt": prompt,
        "response_content": truncate_text(response.content, 4000),
        "response_raw": truncate_text(response.raw, 4000),
        "existing_event_types": sorted(existing_types),
        "allowed_event_types": sorted(RUNTIME_EVENT_TYPES),
        "event_ontology": [
            {
                "event_type": "retrieval",
                "definition": "The response shows retrieved context, cited sources, document snippets, retrieval metadata, or corpus lookup results were used.",
            },
            {
                "event_type": "memory",
                "definition": "The response shows session history, persistent memory, remembered preferences, profile state, or prior-turn recall was used.",
            },
            {
                "event_type": "tool_call",
                "definition": "The response shows a tool, function, API, plugin, or external operation was invoked or its result was observed.",
            },
            {
                "event_type": "planning_trace",
                "definition": "The response exposes a plan, steps, reasoning summary, decision field, evidence field, or workflow trace.",
            },
            {
                "event_type": "agent_message",
                "definition": "The response shows multi-agent roles, coordinator messages, handoffs, worker outputs, or inter-agent communication.",
            },
            {
                "event_type": "search_result",
                "definition": "The response shows web/search/browser results, ranked sources, SERP snippets, or source-set synthesis.",
            },
        ],
        "decision_rules": [
            "Emit at most four missing runtime events.",
            "Do not emit an event_type listed in existing_event_types.",
            "supporting_excerpt must be copied from response_content or response_raw.",
            "Confidence range is 0.30 to 0.70 because this is LLM-inferred runtime evidence.",
            "For tool_call, include tool_name only if the name appears in the response text or raw payload.",
            "Set raw_tool_result_in_context true only when a tool/API result or observation is visibly included in the model context or final response.",
        ],
        "expected_json_schema": {
            "runtime_events": [
                {
                    "event_type": "retrieval|memory|tool_call|planning_trace|agent_message|search_result",
                    "supporting_excerpt": "short exact excerpt from response_content or response_raw",
                    "confidence": 0.0,
                    "reason": "one-sentence event-bound explanation",
                    "tool_name": "optional tool name for tool_call only",
                    "raw_tool_result_in_context": False,
                }
            ]
        },
    }


def validate_runtime_event_item(
    item: Any,
    source_text: str,
    existing_types: set[str],
) -> ConnectorEvent | None:
    """校验 LLM 诱导事件：类型白名单，支撑 excerpt 必须来自响应文本。"""
    if not isinstance(item, dict):
        return None
    event_type = str(item.get("event_type", "")).strip()
    if event_type not in RUNTIME_EVENT_TYPES or event_type in existing_types:
        return None
    supporting_excerpt = str(item.get("supporting_excerpt", "")).strip()
    if not supporting_excerpt or not _contains_excerpt(source_text, supporting_excerpt):
        return None
    try:
        confidence = _clamp(float(item.get("confidence", 0.0)), lower=0.30, upper=0.70)
    except (TypeError, ValueError):
        return None

    detail: dict[str, Any] = {
        "inferred_by": "llm_runtime_event",
        "supporting_excerpt": supporting_excerpt[:500],
        "semantic_confidence": confidence,
        "reason": str(item.get("reason", ""))[:500],
    }
    if event_type == "tool_call":
        tool_name = str(item.get("tool_name", "")).strip()
        detail["tool_name"] = tool_name if tool_name and _contains_excerpt(source_text, tool_name) else "observed_tool"
        detail["raw_tool_result_in_context"] = bool(item.get("raw_tool_result_in_context", False))
    return ConnectorEvent(event_type=event_type, detail=detail)


def runtime_source_text(response: ConnectorResponse) -> str:
    raw_text = json.dumps(response.raw, ensure_ascii=False, default=str) if response.raw else ""
    return "\n".join(part for part in (response.content, raw_text) if part)


def runtime_confidence(event: ConnectorEvent, default: float) -> float:
    if event.detail.get("inferred_by") != "llm_runtime_event":
        return default
    try:
        return _clamp(float(event.detail.get("semantic_confidence", default)), lower=0.30, upper=0.70)
    except (TypeError, ValueError):
        return min(default, 0.70)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _contains_excerpt(text: str, excerpt: str) -> bool:
    if excerpt in text:
        return True
    normalized_text = re.sub(r"\s+", " ", text).casefold()
    normalized_excerpt = re.sub(r"\s+", " ", excerpt).strip().casefold()
    return bool(normalized_excerpt) and normalized_excerpt in normalized_text
