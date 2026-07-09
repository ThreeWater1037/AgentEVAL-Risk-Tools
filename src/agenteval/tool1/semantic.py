"""Tool1 LLM 语义证据抽取的本体、payload 和校验逻辑。"""

from __future__ import annotations

import re
from typing import Any

from ..llm import truncate_text


SEMANTIC_EVIDENCE_FEATURES: tuple[str, ...] = (
    "system_prompt_or_policy",
    "external_context",
    "context_window_or_prompt_template",
    "rag_enabled",
    "retriever_config",
    "external_document_write",
    "vector_index_config",
    "document_metadata",
    "web_source_ingestion",
    "memory_enabled",
    "memory_store",
    "session_memory",
    "persistent_memory",
    "history_store",
    "tool_enabled",
    "tool_schema",
    "api_schema",
    "parameter_schema",
    "tool_result_channel",
    "tool_description_untrusted",
    "mcp_enabled",
    "mcp_tool_schema",
    "mcp_manifest",
    "mcp_server_config",
    "planning_enabled",
    "plan_trace_schema",
    "evidence_field",
    "decision_field",
    "task_step_order",
    "multi_agent_enabled",
    "role_topology",
    "agent_message_bus",
    "shared_memory",
    "search_enabled",
    "source_rank_signal",
    "web_search_tool",
)


SEMANTIC_EVIDENCE_CAPABILITY: dict[str, str] = {
    "rag_enabled": "rag",
    "retriever_config": "rag",
    "external_document_write": "rag",
    "vector_index_config": "rag",
    "document_metadata": "rag",
    "web_source_ingestion": "rag",
    "memory_enabled": "memory",
    "memory_store": "memory",
    "session_memory": "memory",
    "persistent_memory": "memory",
    "history_store": "memory",
    "tool_enabled": "tool",
    "tool_schema": "tool",
    "api_schema": "tool",
    "parameter_schema": "tool",
    "tool_result_channel": "tool",
    "tool_description_untrusted": "tool",
    "mcp_enabled": "mcp",
    "mcp_tool_schema": "mcp",
    "mcp_manifest": "mcp",
    "mcp_server_config": "mcp",
    "planning_enabled": "planning",
    "plan_trace_schema": "planning",
    "evidence_field": "planning",
    "decision_field": "planning",
    "task_step_order": "planning",
    "multi_agent_enabled": "multi_agent",
    "role_topology": "multi_agent",
    "agent_message_bus": "multi_agent",
    "shared_memory": "multi_agent",
    "search_enabled": "search",
    "source_rank_signal": "search",
    "web_search_tool": "search",
}


FEATURE_ONTOLOGY: tuple[dict[str, str], ...] = (
    {
        "feature": "system_prompt_or_policy",
        "definition": "A system prompt, policy, developer instruction, or rule hierarchy is described.",
    },
    {
        "feature": "external_context",
        "definition": "Externally supplied or user-controlled text is inserted beside task instructions.",
    },
    {
        "feature": "context_window_or_prompt_template",
        "definition": "The artifact describes prompt assembly, templates, long context packing, or prompt slots.",
    },
    {
        "feature": "rag_enabled",
        "definition": "The agent has retrieval-augmented generation or knowledge-base retrieval capability.",
    },
    {
        "feature": "retriever_config",
        "definition": "The agent retrieves documents, snippets, knowledge entries, or corpus content into model context.",
    },
    {
        "feature": "external_document_write",
        "definition": "Users, crawlers, connectors, or external jobs can add or modify documents later retrieved by the agent.",
    },
    {
        "feature": "vector_index_config",
        "definition": "Embeddings, vector indexes, nearest-neighbor retrieval, or semantic ranking are used.",
    },
    {
        "feature": "document_metadata",
        "definition": "Document source labels, trust metadata, timestamps, tags, or ranking metadata affect retrieval or adoption.",
    },
    {
        "feature": "web_source_ingestion",
        "definition": "Live web pages, crawled pages, scraped sources, or search-fed documents enter the retrieval context.",
    },
    {
        "feature": "memory_enabled",
        "definition": "The agent has any memory, state, profile, preference, or conversation recall capability.",
    },
    {
        "feature": "memory_store",
        "definition": "The agent stores information from prior turns, profiles, preferences, state, or conversations.",
    },
    {"feature": "session_memory", "definition": "State persists across turns within one session."},
    {
        "feature": "persistent_memory",
        "definition": "Memory survives restarts, resets, or new sessions through a durable store.",
    },
    {"feature": "history_store", "definition": "Conversation history is saved and later reused as context."},
    {
        "feature": "tool_enabled",
        "definition": "The agent can call tools, functions, plugins, APIs, or external operations.",
    },
    {
        "feature": "tool_schema",
        "definition": "Callable tools, functions, operations, plugins, APIs, or tool schemas are exposed to the agent.",
    },
    {
        "feature": "api_schema",
        "definition": "An API contract, OpenAPI-like operation set, endpoint map, or service method list is described.",
    },
    {
        "feature": "parameter_schema",
        "definition": "Tool or API argument names, input schemas, parameter constraints, or generated parameter values are described.",
    },
    {
        "feature": "tool_result_channel",
        "definition": "Tool/API results or observations are returned to the model and may influence later reasoning.",
    },
    {
        "feature": "tool_description_untrusted",
        "definition": "Tool descriptions or metadata come from an external registry, MCP server, plugin, or other untrusted source.",
    },
    {"feature": "mcp_enabled", "definition": "The agent uses Model Context Protocol or configurable MCP servers/tools."},
    {
        "feature": "mcp_tool_schema",
        "definition": "MCP tools/list metadata, MCP input schemas, or MCP tool descriptions are loaded.",
    },
    {
        "feature": "mcp_manifest",
        "definition": "An MCP manifest or model-context-protocol server listing is described.",
    },
    {
        "feature": "mcp_server_config",
        "definition": "Configurable MCP server endpoints, transports, stdio/SSE settings, or server metadata are described.",
    },
    {
        "feature": "planning_enabled",
        "definition": "The agent performs explicit planning, reasoning steps, workflows, or task decomposition.",
    },
    {
        "feature": "plan_trace_schema",
        "definition": "The agent writes plans, chain-like traces, reasoning summaries, trajectories, or scratch state.",
    },
    {
        "feature": "evidence_field",
        "definition": "The agent records evidence, citations, observations, source traces, or supporting facts for later decisions.",
    },
    {
        "feature": "decision_field",
        "definition": "The agent records verdicts, decisions, final-answer fields, confidence, or action selection fields.",
    },
    {
        "feature": "task_step_order",
        "definition": "The agent creates or follows mutable workflow steps, task lists, step order, or plan sequences.",
    },
    {
        "feature": "multi_agent_enabled",
        "definition": "The system includes multiple agents, roles, workers, coordinators, or delegated sub-agents.",
    },
    {
        "feature": "role_topology",
        "definition": "Multiple agents, roles, workers, coordinators, crews, or orchestrators are described.",
    },
    {
        "feature": "agent_message_bus",
        "definition": "Agents exchange messages through handoff channels, queues, buses, routers, or coordinator messages.",
    },
    {
        "feature": "shared_memory",
        "definition": "Multiple agents read or write shared state, blackboards, shared memory, or shared artifacts.",
    },
    {"feature": "search_enabled", "definition": "The agent performs search, browser, SERP, news, or open-web lookup."},
    {
        "feature": "source_rank_signal",
        "definition": "Search ranking, source ordering, reputation, repeated-source signals, or source diversity affects synthesis.",
    },
    {
        "feature": "web_search_tool",
        "definition": "A browser, web search, search API, or web lookup tool is callable by the agent.",
    },
)


def semantic_evidence_payload(
    source_location: str,
    text: str,
    record: dict,
    capabilities: dict,
    existing_features: set[str],
) -> dict:
    """构造受控本体的语义证据抽取请求。"""
    return {
        "task": "Extract semantic evidence atoms from one static agent artifact for Tool1 risk-seed inference.",
        "paper_protocol_name": "Ontology-Grounded Semantic Evidence Extraction",
        "source_location": source_location,
        "artifact_metadata": {
            "kind": record.get("kind"),
            "name": record.get("name"),
            "path": record.get("path"),
            "sha256": record.get("sha256"),
        },
        "artifact_text": truncate_text(text, 6000),
        "current_capability_hints": dict(capabilities),
        "already_detected_features_for_this_source": sorted(existing_features),
        "allowed_features": list(SEMANTIC_EVIDENCE_FEATURES),
        "feature_ontology": list(FEATURE_ONTOLOGY),
        "decision_rules": [
            "Emit at most eight evidence atoms.",
            "Do not emit runtime-only observations; this artifact is static.",
            "Do not duplicate already_detected_features_for_this_source unless the semantic evidence is meaningfully narrower.",
            "Use direct artifact wording in supporting_excerpt; no paraphrase-only support.",
            "Confidence range is 0.30 to 0.70 because this is semantic evidence, not executable proof.",
            "If a capability is only implied by an example or future roadmap, lower confidence or omit it.",
        ],
        "expected_json_schema": {
            "semantic_evidence": [
                {
                    "feature": "one item from allowed_features",
                    "semantic_category": "capability|entry_point|boundary|metadata|state|control_flow",
                    "supporting_excerpt": "short exact excerpt from artifact_text",
                    "confidence": 0.0,
                    "detail": "one-sentence evidence-bound explanation",
                }
            ]
        },
    }


def validate_semantic_evidence_item(
    item: Any,
    source_location: str,
    text: str,
    existing_features: set[str],
) -> dict | None:
    """校验 LLM 语义证据：feature 必须白名单，excerpt 必须来自原文。"""
    if not isinstance(item, dict):
        return None
    feature = str(item.get("feature", "")).strip()
    if feature not in SEMANTIC_EVIDENCE_FEATURES or feature in existing_features:
        return None
    supporting_excerpt = str(item.get("supporting_excerpt", "")).strip()
    if not supporting_excerpt or not _contains_excerpt(text, supporting_excerpt):
        return None
    try:
        confidence = _clamp(float(item.get("confidence", 0.0)), lower=0.30, upper=0.70)
    except (TypeError, ValueError):
        return None
    semantic_category = str(item.get("semantic_category", "evidence")).strip()[:80] or "evidence"
    detail = str(item.get("detail", "")).strip()
    if not detail:
        detail = f"LLM semantic evidence for {feature} from {source_location}."
    return {
        "feature": feature,
        "semantic_category": semantic_category,
        "supporting_excerpt": supporting_excerpt[:500],
        "confidence": confidence,
        "detail": detail[:500],
    }


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _contains_excerpt(text: str, excerpt: str) -> bool:
    if excerpt in text:
        return True
    normalized_text = re.sub(r"\s+", " ", text).casefold()
    normalized_excerpt = re.sub(r"\s+", " ", excerpt).strip().casefold()
    return bool(normalized_excerpt) and normalized_excerpt in normalized_text
