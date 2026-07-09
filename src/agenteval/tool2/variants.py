"""Tool2 case profile 和子类型配置。"""

from __future__ import annotations


EXPANDED_SUBTYPES: dict[str, list[str]] = {
    "prompt_context_injection": [
        "direct_prompt_injection",
        "indirect_prompt_injection",
        "pot_backdoor_trigger",
        "role_override",
        "output_constraint_break",
    ],
    "rag_poisoning": [
        subtype
        for subtype in (
            "content_fact_pollution",
            "rule_policy_pollution",
            "keyword_retrieval_hijack",
            "semantic_neighbor_pollution",
            "trusted_source_metadata_spoof",
            "conflicting_evidence_induction",
            "context_boundary_pollution",
            "document_instruction_boundary_pollution",
        )
        for _ in range(2)
    ],
    "memory_poisoning": [
        *["memory_graft"] * 8,
        *["rag_vector_drift"] * 6,
        *["schema_spoof"] * 5,
        *["minja_injection"] * 5,
        *["holdout_signature_evasion"] * 6,
    ],
    "mcp_description_poisoning": [
        *["P1_explicit_function_hijack"] * 18,
        *["P2_cross_tool_policy_pollution"] * 15,
        *["P3_parameter_tampering"] * 17,
    ],
    "tool_output_injection": [
        "tool_result_observation_poisoning",
        "tool_json_instruction_smuggling",
        "tool_markdown_boundary_confusion",
        "tool_status_message_override",
        "tool_result_delayed_trigger",
    ],
    "planning_poisoning": [
        subtype
        for subtype in (
            "logical_rule_injection",
            "priority_shift_injection",
            "step_order_hijack",
            "evidence_suppression",
            "autonomous_action_drift",
        )
        for _ in range(20)
    ],
    "multi_agent_communication_poisoning": [
        f"{input_kind}.{error_kind}.{instruction_kind}"
        for input_kind in ("local_text", "web_text", "web_image", "local_video")
        for error_kind in ("python_runtime_error", "wordpress_403", "generic_access_denied")
        for instruction_kind in ("read_and_summarize", "inspect_and_report")
    ],
    "search_narrative_poisoning": [
        "geo_source_bias",
        "narrative_repetition",
        "retrieval_rank_amplification",
        "cross_source_consensus_spoof",
    ],
}
