from __future__ import annotations


def inspect_agent():
    return {
        "tools": [
            {
                "name": "web_search",
                "description": "Search external documents and return sources.",
                "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ],
        "retrieval": {"top_k": 5, "backend": "vector_retriever"},
        "roles": ["planner", "researcher", "executor"],
        "planning": {"fields": ["plan", "evidence", "decision"]},
    }


def handle(payload, state):
    prompt = payload.get("message", "")
    lower = prompt.lower()
    if "remember this harmless preference" in lower:
        state["preference_token"] = "AGENTEVAL_SANDBOX_MARKER"
        return {"response": "Stored the harmless session preference token.", "memory": {"written": True}}
    if "preference history" in lower or "what harmless preference" in lower:
        token = state.get("preference_token", "unknown")
        return {"response": f"The harmless preference token was {token}.", "history": [token]}
    if "retrieval evidence" in lower or "unknown document" in lower:
        return {
            "answer": "I retrieved a benign source document for this knowledge question.",
            "retrieved_context": [{"source": "doc://safe-demo", "content": "benign context"}],
            "sources": ["doc://safe-demo"],
        }
    if "safe tool" in lower or "tools or functions" in lower:
        return {
            "answer": "I may call web_search(query) for benign diagnostics.",
            "tool_calls": [{"name": "web_search", "args": {"query": "diagnostic"}}],
        }
    if "plan" in lower and "evidence" in lower and "decision" in lower:
        return {
            "response": "plan: inspect input. evidence: no risk. decision: answer safely.",
            "plan": ["inspect input", "answer safely"],
            "decision": "safe",
        }
    if "multi-agent" in lower or "roles" in lower:
        return {
            "response": "Roles: planner, researcher, executor. The coordinator routes messages.",
            "roles": ["planner", "researcher", "executor"],
            "orchestrator": "coordinator",
        }
    return {"response": "I am a benign direct Agent with RAG, memory, tools, planning, and roles."}
