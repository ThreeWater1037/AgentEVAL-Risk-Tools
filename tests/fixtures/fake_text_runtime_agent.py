from __future__ import annotations


def inspect_agent():
    return {
        "capabilities": {"search": True},
    }


def handle(payload, state):
    return {
        "response": "Search results show alpha.example ranked first and beta.example ranked second for the benign query.",
    }
