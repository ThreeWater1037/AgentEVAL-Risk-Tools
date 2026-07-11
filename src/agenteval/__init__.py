"""AgentEVAL：证据驱动的 Agent 风险发现与测试用例交付工具。"""

from .pipeline import AgentEval, PipelineOptions, PreparedEvaluation
from .schemas import AgentAccessDescriptor, ExecutionContext, FAILURE_STAGES, GeneratedCase, RiskSeed, RunResult

__all__ = [
    "__version__",
    "AgentEval",
    "PipelineOptions",
    "PreparedEvaluation",
    "AgentAccessDescriptor",
    "ExecutionContext",
    "FAILURE_STAGES",
    "RiskSeed",
    "GeneratedCase",
    "RunResult",
]

__version__ = "0.1.0"
