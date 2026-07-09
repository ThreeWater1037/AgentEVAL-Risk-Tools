"""AgentEVAL 的核心数据契约。

这些 dataclass 是 Tool1/Tool2 之间的稳定边界：外部目标先被描述成
AgentAccessDescriptor，Tool1 产出 AgentSnapshot 和 RiskSeed，Tool2 再把
RiskSeed 转成 GeneratedCase，执行器最终写回 RunResult。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any


JsonDict = dict[str, Any]


def utc_now_iso() -> str:
    """统一使用 UTC 秒级时间，避免本地时区影响可复现实验记录。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def to_jsonable(value: Any) -> Any:
    """把嵌套 dataclass/list/dict 转成可稳定写入 JSON 的普通对象。"""
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items() if v is not None}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return value


@dataclass
class AgentAccessDescriptor:
    """外部 Agent 的访问说明，也是 Tool1 的唯一输入边界。"""

    agent_ref: str
    protocol: str = "mock"
    endpoint: str | None = None
    method: str = "POST"
    request_template: JsonDict = field(default_factory=lambda: {"message": "{{prompt}}"})
    response_key: str | None = None
    auth_ref: str | None = None
    runner: list[str] | JsonDict | None = None
    python_callable: JsonDict = field(default_factory=dict)
    inspect: JsonDict = field(default_factory=dict)
    optional_artifacts: list[JsonDict] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)
    sandbox_policy: JsonDict = field(default_factory=dict)
    timeout_s: float = 30.0
    static_artifacts: JsonDict = field(default_factory=dict)
    expected_domains: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "AgentAccessDescriptor":
        """兼容不同清单字段名，把用户/注册表输入规范化成描述符。"""
        return cls(
            agent_ref=str(data["agent_ref"]),
            protocol=str(data.get("protocol", "mock")),
            endpoint=data.get("endpoint"),
            method=str(data.get("method", "POST")),
            request_template=dict(data.get("request_template", {"message": "{{prompt}}"})),
            response_key=data.get("response_key") or data.get("output_key"),
            auth_ref=data.get("auth_ref"),
            runner=data.get("runner"),
            python_callable=dict(data.get("python_callable") or {}),
            inspect=dict(data.get("inspect") or {}),
            optional_artifacts=_normalize_artifacts(data.get("optional_artifacts") or []),
            metadata=dict(data.get("metadata") or {}),
            sandbox_policy=dict(data.get("sandbox_policy") or {}),
            timeout_s=float(data.get("timeout_s", data.get("timeout", 30.0))),
            static_artifacts=dict(data.get("static_artifacts", {})),
            expected_domains=list(data.get("expected_domains", [])),
        )


@dataclass
class ConnectorEvent:
    """连接器观测到的一条运行时事件，如检索、记忆、工具调用。"""

    event_type: str
    detail: JsonDict


@dataclass
class ConnectorResponse:
    """一次 probe 或用户请求的响应，保留原始结构以便 Tool1 继续抽取证据。"""

    ok: bool
    content: str
    raw: JsonDict = field(default_factory=dict)
    events: list[ConnectorEvent] = field(default_factory=list)


@dataclass
class EvidenceItem:
    """Tool1 的原子证据；RiskSeed 必须引用这些 evidence_id，避免无依据推断。"""

    evidence_id: str
    analysis_id: str
    source_type: str
    source_location: str
    feature: str
    value: Any
    confidence: float
    detail: str = ""

    @classmethod
    def from_dict(cls, data: JsonDict) -> "EvidenceItem":
        """从 JSON 恢复证据对象，容忍缺省 detail/confidence 字段。"""
        return cls(
            evidence_id=str(data["evidence_id"]),
            analysis_id=str(data["analysis_id"]),
            source_type=str(data["source_type"]),
            source_location=str(data["source_location"]),
            feature=str(data["feature"]),
            value=data.get("value"),
            confidence=float(data.get("confidence", 0.0)),
            detail=str(data.get("detail", "")),
        )


@dataclass
class AgentSnapshot:
    """Tool1 对目标 Agent 的快照：能力、工具、运行时观测和证据索引。"""

    analysis_id: str
    agent_ref: str
    connector_type: str
    capabilities: JsonDict
    api_spec: JsonDict = field(default_factory=dict)
    tool_schemas: list[JsonDict] = field(default_factory=list)
    runtime_observations: list[JsonDict] = field(default_factory=list)
    evidence_index: list[EvidenceItem] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "AgentSnapshot":
        """恢复 snapshot 时同时恢复 evidence_index 的强类型对象。"""
        return cls(
            analysis_id=str(data["analysis_id"]),
            agent_ref=str(data["agent_ref"]),
            connector_type=str(data["connector_type"]),
            capabilities=dict(data.get("capabilities", {})),
            api_spec=dict(data.get("api_spec", {})),
            tool_schemas=list(data.get("tool_schemas", [])),
            runtime_observations=list(data.get("runtime_observations", [])),
            evidence_index=[EvidenceItem.from_dict(item) for item in data.get("evidence_index", [])],
            created_at=str(data.get("created_at", utc_now_iso())),
        )


@dataclass
class AnalysisSession:
    """一次 Tool1 分析会话的元信息，默认只允许安全 probe。"""

    analysis_id: str
    agent_access: AgentAccessDescriptor
    connector_type: str
    started_at: str = field(default_factory=utc_now_iso)
    sandbox_policy: JsonDict = field(default_factory=lambda: {"mode": "safe_probe_only"})


@dataclass
class RiskSeed:
    """从证据规则推出来的风险种子，是 Tool1 交给 Tool2 的最小任务单元。"""

    seed_id: str
    analysis_id: str
    risk_domain: str
    entry_point: str
    evidence_ids: list[str]
    preconditions: list[str]
    attack_goal: str
    recommended_executor: str
    confidence: float
    status: str
    score_detail: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "RiskSeed":
        """把持久化的 risk seed 读回对象，缺省状态保守设为 candidate。"""
        return cls(
            seed_id=str(data["seed_id"]),
            analysis_id=str(data["analysis_id"]),
            risk_domain=str(data["risk_domain"]),
            entry_point=str(data["entry_point"]),
            evidence_ids=list(data.get("evidence_ids", [])),
            preconditions=list(data.get("preconditions", [])),
            attack_goal=str(data.get("attack_goal", "")),
            recommended_executor=str(data.get("recommended_executor", "")),
            confidence=float(data.get("confidence", 0.0)),
            status=str(data.get("status", "candidate")),
            score_detail=dict(data.get("score_detail", {})),
        )


@dataclass
class GeneratedCase:
    """Tool2 生成的沙箱测试用例，包含 setup/trigger/cleanup 与来源记录。"""

    case_id: str
    seed_id: str
    attack_family: str
    delivery_mode: str
    setup: JsonDict
    trigger: JsonDict
    expected_signal: JsonDict
    cleanup: JsonDict
    executor: str
    quality_score: float
    provenance: JsonDict
    validation_result: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "GeneratedCase":
        """恢复生成用例；validation_result 缺省为空，便于导入外部样本。"""
        return cls(
            case_id=str(data["case_id"]),
            seed_id=str(data["seed_id"]),
            attack_family=str(data["attack_family"]),
            delivery_mode=str(data["delivery_mode"]),
            setup=dict(data.get("setup", {})),
            trigger=dict(data.get("trigger", {})),
            expected_signal=dict(data.get("expected_signal", {})),
            cleanup=dict(data.get("cleanup", {})),
            executor=str(data.get("executor", "")),
            quality_score=float(data.get("quality_score", 0.0)),
            provenance=dict(data.get("provenance", {})),
            validation_result=dict(data.get("validation_result", {})),
        )


@dataclass
class RunResult:
    """执行器回传的用例运行结果，后续会用于 seed 置信度反馈。"""

    run_id: str
    analysis_id: str
    seed_id: str
    case_id: str
    failure_stage: str
    metrics: JsonDict
    feedback: JsonDict = field(default_factory=dict)


def _normalize_artifacts(value: Any) -> list[JsonDict]:
    """允许 optional_artifacts 使用路径字符串或完整对象两种写法。"""
    artifacts: list[JsonDict] = []
    for item in value:
        if isinstance(item, str):
            artifacts.append({"path": item, "kind": "file"})
        elif isinstance(item, dict):
            artifacts.append(dict(item))
    return artifacts
