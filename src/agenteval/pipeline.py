"""AgentEVAL 对外稳定编排接口。

默认主路径只负责 ``Agent -> Tool1 -> RiskSeed -> Tool2 -> Case``，并输出一个
可交给独立下游执行器的 execution bundle。真实执行结果通过 submit_results 回传；
deterministic sandbox 仅在调用方显式要求时执行。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adapters import load_agent_descriptors
from .experiment import DEFAULT_EXECUTOR_REGISTRY, ExecutorRegistry, SandboxExecutor
from .feedback import apply_feedback_to_analysis
from .io import ensure_dir, load_json, write_json
from .llm import DeepSeekJSONClient
from .schemas import (
    AgentAccessDescriptor,
    AgentSnapshot,
    AnalysisSession,
    ExecutionContext,
    FAILURE_STAGES,
    GeneratedCase,
    RiskSeed,
    RunResult,
    to_jsonable,
)
from .structured import load_structured_file
from .tool1 import Tool1Analyzer
from .tool2 import Tool2Generator


EXECUTION_SCHEMA_VERSION = "agenteval.execution.v1"
RESULT_SCHEMA_VERSION = "agenteval.results.v1"
EVALUATION_SCHEMA_VERSION = "agenteval.evaluation.v1"
RESULT_STAGES = FAILURE_STAGES


@dataclass(frozen=True)
class PipelineOptions:
    """统一 CLI、Python API 和 HTTP API 的主路径选项。"""

    count: int = 1
    profile: str = "compact"
    dynamic_probe: bool = True
    llm: bool | None = False
    llm_evidence: bool | None = None
    llm_runtime_events: bool | None = None
    llm_review: bool | None = None
    llm_variants: bool | None = None
    llm_siraj_enrichment: bool | None = None
    use_siraj_prompts: bool = True
    strategies: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("count must be at least 1")
        if self.profile not in {"compact", "expanded"}:
            raise ValueError("profile must be compact or expanded")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> "PipelineOptions":
        data = data or {}
        strategies = data.get("strategies")
        if "use_siraj_prompts" in data or "siraj_prompts" in data:
            use_siraj_prompts = _bool_value(data.get("use_siraj_prompts", data.get("siraj_prompts")))
        else:
            use_siraj_prompts = not _bool_value(data.get("legacy_prompts", False))
        return cls(
            count=int(data.get("count", data.get("case_count", 1))),
            profile=str(data.get("profile", "compact")),
            dynamic_probe=_bool_value(data.get("dynamic_probe", data.get("enable_dynamic_probe", True))),
            llm=_llm_value(data.get("llm", "off")),
            llm_evidence=_optional_bool(data.get("llm_evidence", data.get("enable_llm_evidence"))),
            llm_runtime_events=_optional_bool(data.get("llm_runtime_events", data.get("enable_llm_runtime_events"))),
            llm_review=_optional_bool(data.get("llm_review", data.get("enable_llm_review"))),
            llm_variants=_optional_bool(data.get("llm_variants", data.get("enable_llm_variants"))),
            llm_siraj_enrichment=_optional_bool(data.get("llm_siraj_enrichment")),
            use_siraj_prompts=use_siraj_prompts,
            strategies=tuple(str(item) for item in strategies) if strategies else None,
        )

    def llm_for(self, stage: str) -> bool | None:
        value = getattr(self, stage)
        return self.llm if value is None else value


@dataclass
class PreparedEvaluation:
    """一次已准备完成、等待下游执行的评估。"""

    output_dir: Path
    session: AnalysisSession
    snapshot: AgentSnapshot
    seeds: list[RiskSeed]
    cases: list[GeneratedCase]
    options: PipelineOptions

    @property
    def evaluation_id(self) -> str:
        return self.session.analysis_id

    @property
    def status(self) -> str:
        return "ready_for_execution" if self.cases else "no_executable_cases"

    def execution_context(self) -> ExecutionContext:
        return ExecutionContext(
            analysis_id=self.session.analysis_id,
            analysis_dir=str(self.output_dir),
            agent_access=self.session.agent_access,
            snapshot=self.snapshot,
            sandbox_policy=dict(self.session.sandbox_policy),
            defense_config=dict(self.session.agent_access.defense_config),
        )

    def execution_bundle(self) -> dict[str, Any]:
        target = _public_agent_access(self.session.agent_access)
        executors: dict[str, int] = {}
        for case in self.cases:
            executors[case.executor] = executors.get(case.executor, 0) + 1
        return {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "evaluation_id": self.evaluation_id,
            "analysis_id": self.evaluation_id,
            "status": self.status,
            "target": target,
            "context": {
                "agent_ref": self.snapshot.agent_ref,
                "connector_type": self.snapshot.connector_type,
                "capabilities": self.snapshot.capabilities,
                "tool_schemas": self.snapshot.tool_schemas,
                "sandbox_policy": self.session.sandbox_policy,
                "defense_config": self.session.agent_access.defense_config,
            },
            "summary": {
                "seed_count": len(self.seeds),
                "case_count": len(self.cases),
                "cases_per_seed": self.options.count,
                "profile": self.options.profile,
                "executors": executors,
            },
            "seeds": [to_jsonable(seed) for seed in self.seeds],
            "risk_seeds": [to_jsonable(seed) for seed in self.seeds],
            "cases": [to_jsonable(case) for case in self.cases],
            "result_contract": {
                "schema_version": RESULT_SCHEMA_VERSION,
                "required_per_case": ["case_id", "failure_stage", "metrics"],
                "optional_per_case": ["run_id", "feedback"],
                "failure_stages": sorted(RESULT_STAGES),
            },
        }

    def summary(self, *, include_bundle: bool = False) -> dict[str, Any]:
        value = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "evaluation_id": self.evaluation_id,
            "analysis_id": self.evaluation_id,
            "analysis_dir": str(self.output_dir),
            "agent_ref": self.snapshot.agent_ref,
            "status": self.status,
            "evidence_count": len(self.snapshot.evidence_index),
            "seed_count": len(self.seeds),
            "case_count": len(self.cases),
            "execution_bundle_path": str(self.output_dir / "execution_bundle.json"),
        }
        if include_bundle:
            value["execution_bundle"] = self.execution_bundle()
        return value


class AgentEval:
    """CLI、HTTP API 和 Python 调用共同使用的高层门面。"""

    def __init__(self, run_root: str | Path = "runs", executor_registry: ExecutorRegistry | None = None):
        self.run_root = Path(run_root)
        self.executor_registry = executor_registry or DEFAULT_EXECUTOR_REGISTRY

    def prepare(
        self,
        target: AgentAccessDescriptor | Mapping[str, Any] | str | Path,
        *,
        out_dir: str | Path | None = None,
        agent_ref: str | None = None,
        options: PipelineOptions | Mapping[str, Any] | None = None,
        count: int | None = None,
        profile: str | None = None,
        llm: str | bool | None = None,
        dynamic_probe: bool | None = None,
        use_siraj_prompts: bool | None = None,
    ) -> PreparedEvaluation:
        """分析一个 Agent、生成 cases，并写出 execution_bundle.json。"""
        descriptor = self._descriptor(target, agent_ref=agent_ref)
        configured = options if isinstance(options, PipelineOptions) else PipelineOptions.from_mapping(options)
        overrides: dict[str, Any] = {}
        if count is not None:
            overrides["count"] = count
        if profile is not None:
            overrides["profile"] = profile
        if llm is not None:
            overrides["llm"] = _llm_value(llm)
        if dynamic_probe is not None:
            overrides["dynamic_probe"] = dynamic_probe
        if use_siraj_prompts is not None:
            overrides["use_siraj_prompts"] = use_siraj_prompts
        configured = replace(configured, **overrides) if overrides else configured
        llm_settings = (
            configured.llm,
            configured.llm_evidence,
            configured.llm_runtime_events,
            configured.llm_review,
            configured.llm_variants,
            configured.llm_siraj_enrichment,
        )
        if any(value is True for value in llm_settings) and not DeepSeekJSONClient().available:
            raise ValueError("llm=on requires DEEPSEEK_API_KEY")

        analyzer = Tool1Analyzer(
            enable_dynamic_probe=configured.dynamic_probe,
            enable_llm_evidence=configured.llm_for("llm_evidence"),
            enable_llm_runtime_events=configured.llm_for("llm_runtime_events"),
            enable_llm_review=configured.llm_for("llm_review"),
            enable_llm_siraj_enrichment=configured.llm_for("llm_siraj_enrichment"),
        )
        session, snapshot, seeds = analyzer.analyze(descriptor)
        output_dir = ensure_dir(out_dir or self.run_root / session.analysis_id)
        # 复用显式输出目录时，只清除上一轮的下游状态，避免新 case 搭配旧结果。
        for stale_name in (
            "generated_cases.json",
            "execution_bundle.json",
            "pipeline_summary.json",
            "run_result.json",
            "feedback_summary.json",
            "generation_job.json",
        ):
            stale_path = output_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()
        analyzer.write_outputs(output_dir, session, snapshot, seeds)
        cases = Tool2Generator(
            enable_llm_variants=configured.llm_for("llm_variants"),
            available_executors=self.executor_registry.names(),
        ).generate(
            snapshot,
            seeds,
            count=configured.count,
            strategies=list(configured.strategies) if configured.strategies else None,
            out_dir=output_dir,
            profile=configured.profile,
            use_siraj_prompts=configured.use_siraj_prompts,
        )
        evaluation = PreparedEvaluation(output_dir, session, snapshot, seeds, cases, configured)
        write_json(output_dir / "execution_bundle.json", evaluation.execution_bundle())
        write_json(output_dir / "pipeline_summary.json", evaluation.summary())
        return evaluation

    def load(self, evaluation: str | Path) -> PreparedEvaluation:
        """从固定产物文件恢复一次评估。"""
        root = self._analysis_dir(evaluation)
        session = AnalysisSession.from_dict(load_json(root / "analysis_session.json"))
        snapshot = AgentSnapshot.from_dict(load_json(root / "agent_snapshot.json"))
        seeds = [RiskSeed.from_dict(item) for item in load_json(root / "risk_seeds.json")]
        cases = [GeneratedCase.from_dict(item) for item in load_json(root / "generated_cases.json")]
        summary_path = root / "execution_bundle.json"
        summary = load_json(summary_path).get("summary", {}) if summary_path.exists() else {}
        options = PipelineOptions(count=int(summary.get("cases_per_seed", 1)), profile=str(summary.get("profile", "compact")))
        return PreparedEvaluation(root, session, snapshot, seeds, cases, options)

    def execute(
        self,
        evaluation: PreparedEvaluation | str | Path,
        *,
        allow_sandbox_fallback: bool = False,
        apply_feedback: bool = False,
    ) -> list[RunResult]:
        """显式执行 cases；真实模式默认 fail closed，不静默降级 sandbox。"""
        prepared = evaluation if isinstance(evaluation, PreparedEvaluation) else self.load(evaluation)
        results = self.executor_registry.run(
            prepared.execution_context(),
            prepared.cases,
            allow_fallback=allow_sandbox_fallback,
        )
        write_json(prepared.output_dir / "run_result.json", results)
        if apply_feedback:
            apply_feedback_to_analysis(prepared.output_dir)
        return results

    def execute_sandbox(
        self,
        evaluation: PreparedEvaluation | str | Path,
        *,
        apply_feedback: bool = False,
    ) -> list[RunResult]:
        """强制使用内置代理沙箱，绝不分发到已注册的真实执行器。"""
        prepared = evaluation if isinstance(evaluation, PreparedEvaluation) else self.load(evaluation)
        results = SandboxExecutor().run(prepared.evaluation_id, prepared.cases)
        write_json(prepared.output_dir / "run_result.json", results)
        if apply_feedback:
            apply_feedback_to_analysis(prepared.output_dir)
        return results

    def submit_results(
        self,
        evaluation: PreparedEvaluation | str | Path,
        results: Sequence[RunResult | Mapping[str, Any]] | Mapping[str, Any] | str | Path,
        *,
        apply_feedback: bool = True,
    ) -> dict[str, Any]:
        """接收下游最小结果，补齐 ID、严格校验并写回反馈。"""
        prepared = evaluation if isinstance(evaluation, PreparedEvaluation) else self.load(evaluation)
        payload = load_structured_file(results) if isinstance(results, (str, Path)) else results
        wrapper: Mapping[str, Any] = payload if isinstance(payload, Mapping) else {}
        if wrapper:
            schema_version = wrapper.get("schema_version")
            if schema_version and str(schema_version) != RESULT_SCHEMA_VERSION:
                raise ValueError(f"unsupported result schema_version: {schema_version}")
            for id_field in ("evaluation_id", "analysis_id"):
                submitted_id = wrapper.get(id_field)
                if submitted_id and str(submitted_id) != prepared.evaluation_id:
                    raise ValueError(f"result {id_field} does not match the analysis")
            records = wrapper.get("results")
            if records is None and "case_id" in wrapper:
                records = [wrapper]
        else:
            records = payload
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ValueError("results must be a list or an object containing a results list")

        case_by_id = {case.case_id: case for case in prepared.cases}
        normalized: list[RunResult] = []
        seen_cases: set[str] = set()
        seen_runs: set[str] = set()
        for raw in records:
            if isinstance(raw, RunResult):
                item = to_jsonable(raw)
            elif isinstance(raw, Mapping):
                item = dict(raw)
            else:
                raise ValueError("each result must be an object")
            case_id = str(item.get("case_id", ""))
            if case_id not in case_by_id:
                raise ValueError(f"unknown case_id: {case_id or '<missing>'}")
            if case_id in seen_cases:
                raise ValueError(f"duplicate case_id in results: {case_id}")
            case = case_by_id[case_id]
            stage = item.get("failure_stage", item.get("outcome"))
            if not stage:
                raise ValueError(f"missing failure_stage for case: {case_id}")
            if str(stage) not in RESULT_STAGES:
                raise ValueError(f"unsupported failure_stage for case {case_id}: {stage}")
            if "metrics" not in item or not isinstance(item.get("metrics"), Mapping):
                raise ValueError(f"metrics must be an object for case: {case_id}")
            if item.get("analysis_id") and str(item["analysis_id"]) != prepared.evaluation_id:
                raise ValueError(f"analysis_id mismatch for case: {case_id}")
            if item.get("seed_id") and str(item["seed_id"]) != case.seed_id:
                raise ValueError(f"seed_id mismatch for case: {case_id}")
            run_id = str(item.get("run_id") or f"run_external_{case_id}")
            if run_id in seen_runs:
                raise ValueError(f"duplicate run_id in results: {run_id}")
            feedback = dict(item.get("feedback") or {})
            if item.get("error"):
                feedback.setdefault("error", str(item["error"]))
            normalized.append(
                RunResult(
                    run_id=run_id,
                    analysis_id=prepared.evaluation_id,
                    seed_id=case.seed_id,
                    case_id=case_id,
                    failure_stage=str(stage),
                    metrics=dict(item.get("metrics") or {}),
                    feedback=feedback,
                )
            )
            seen_cases.add(case_id)
            seen_runs.add(run_id)

        missing = set(case_by_id) - seen_cases
        if missing:
            raise ValueError(f"missing results for {len(missing)} case(s)")
        write_json(prepared.output_dir / "run_result.json", normalized)
        feedback_summary = apply_feedback_to_analysis(prepared.output_dir) if apply_feedback else None
        submission = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "evaluation_id": prepared.evaluation_id,
            "accepted": len(normalized),
            "status": "feedback_applied" if apply_feedback else "results_saved",
            "run_result_path": str(prepared.output_dir / "run_result.json"),
        }
        if feedback_summary is not None:
            submission["feedback"] = feedback_summary
        return submission

    def _descriptor(
        self,
        target: AgentAccessDescriptor | Mapping[str, Any] | str | Path,
        *,
        agent_ref: str | None,
    ) -> AgentAccessDescriptor:
        if isinstance(target, AgentAccessDescriptor):
            descriptors = [target]
        elif isinstance(target, (str, Path)):
            descriptors = load_agent_descriptors(target)
        elif isinstance(target, Mapping):
            nested = target.get("target") or target.get("agent") or target.get("agent_access")
            descriptors = [AgentAccessDescriptor.from_dict(dict(nested or target))]
        else:
            raise TypeError("target must be an AgentAccessDescriptor, mapping, or file path")
        if agent_ref:
            descriptors = [item for item in descriptors if item.agent_ref == agent_ref]
            if not descriptors:
                raise ValueError(f"agent not found: {agent_ref}")
        if len(descriptors) != 1:
            raise ValueError("the main run interface requires one Agent; use --select for a multi-Agent file")
        return descriptors[0]

    def _analysis_dir(self, evaluation: str | Path) -> Path:
        direct = Path(evaluation)
        if direct.exists():
            root = direct
        else:
            base = self.run_root.resolve()
            root = (base / direct).resolve()
            if not root.is_relative_to(base):
                raise ValueError("evaluation id escapes run_root")
        if not root.is_dir():
            raise FileNotFoundError(f"analysis directory not found: {root}")
        return root


def _llm_value(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"auto", "default", ""}:
        return None
    if normalized in {"on", "true", "1", "yes"}:
        return True
    if normalized in {"off", "false", "0", "no"}:
        return False
    raise ValueError("llm must be auto, on, or off")


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else _bool_value(value)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off", "none", ""}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _public_agent_access(descriptor: AgentAccessDescriptor) -> dict[str, Any]:
    value = to_jsonable(descriptor)
    headers = dict(value.get("headers") or {})
    sensitive = {descriptor.auth_header.lower(), "authorization", "proxy-authorization", "cookie", "x-api-key"}
    for key in list(headers):
        if key.lower() in sensitive or any(token in key.lower() for token in ("token", "secret", "api-key")):
            headers[key] = "<redacted>"
    value["headers"] = headers
    return value
