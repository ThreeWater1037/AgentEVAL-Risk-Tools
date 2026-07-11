"""FastAPI 服务入口。

API 端点复用 CLI 的同一套 Tool1/Tool2/执行器逻辑，并把每个 analysis_id 映射到
run_root 下的一个目录。默认生成仍走 SIRAJ prompt；legacy_prompts 仅作为显式回退。
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Any

from .experiment import DEFAULT_EXECUTOR_REGISTRY, ExecutorRegistry, SandboxExecutor
from .feedback import apply_feedback_to_analysis
from .io import ensure_dir, load_json, write_json
from .pipeline import AgentEval, PipelineOptions
from .schemas import AgentAccessDescriptor, AgentSnapshot, GeneratedCase, RiskSeed
from .tool1 import Tool1Analyzer
from .tool2 import Tool2Generator

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    FastAPI = None  # type: ignore
    HTTPException = RuntimeError  # type: ignore
    Request = Any  # type: ignore
    JSONResponse = None  # type: ignore
    BaseModel = object  # type: ignore
    Field = None  # type: ignore


if FastAPI is not None:
    class EvaluationRequest(BaseModel):
        target: dict[str, Any] | None = None
        agent: dict[str, Any] | None = None
        agent_access: dict[str, Any] | None = None
        count: int = 1
        profile: str = "compact"
        llm: str = "off"
        dynamic_probe: bool = True
        legacy_prompts: bool = False
        execute_sandbox: bool = False
        options: dict[str, Any] = Field(default_factory=dict)


    class ResultSubmissionRequest(BaseModel):
        schema_version: str | None = None
        evaluation_id: str | None = None
        analysis_id: str | None = None
        results: list[dict[str, Any]] = Field(default_factory=list)
        apply_feedback: bool = True


def create_app(
    run_root: str | Path = "runs/api_sessions",
    *,
    executor_registry: ExecutorRegistry | None = None,
    allow_local_execution: bool | None = None,
    allow_target_auth: bool | None = None,
    api_token: str | None = None,
):
    """创建 FastAPI 应用，并把运行状态持久化到 run_root。"""
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed.")

    root = ensure_dir(run_root)
    registry = executor_registry or DEFAULT_EXECUTOR_REGISTRY
    service = AgentEval(root, executor_registry=registry)
    allow_local = _env_bool("AGENTEVAL_ALLOW_LOCAL_EXECUTION", False) if allow_local_execution is None else allow_local_execution
    allow_auth = _env_bool("AGENTEVAL_ALLOW_TARGET_AUTH", False) if allow_target_auth is None else allow_target_auth
    expected_token = os.environ.get("AGENTEVAL_API_TOKEN", "") if api_token is None else api_token
    app = FastAPI(title="AgentEVAL API", version="1.0.0")

    if expected_token:
        @app.middleware("http")
        async def require_api_token(request: Request, call_next):
            if request.url.path == "/healthz":
                return await call_next(request)
            supplied = request.headers.get("x-api-key", "")
            authorization = request.headers.get("authorization", "")
            if not supplied and authorization.lower().startswith("bearer "):
                supplied = authorization[7:]
            if not hmac.compare_digest(supplied, expected_token):
                return JSONResponse(status_code=401, content={"detail": "invalid or missing API token"})
            return await call_next(request)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "service": "agenteval"}

    @app.post("/api/v1/evaluations", status_code=201)
    def prepare_evaluation(payload: EvaluationRequest) -> dict[str, Any]:
        """一请求完成 Tool1+Tool2，直接返回可交给下游的执行包。"""
        target = payload.target or payload.agent or payload.agent_access
        if not target:
            raise HTTPException(status_code=400, detail="target is required")
        _ensure_protocol_allowed(target, allow_local, allow_auth)
        option_data = dict(payload.options)
        option_data.update(
            {
                "count": payload.count,
                "profile": payload.profile,
                "llm": payload.llm,
                "dynamic_probe": payload.dynamic_probe,
                "legacy_prompts": payload.legacy_prompts,
            }
        )
        try:
            evaluation = service.prepare(target, options=PipelineOptions.from_mapping(option_data))
            response = evaluation.summary(include_bundle=True)
            if payload.execute_sandbox:
                results = service.execute_sandbox(evaluation)
                response["status"] = "sandbox_complete"
                response["sandbox_result_count"] = len(results)
            return response
        except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/v1/evaluations/{evaluation_id}")
    def get_evaluation(evaluation_id: str) -> dict[str, Any]:
        analysis_dir = _analysis_dir(root, evaluation_id)
        try:
            evaluation = service.load(analysis_dir)
        except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        response = evaluation.summary()
        bundle_path = analysis_dir / "execution_bundle.json"
        response["execution_bundle"] = load_json(bundle_path) if bundle_path.exists() else evaluation.execution_bundle()
        response["run_results"] = _load_optional(analysis_dir / "run_result.json", [])
        response["feedback"] = _load_optional(analysis_dir / "feedback_summary.json", {})
        if response["run_results"]:
            response["status"] = "results_received"
        return response

    @app.post("/api/v1/evaluations/{evaluation_id}/results")
    def submit_evaluation_results(evaluation_id: str, payload: ResultSubmissionRequest) -> dict[str, Any]:
        analysis_dir = _analysis_dir(root, evaluation_id)
        body = _model_dump(payload)
        supplied_id = body.get("evaluation_id") or body.get("analysis_id")
        if supplied_id and str(supplied_id) != evaluation_id:
            raise HTTPException(status_code=400, detail="result evaluation_id does not match the URL")
        body["evaluation_id"] = body.get("evaluation_id") or evaluation_id
        apply_feedback = bool(body.pop("apply_feedback", True))
        try:
            return service.submit_results(
                analysis_dir,
                body,
                apply_feedback=apply_feedback,
            )
        except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/risk-discovery/analyze")
    def analyze(payload: dict[str, Any]) -> dict[str, Any]:
        """运行 Tool1，并返回 analysis_id、能力摘要和 seed 列表。"""
        descriptor_payload = payload.get("agent_access") or payload
        _ensure_protocol_allowed(descriptor_payload, allow_local, allow_auth)
        descriptor = AgentAccessDescriptor.from_dict(descriptor_payload)
        analyzer = Tool1Analyzer(
            enable_dynamic_probe=bool(payload.get("enable_dynamic_probe", True)),
            enable_llm_evidence=payload.get("llm_evidence"),
            enable_llm_runtime_events=payload.get("llm_runtime_events"),
            enable_llm_review=payload.get("llm_review"),
        )
        session, snapshot, seeds = analyzer.analyze(descriptor)
        analysis_dir = ensure_dir(root / session.analysis_id)
        analyzer.write_outputs(analysis_dir, session, snapshot, seeds)
        return {
            "analysis_id": session.analysis_id,
            "analysis_dir": str(analysis_dir),
            "connection_status": snapshot.runtime_observations[0] if snapshot.runtime_observations else {},
            "capabilities": snapshot.capabilities,
            "seed_count": len(seeds),
            "seeds": [seed.seed_id for seed in seeds],
        }

    @app.get("/api/analysis-sessions/{analysis_id}")
    def get_session(analysis_id: str) -> dict[str, Any]:
        """读取某次分析的所有已生成文件；不存在的文件返回默认空值。"""
        analysis_dir = _analysis_dir(root, analysis_id)
        return {
            "analysis_id": analysis_id,
            "analysis_session": _load_optional(analysis_dir / "analysis_session.json"),
            "agent_snapshot": _load_optional(analysis_dir / "agent_snapshot.json"),
            "risk_seeds": _load_optional(analysis_dir / "risk_seeds.json", []),
            "generated_cases": _load_optional(analysis_dir / "generated_cases.json", []),
            "run_result": _load_optional(analysis_dir / "run_result.json", []),
        }

    @app.post("/api/case-generation/generate")
    def generate(payload: dict[str, Any]) -> dict[str, Any]:
        """按 analysis_id 或 seed_id 生成 Tool2 cases，并写 generation_job.json。"""
        analysis_id, analysis_dir = _resolve_analysis(root, payload)
        snapshot = AgentSnapshot.from_dict(load_json(analysis_dir / "agent_snapshot.json"))
        seeds = [RiskSeed.from_dict(item) for item in load_json(analysis_dir / "risk_seeds.json")]
        requested_seed_id = payload.get("seed_id")
        if requested_seed_id:
            # 允许前端只针对某一个 RiskSeed 生成 case。
            seeds = [seed for seed in seeds if seed.seed_id == requested_seed_id]
        if not seeds:
            raise HTTPException(status_code=404, detail="no matching seed found")
        cases = Tool2Generator(enable_llm_variants=payload.get("llm_variants")).generate(
            snapshot,
            seeds,
            count=int(payload.get("count", 3)),
            strategies=payload.get("strategies"),
            out_dir=analysis_dir,
            profile=str(payload.get("profile", "compact")),
            use_siraj_prompts=bool(payload.get("siraj_prompts", True)) and not bool(payload.get("legacy_prompts", False)),
        )
        job = {
            "job_id": f"generation_{analysis_id}",
            "analysis_id": analysis_id,
            "generated": len(cases),
            "valid": sum(1 for case in cases if case.validation_result.get("schema_valid")),
            "review": sum(1 for case in cases if case.quality_score < 0.80),
            "case_ids": [case.case_id for case in cases],
        }
        write_json(analysis_dir / "generation_job.json", job)
        return job

    @app.get("/api/generation-jobs/{job_id}")
    def generation_job(job_id: str) -> dict[str, Any]:
        """按 job_id 在 run_root 下查找最近写出的生成任务摘要。"""
        for analysis_dir in root.iterdir():
            job_path = analysis_dir / "generation_job.json"
            if analysis_dir.is_dir() and job_path.exists():
                job = load_json(job_path)
                if job.get("job_id") == job_id:
                    return job
        raise HTTPException(status_code=404, detail="generation job not found")

    @app.post("/api/experiments/from-seeds")
    def experiments_from_seeds(payload: dict[str, Any]) -> dict[str, Any]:
        """执行已生成 case；当前默认执行器为确定性 sandbox。"""
        analysis_id, analysis_dir = _resolve_analysis(root, payload)
        cases = [GeneratedCase.from_dict(item) for item in load_json(analysis_dir / "generated_cases.json")]
        requested_case_ids = set(payload.get("case_ids") or [])
        if requested_case_ids:
            cases = [case for case in cases if case.case_id in requested_case_ids]
        results = SandboxExecutor().run(analysis_id, cases)
        write_json(analysis_dir / "run_result.json", results)
        return {
            "analysis_id": analysis_id,
            "run_id": analysis_id,
            "results": len(results),
            "case_ids": [result.case_id for result in results],
        }

    @app.post("/api/results/{run_id}/feedback")
    def result_feedback(run_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """把执行结果反馈回对应 analysis 目录的 risk_seeds.json。"""
        payload = payload or {}
        analysis_id = str(payload.get("analysis_id") or run_id)
        analysis_dir = _analysis_dir(root, analysis_id)
        return apply_feedback_to_analysis(analysis_dir)

    return app

def _analysis_dir(root: Path, analysis_id: str) -> Path:
    """限制 ID 和目录边界，避免通过路径片段读取 run_root 外文件。"""
    if not analysis_id.startswith("analysis_") or Path(analysis_id).name != analysis_id:
        raise HTTPException(status_code=404, detail="invalid analysis id")
    resolved_root = root.resolve()
    path = (resolved_root / analysis_id).resolve()
    if not path.is_relative_to(resolved_root) or not path.is_dir():
        raise HTTPException(status_code=404, detail=f"analysis session not found: {analysis_id}")
    return path


def _resolve_analysis(root: Path, payload: dict[str, Any]) -> tuple[str, Path]:
    """优先用 analysis_id 定位；否则根据 seed_id 反查所属 analysis。"""
    analysis_id = payload.get("analysis_id")
    if analysis_id:
        return str(analysis_id), _analysis_dir(root, str(analysis_id))
    seed_id = str(payload.get("seed_id", ""))
    for analysis_dir in root.iterdir():
        seeds_path = analysis_dir / "risk_seeds.json"
        if not analysis_dir.is_dir() or not seeds_path.exists():
            continue
        for seed in load_json(seeds_path):
            if seed.get("seed_id") == seed_id:
                return str(seed.get("analysis_id") or analysis_dir.name), analysis_dir
    raise HTTPException(status_code=404, detail="analysis session not found")


def _load_optional(path: Path, default: Any | None = None) -> Any:
    """API 读取目录状态时容忍某些阶段尚未生成文件。"""
    if path.exists():
        return load_json(path)
    return default if default is not None else {}


def _ensure_protocol_allowed(
    payload: dict[str, Any],
    allow_local_execution: bool,
    allow_target_auth: bool,
) -> None:
    try:
        descriptor = AgentAccessDescriptor.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if descriptor.protocol.lower() in {"python", "python_function", "local_python", "runner", "local_runner", "subprocess"} and not allow_local_execution:
        raise HTTPException(
            status_code=403,
            detail="local python/runner targets are disabled for the API; set AGENTEVAL_ALLOW_LOCAL_EXECUTION=1 only in a trusted environment",
        )
    if not allow_local_execution and any(item.get("path") for item in descriptor.optional_artifacts):
        raise HTTPException(
            status_code=403,
            detail="local optional_artifact paths are disabled for the API; use inline text or a trusted local CLI run",
        )
    sensitive_headers = {
        str(name).lower()
        for name in descriptor.headers
        if str(name).lower() in {"authorization", "proxy-authorization", "cookie", "x-api-key"}
        or any(token in str(name).lower() for token in ("token", "secret", "api-key"))
    }
    if (descriptor.auth_ref or sensitive_headers) and not allow_target_auth:
        raise HTTPException(
            status_code=403,
            detail="target auth references and sensitive headers are disabled for the API; set AGENTEVAL_ALLOW_TARGET_AUTH=1 only for trusted clients and targets",
        )


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return dict(model.model_dump())
    return dict(model.dict())


app = create_app()
