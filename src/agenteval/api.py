"""FastAPI 服务入口。

API 端点复用 CLI 的同一套 Tool1/Tool2/执行器逻辑，并把每个 analysis_id 映射到
run_root 下的一个目录。默认生成仍走 SIRAJ prompt；legacy_prompts 仅作为显式回退。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .experiment import DEFAULT_EXECUTOR_REGISTRY
from .feedback import apply_feedback_to_analysis
from .io import ensure_dir, load_json, write_json
from .schemas import AgentAccessDescriptor, AgentSnapshot, GeneratedCase, RiskSeed
from .tool1 import Tool1Analyzer
from .tool2 import Tool2Generator

try:
    from fastapi import FastAPI, HTTPException
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    FastAPI = None  # type: ignore
    HTTPException = RuntimeError  # type: ignore


def create_app(run_root: str | Path = "runs/api_sessions"):
    """创建 FastAPI 应用，并把运行状态持久化到 run_root。"""
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed.")

    root = ensure_dir(run_root)
    app = FastAPI(title="AgentEVAL Tool1/Tool2 API", version="0.1.0")

    @app.post("/api/risk-discovery/analyze")
    def analyze(payload: dict[str, Any]) -> dict[str, Any]:
        """运行 Tool1，并返回 analysis_id、能力摘要和 seed 列表。"""
        descriptor_payload = payload.get("agent_access") or payload
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
        results = DEFAULT_EXECUTOR_REGISTRY.run(analysis_id, cases)
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


app = create_app()


def _analysis_dir(root: Path, analysis_id: str) -> Path:
    """校验 analysis 目录存在，并在缺失时转成 HTTP 404。"""
    path = root / analysis_id
    if not path.exists():
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
