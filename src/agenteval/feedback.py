"""执行反馈到 RiskSeed 置信度的轻量闭环。

该模块读取 run_result.json 和 generated_cases.json，只对已有 seed 的 confidence
和 status 做小幅调整，并记录原因；不会新增 seed，也不会覆盖 Tool1 的证据链。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import load_json, write_json
from .schemas import RiskSeed, utc_now_iso


STAGE_ADJUSTMENTS = {
    # failure_stage 到置信度调整量的代理映射，幅度故意较小，避免单轮结果过拟合。
    "attack_success": 0.06,
    "action_blocked": 0.03,
    "retrieved_not_adopted": 0.02,
    "adopted_no_action": 0.01,
    "require_review": -0.03,
    "not_triggered": -0.04,
    "setup_failed": -0.08,
}


def apply_feedback_to_analysis(analysis_dir: str | Path) -> dict[str, Any]:
    """读取一次分析目录的执行结果，回写 risk_seeds.json 和 feedback_summary.json。"""
    root = Path(analysis_dir)
    print(f"【反馈】开始处理执行反馈：{root}")
    seeds_path = root / "risk_seeds.json"
    results_path = root / "run_result.json"
    cases_path = root / "generated_cases.json"
    if not seeds_path.exists():
        raise FileNotFoundError(f"missing risk_seeds.json: {seeds_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"missing run_result.json: {results_path}")

    seeds = [RiskSeed.from_dict(item) for item in load_json(seeds_path)]
    results = load_json(results_path)
    cases = load_json(cases_path) if cases_path.exists() else []
    case_quality = {str(item.get("case_id")): float(item.get("quality_score", 0.0)) for item in cases}
    by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_seed[str(result.get("seed_id", ""))].append(result)

    changed = 0
    per_seed = []
    for seed in seeds:
        seed_results = by_seed.get(seed.seed_id, [])
        if not seed_results:
            continue
        before = seed.confidence
        stages = Counter(str(item.get("failure_stage", "unknown")) for item in seed_results)
        # stage 调整反映触发轨迹，quality 调整反映 Tool2 case 自身质量。
        stage_adjustment = sum(STAGE_ADJUSTMENTS.get(stage, -0.01) * count for stage, count in stages.items()) / max(1, len(seed_results))
        quality_adjustment = _quality_adjustment(seed_results, case_quality)
        adjustment = round(stage_adjustment + quality_adjustment, 3)
        after = _clamp(round(before + adjustment, 3))
        seed.confidence = after
        seed.status = _status_from_confidence(after)
        seed.score_detail["feedback"] = {
            "updated_at": utc_now_iso(),
            "confidence_before": before,
            "confidence_after": after,
            "adjustment": adjustment,
            "observed_failure_stages": dict(stages),
            "result_count": len(seed_results),
            "reason": _feedback_reason(stages),
        }
        changed += 1
        per_seed.append(
            {
                "seed_id": seed.seed_id,
                "confidence_before": before,
                "confidence_after": after,
                "status": seed.status,
                "observed_failure_stages": dict(stages),
            }
        )

    write_json(seeds_path, seeds)
    summary = {
        "analysis_dir": str(root),
        "updated_seeds": changed,
        "total_seeds": len(seeds),
        "result_records": len(results),
        "per_seed": per_seed,
    }
    write_json(root / "feedback_summary.json", summary)
    print(f"【反馈】完成：更新seed数={changed}，总seed数={len(seeds)}，结果记录数={len(results)}")
    return summary


def _quality_adjustment(results: list[dict[str, Any]], case_quality: dict[str, float]) -> float:
    """用相关 case 的平均质量分补充微调 seed 置信度。"""
    qualities = [case_quality.get(str(item.get("case_id", "")), 0.0) for item in results]
    if not qualities:
        return 0.0
    avg = sum(qualities) / len(qualities)
    if avg >= 0.85:
        return 0.015
    if avg < 0.60:
        return -0.025
    return 0.0


def _feedback_reason(stages: Counter[str]) -> str:
    """把主要执行阶段转成可读反馈原因，便于报告和调试。"""
    if stages.get("attack_success"):
        return "execution evidence confirmed at least one generated case"
    if stages.get("setup_failed"):
        return "setup failed for generated case; reduce confidence until executor or case is fixed"
    if stages.get("not_triggered"):
        return "case did not trigger observable target behavior"
    if stages.get("retrieved_not_adopted") or stages.get("adopted_no_action"):
        return "preconditions partially held, but downstream adoption/action was incomplete"
    return "feedback applied from execution failure stages"


def _status_from_confidence(confidence: float) -> str:
    return "auto_generate" if confidence >= 0.75 else "review" if confidence >= 0.50 else "candidate"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
