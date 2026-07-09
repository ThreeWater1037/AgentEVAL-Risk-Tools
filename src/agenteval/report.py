"""运行目录的 Markdown 报告生成。

报告只读取 Tool1/Tool2/执行器输出并汇总，不重新计算 seed 或重新执行 case。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .experiment import summarize_run_root
from .io import load_json
from .schemas import GeneratedCase, RiskSeed, RunResult


def build_run_markdown(run_root: str | Path) -> str:
    """根据运行目录生成面向阅读的 Tool1/Tool2 汇总报告。"""
    root = Path(run_root)
    summary = summarize_run_root(root)
    lines: list[str] = []
    lines.append("# Tool1/Tool2 Evaluation Summary")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Agents: {summary['agents']}")
    lines.append(f"- Risk seeds: {summary['seeds']}")
    lines.append(f"- Generated cases: {summary['cases']}")
    lines.append(f"- Result records: {summary['results']}")
    lines.append(f"- Schema valid rate: {summary['schema_valid_rate']:.2%}")
    lines.append(f"- Executable rate: {summary['executable_rate']:.2%}")
    lines.append(f"- Sandbox success rate: {summary['sandbox_success_rate']:.2%}")
    if "tool1_precision_proxy" in summary:
        lines.append(f"- Tool1 precision proxy: {summary['tool1_precision_proxy']:.2%}")
        lines.append(f"- Tool1 recall proxy: {summary['tool1_recall_proxy']:.2%}")
        lines.append(f"- Invalid test reduction vs all domains: {summary['invalid_test_reduction_vs_all_domains']:.2%}")
    lines.append("")
    lines.append(
        "Simulation note: sandbox results validate orchestration, schema, and provenance. "
        "Live ASR should replace these rows after real executors are attached."
    )
    lines.append("")

    lines.append("## Risk Seeds By Domain")
    lines.append("")
    lines.append("| Risk domain | Seeds |")
    lines.append("|---|---:|")
    for domain, count in sorted(summary["risk_domains"].items()):
        lines.append(f"| {domain} | {count} |")
    lines.append("")

    # 详细 case/result 统计来自各 Agent 子目录，而总体指标来自 summary。
    family_counts, subtype_counts, result_stage_counts = _collect_case_result_counts(root)
    lines.append("## Cases By Family")
    lines.append("")
    lines.append("| Attack family | Cases |")
    lines.append("|---|---:|")
    for family, count in sorted(family_counts.items()):
        lines.append(f"| {family} | {count} |")
    lines.append("")

    lines.append("## Result Stages")
    lines.append("")
    lines.append("| Stage | Records |")
    lines.append("|---|---:|")
    for stage, count in sorted(result_stage_counts.items()):
        lines.append(f"| {stage} | {count} |")
    lines.append("")

    if subtype_counts:
        lines.append("## Top Case Subtypes")
        lines.append("")
        lines.append("| Family | Subtype | Cases |")
        lines.append("|---|---|---:|")
        for (family, subtype), count in subtype_counts.most_common(20):
            lines.append(f"| {family} | {subtype} | {count} |")
        lines.append("")

    lines.append("## Per-Agent Summary")
    lines.append("")
    lines.append("| Agent | Seeds | Cases | Valid cases | Executable cases | Sandbox success |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for item in summary["per_agent"]:
        lines.append(
            f"| {item['agent_ref']} | {item['seeds']} | {item['cases']} | "
            f"{item['valid_cases']} | {item['executable_cases']} | {item['sandbox_success']} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_run_markdown(run_root: str | Path, out: str | Path) -> Path:
    """生成并写出 Markdown 报告。"""
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_run_markdown(run_root), encoding="utf-8")
    return target


def _collect_case_result_counts(root: Path) -> tuple[Counter[str], Counter[tuple[str, str]], Counter[str]]:
    """统计 case family、subtype 和执行阶段，用于报告附表。"""
    family_counts: Counter[str] = Counter()
    subtype_counts: Counter[tuple[str, str]] = Counter()
    result_stage_counts: Counter[str] = Counter()
    for agent_dir in root.iterdir():
        if not agent_dir.is_dir():
            continue
        case_path = agent_dir / "generated_cases.json"
        if case_path.exists():
            for record in load_json(case_path):
                case = GeneratedCase.from_dict(record)
                family_counts[case.attack_family] += 1
                subtype = str(case.provenance.get("subtype", case.provenance.get("mutation_strategy", "default")))
                subtype_counts[(case.attack_family, subtype)] += 1
        result_path = agent_dir / "run_result.json"
        if result_path.exists():
            for record in load_json(result_path):
                result_stage_counts[str(record.get("failure_stage", "unknown"))] += 1
    return family_counts, subtype_counts, result_stage_counts
