"""Tool1/Tool2 的论文式代理评估。

该模块跑完整 `Agent -> Tool1 -> Risk Seed -> Tool2 -> Case -> Sandbox` 流程，
并生成 CSV/JSON/Markdown 表格。默认指标是 dry-run/proxy 指标，不代表真实攻击
成功率；真实执行器接入后可以复用同一表格入口。
"""

from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .experiment import DEFAULT_EXECUTOR_REGISTRY
from .feedback import apply_feedback_to_analysis
from .io import ensure_dir, load_json, write_json
from .llm import DeepSeekJSONClient, LLMUnavailable, truncate_text
from .prompts import load_prompt
from .schemas import AgentAccessDescriptor, AgentSnapshot, RiskSeed, GeneratedCase
from .tool1 import Tool1Analyzer
from .tool2 import Tool2Generator
from .tool2.templates import SAFE_MARKER


ALL_RISK_DOMAINS = (
    # 评估、baseline 和消融共同使用的风险域全集。
    "prompt_context_injection",
    "rag_poisoning",
    "memory_poisoning",
    "tool_output_injection",
    "mcp_description_poisoning",
    "planning_poisoning",
    "multi_agent_communication_poisoning",
    "search_narrative_poisoning",
)

EXECUTOR_BY_DOMAIN = {
    # 真实执行器未注册时，Tool2 dry-run 会落到 sandbox fallback。
    "prompt_context_injection": "prompt_orchestrator",
    "rag_poisoning": "rag_poison_runner",
    "memory_poisoning": "memory_runner",
    "tool_output_injection": "tool_output_runner",
    "mcp_description_poisoning": "mcp_runner",
    "planning_poisoning": "planning_trace_runner",
    "multi_agent_communication_poisoning": "multi_agent_runner",
    "search_narrative_poisoning": "search_rag_runner",
}


def evaluate_tool12(
    descriptors: list[AgentAccessDescriptor],
    out_dir: str | Path,
    labels: dict[str, list[str]] | None = None,
    count: int = 3,
    profile: str = "compact",
    enable_llm_evidence: bool | None = False,
    enable_llm_runtime_events: bool | None = False,
    enable_llm_review: bool | None = False,
    enable_llm_variants: bool | None = False,
    use_siraj_prompts: bool = True,
    include_direct_llm: bool = False,
    random_seed: int = 13,
) -> dict[str, Any]:
    """运行主方法、baseline、消融，并写出论文表格所需的所有中间指标。"""
    output = ensure_dir(out_dir)
    labels = labels or {}
    label_source = "explicit_labels" if labels else "descriptor_expected_domains"
    print(f"【实验】开始Tool1/Tool2评估：agents={len(descriptors)}，label_source={label_source}，输出目录={output}")

    full_rows: list[dict[str, Any]] = []
    tool1_rows: list[dict[str, Any]] = []
    tool2_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []

    for agent_index, descriptor in enumerate(descriptors, start=1):
        print(f"【实验】[{agent_index}/{len(descriptors)}] 开始处理Agent：{descriptor.agent_ref}")
        truth = set(labels.get(descriptor.agent_ref, descriptor.expected_domains))
        agent_dir = ensure_dir(output / "runs" / "ours" / _safe_name(descriptor.agent_ref))
        start = time.perf_counter()
        # 主方法：Tool1 先证据驱动发现 seed，Tool2 再按 seed 生成上下文绑定用例。
        analyzer = Tool1Analyzer(
            enable_dynamic_probe=True,
            enable_llm_evidence=enable_llm_evidence,
            enable_llm_runtime_events=enable_llm_runtime_events,
            enable_llm_review=enable_llm_review,
        )
        print(f"【Tool1】开始风险发现：agent={descriptor.agent_ref}")
        session, snapshot, seeds_before_feedback = analyzer.analyze(descriptor, agent_dir)
        discovery_cost_s = round(time.perf_counter() - start, 4)
        print(
            f"【Tool1】完成风险发现：agent={descriptor.agent_ref}，"
            f"evidence={len(snapshot.evidence_index)}，seeds={len(seeds_before_feedback)}，耗时={discovery_cost_s}s"
        )
        generator = Tool2Generator(enable_llm_variants=enable_llm_variants)
        print(f"【Tool2】开始生成测试用例：agent={descriptor.agent_ref}，count={count}，profile={profile}，siraj={use_siraj_prompts}")
        cases = generator.generate(
            snapshot,
            seeds_before_feedback,
            count=count,
            out_dir=agent_dir,
            profile=profile,
            use_siraj_prompts=use_siraj_prompts,
        )
        print(f"【Tool2】完成测试用例生成：agent={descriptor.agent_ref}，cases={len(cases)}")
        print(f"【执行器】开始执行测试用例：agent={descriptor.agent_ref}，cases={len(cases)}")
        results = DEFAULT_EXECUTOR_REGISTRY.run(session.analysis_id, cases)
        write_json(agent_dir / "run_result.json", results)
        print(f"【执行器】完成测试用例执行：agent={descriptor.agent_ref}，results={len(results)}")
        # 执行反馈会回写 risk_seeds.json，因此后面重新读取更新后的 seed。
        feedback_summary = apply_feedback_to_analysis(agent_dir)
        seeds = [RiskSeed.from_dict(item) for item in load_json(agent_dir / "risk_seeds.json")]

        print(f"【指标】开始计算Tool1/Tool2指标：agent={descriptor.agent_ref}")
        tool1 = compute_tool1_metrics(
            descriptor.agent_ref,
            "ours",
            seeds,
            snapshot,
            truth,
            label_source,
            discovery_cost_s=discovery_cost_s,
        )
        tool2 = compute_tool2_metrics(descriptor.agent_ref, "ours", cases, snapshot)
        tool1["feedback_updated_seeds"] = feedback_summary["updated_seeds"]
        full_rows.append({**tool1, **_prefixed(tool2, "tool2_")})
        tool1_rows.append(tool1)
        tool2_rows.append(tool2)
        print(
            f"【指标】完成主方法指标：agent={descriptor.agent_ref}，"
            f"seed_f1={tool1['seed_f1']}，schema_valid_rate={tool2['schema_valid_rate']}，"
            f"updated_seeds={feedback_summary['updated_seeds']}"
        )

        print(f"【Baseline】开始对比实验：agent={descriptor.agent_ref}")
        baseline_rows.extend(
            _evaluate_baselines(
                descriptor,
                snapshot,
                truth,
                label_source,
                count,
                profile,
                random_seed,
                include_direct_llm,
            )
        )
        print(f"【Baseline】完成对比实验：agent={descriptor.agent_ref}")
        print(f"【Ablation】开始消融实验：agent={descriptor.agent_ref}")
        ablation_rows.extend(
            _evaluate_ablations(
                descriptor,
                truth,
                label_source,
                count,
                profile,
                enable_llm_evidence,
                enable_llm_runtime_events,
                enable_llm_review,
                enable_llm_variants,
                use_siraj_prompts,
                output / "runs" / "ablations",
            )
        )
        print(f"【Ablation】完成消融实验：agent={descriptor.agent_ref}")

    aggregate = {
        "label_source": label_source,
        "agents": len(descriptors),
        "tool1": aggregate_rows(tool1_rows, group_key="method"),
        "tool2": aggregate_rows(tool2_rows, group_key="method"),
        "baselines": aggregate_rows(baseline_rows, group_key="method"),
        "ablations": aggregate_rows(ablation_rows, group_key="method"),
    }
    _write_metric_bundle(output, "tool1_metrics", tool1_rows)
    _write_metric_bundle(output, "tool2_metrics", tool2_rows)
    _write_metric_bundle(output, "baseline_metrics", baseline_rows)
    _write_metric_bundle(output, "ablation_metrics", ablation_rows)
    write_json(output / "evaluation_summary.json", aggregate)
    (output / "paper_tables.md").write_text(
        build_paper_tables(tool1_rows, tool2_rows, baseline_rows, ablation_rows, aggregate),
        encoding="utf-8",
    )
    print(
        "【实验】评估完成："
        f"tool1_metrics/tool2_metrics/baseline_metrics/ablation_metrics/evaluation_summary/paper_tables 已写入 {output}"
    )
    return aggregate


def compute_tool1_metrics(
    agent_ref: str,
    method: str,
    seeds: list[RiskSeed],
    snapshot: AgentSnapshot,
    truth: set[str],
    label_source: str,
    discovery_cost_s: float = 0.0,
) -> dict[str, Any]:
    """计算 Tool1 risk-domain 发现质量和证据完整性。"""
    detected = {seed.risk_domain for seed in seeds if seed.status != "candidate"}
    hits = detected & truth
    precision = len(hits) / max(1, len(detected))
    recall = len(hits) / max(1, len(truth))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    evidence_ids = {item.evidence_id for item in snapshot.evidence_index}
    evidence_complete = [
        bool(seed.evidence_ids) and set(seed.evidence_ids).issubset(evidence_ids)
        for seed in seeds
        if seed.status != "candidate"
    ]
    return {
        "agent_ref": agent_ref,
        "method": method,
        "label_source": label_source,
        "truth_domains": "|".join(sorted(truth)),
        "detected_domains": "|".join(sorted(detected)),
        "seed_count": len(seeds),
        "auto_or_review_seed_count": sum(1 for seed in seeds if seed.status != "candidate"),
        "seed_precision": round(precision, 4),
        "seed_recall": round(recall, 4),
        "seed_f1": round(f1, 4),
        "evidence_completeness": round(sum(evidence_complete) / max(1, len(evidence_complete)), 4),
        "average_confidence": round(mean([seed.confidence for seed in seeds]) if seeds else 0.0, 4),
        "invalid_test_reduction": round(1.0 - len(detected) / len(ALL_RISK_DOMAINS), 4),
        "discovery_cost_s": discovery_cost_s,
    }


def compute_tool2_metrics(
    agent_ref: str,
    method: str,
    cases: list[GeneratedCase],
    snapshot: AgentSnapshot,
    ignore_dry_run: bool = False,
) -> dict[str, Any]:
    """计算 Tool2 生成用例的 schema、dry-run、相关性、多样性和溯源指标。"""
    case_count = len(cases)
    schema_valid = [bool(case.validation_result.get("schema_valid")) for case in cases]
    dry_valid = [bool(case.validation_result.get("dry_run_valid", True)) for case in cases]
    fallback = [
        bool(case.validation_result.get("sandbox_fallback_available")) and not bool(case.validation_result.get("executor_available"))
        for case in cases
    ]
    provenance = [_has_provenance(case) for case in cases]
    relevance = [_target_relevant(case, snapshot) for case in cases]
    unique_fingerprints = {
        json.dumps(
            {
                "family": case.attack_family,
                "setup": case.setup,
                "trigger": case.trigger,
                "strategy": case.provenance.get("mutation_strategy"),
                "subtype": case.provenance.get("subtype"),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        for case in cases
    }
    return {
        "agent_ref": agent_ref,
        "method": method,
        "case_count": case_count,
        "schema_valid_rate": round(sum(schema_valid) / max(1, case_count), 4),
        "dry_run_valid_rate": 1.0 if ignore_dry_run else round(sum(dry_valid) / max(1, case_count), 4),
        "executor_fallback_rate": round(sum(fallback) / max(1, case_count), 4),
        "target_relevance": round(sum(relevance) / max(1, case_count), 4),
        "case_diversity": round(len(unique_fingerprints) / max(1, case_count), 4),
        "review_required_rate": round(sum(1 for case in cases if case.quality_score < 0.80) / max(1, case_count), 4),
        "average_quality_score": round(mean([case.quality_score for case in cases]) if cases else 0.0, 4),
        "case_provenance_coverage": round(sum(provenance) / max(1, case_count), 4),
        "metric_scope": "dry_run_proxy",
    }


def load_label_file(path: str | Path) -> dict[str, list[str]]:
    """读取 JSON/CSV/JSONL/XLSX 标签文件，输出 agent_ref -> risk domains。"""
    data = _read_records_or_mapping(path)
    if isinstance(data, dict):
        labels: dict[str, list[str]] = {}
        for agent, domains in data.items():
            if isinstance(domains, str):
                labels[str(agent)] = _split_domains(domains)
            else:
                labels[str(agent)] = sorted({str(item) for item in domains})
        return labels
    labels: dict[str, set[str]] = defaultdict(set)
    for row in data:
        agent = str(row.get("agent_ref") or row.get("agent") or row.get("target") or "").strip()
        if not agent:
            continue
        domains = row.get("risk_domain") or row.get("expected_domains") or row.get("domains") or ""
        for domain in _split_domains(str(domains)):
            labels[agent].add(domain)
    return {agent: sorted(domains) for agent, domains in labels.items()}


def import_paper_results(input_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """导入外部/人工/真实执行结果，只归一化和汇总，不改写原始数值。"""
    output = ensure_dir(out_dir)
    records = _read_records(input_path)
    normalized = [_normalize_result_record(record) for record in records]
    aggregate = aggregate_imported_results(normalized)
    write_json(output / "imported_results.json", normalized)
    write_json(output / "imported_summary.json", aggregate)
    write_csv(output / "imported_results.csv", normalized)
    (output / "paper_tables.md").write_text(build_imported_result_tables(normalized, aggregate), encoding="utf-8")
    return aggregate


def aggregate_rows(rows: list[dict[str, Any]], group_key: str = "method") -> list[dict[str, Any]]:
    """按 method/source 等字段聚合数值列平均值。"""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, "unknown"))].append(row)
    result: list[dict[str, Any]] = []
    for group, items in sorted(grouped.items()):
        record = {group_key: group, "rows": len(items)}
        for key in sorted({key for item in items for key in item}):
            values = [_as_float(item.get(key)) for item in items]
            numeric = [value for value in values if value is not None]
            if numeric and len(numeric) == len(items):
                record[f"avg_{key}"] = round(mean(numeric), 4)
        result.append(record)
    return result


def build_paper_tables(
    tool1_rows: list[dict[str, Any]],
    tool2_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> str:
    """把各类指标渲染成 Markdown 表，便于论文/报告直接引用。"""
    lines = [
        "# Tool1/Tool2 Paper Tables",
        "",
        "All metrics are generated from explicit labels or descriptor expected domains. Sandbox outputs are reported only as dry-run/proxy metrics, not real ASR.",
        "",
        "## Overview",
        "",
        f"- Agents: {aggregate['agents']}",
        f"- Label source: {aggregate['label_source']}",
        "",
    ]
    lines.extend(_markdown_table("Tool1 Aggregate", aggregate["tool1"]))
    lines.extend(_markdown_table("Tool2 Aggregate", aggregate["tool2"]))
    lines.extend(_markdown_table("Baseline Aggregate", aggregate["baselines"]))
    lines.extend(_markdown_table("Ablation Aggregate", aggregate["ablations"]))
    lines.extend(_markdown_table("Tool1 Per-Agent", tool1_rows, columns=["agent_ref", "method", "seed_precision", "seed_recall", "seed_f1", "evidence_completeness", "invalid_test_reduction"]))
    lines.extend(_markdown_table("Tool2 Per-Agent", tool2_rows, columns=["agent_ref", "method", "schema_valid_rate", "dry_run_valid_rate", "target_relevance", "case_diversity", "average_quality_score"]))
    lines.extend(_markdown_table("Baselines", baseline_rows, columns=["agent_ref", "method", "seed_precision", "seed_recall", "invalid_test_reduction", "schema_valid_rate", "dry_run_valid_rate", "target_relevance"]))
    lines.extend(_markdown_table("Ablations", ablation_rows, columns=["agent_ref", "method", "seed_precision", "seed_recall", "average_confidence", "schema_valid_rate", "dry_run_valid_rate", "target_relevance"]))
    return "\n".join(lines).rstrip() + "\n"


def build_imported_result_tables(records: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    lines = [
        "# Imported Paper Results",
        "",
        "Imported records are summarized without changing their values. The `source` column should identify whether values are manual labels, real executor results, or dry-run proxies.",
        "",
    ]
    lines.extend(_markdown_table("Aggregate By Method", aggregate.get("by_method", [])))
    lines.extend(_markdown_table("Aggregate By Source", aggregate.get("by_source", [])))
    lines.extend(_markdown_table("Records", records[:50]))
    return "\n".join(lines).rstrip() + "\n"


def aggregate_imported_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "by_method": aggregate_rows(records, "method"),
        "by_source": aggregate_rows(records, "source"),
    }


def _evaluate_baselines(
    descriptor: AgentAccessDescriptor,
    snapshot: AgentSnapshot,
    truth: set[str],
    label_source: str,
    count: int,
    profile: str,
    random_seed: int,
    include_direct_llm: bool,
) -> list[dict[str, Any]]:
    """生成 all_domains、random_domains、fixed_template 和可选 direct_llm 对照。"""
    rows: list[dict[str, Any]] = []
    rng = random.Random(f"{random_seed}:{descriptor.agent_ref}")
    scenarios = {
        "all_domains": list(ALL_RISK_DOMAINS),
        "random_domains": sorted(rng.sample(list(ALL_RISK_DOMAINS), k=max(1, min(len(ALL_RISK_DOMAINS), len(truth) or 1)))),
    }
    for method, domains in scenarios.items():
        seeds = _synthetic_seeds(snapshot, domains, method)
        cases = Tool2Generator(enable_llm_variants=False).generate(snapshot, seeds, count=count, profile=profile, use_siraj_prompts=False)
        rows.append(
            {
                **compute_tool1_metrics(descriptor.agent_ref, method, seeds, snapshot, truth, label_source),
                **_prefixed(compute_tool2_metrics(descriptor.agent_ref, method, cases, snapshot), ""),
            }
        )

    fixed_cases = Tool2Generator(enable_llm_variants=False).generate(
        snapshot,
        _synthetic_seeds(snapshot, sorted(truth or {ALL_RISK_DOMAINS[0]}), "fixed_template"),
        count=count,
        strategies=["template"],
        profile=profile,
        use_siraj_prompts=False,
    )
    rows.append(
        {
            **compute_tool1_metrics(descriptor.agent_ref, "fixed_template", _synthetic_seeds(snapshot, sorted(truth), "fixed_template"), snapshot, truth, label_source),
            **_prefixed(compute_tool2_metrics(descriptor.agent_ref, "fixed_template", fixed_cases, snapshot), ""),
        }
    )
    if include_direct_llm:
        rows.append(_evaluate_direct_llm_baseline(descriptor, snapshot, truth, label_source, count))
    return rows


def _evaluate_direct_llm_baseline(
    descriptor: AgentAccessDescriptor,
    snapshot: AgentSnapshot,
    truth: set[str],
    label_source: str,
    count: int,
) -> dict[str, Any]:
    """直接让 LLM 从 snapshot 生成 case 的 baseline；无 key 时显式 skipped。"""
    client = DeepSeekJSONClient()
    if not client.available:
        return {
            "agent_ref": descriptor.agent_ref,
            "method": "direct_llm",
            "label_source": label_source,
            "status": "skipped_no_llm_key",
            "metric_scope": "not_run",
        }

    system = load_prompt("evaluation_direct_llm_baseline_system")
    request = {
        "task": "Generate direct LLM baseline cases for comparison with Tool1/Tool2.",
        "agent_ref": descriptor.agent_ref,
        "allowed_risk_domains": list(ALL_RISK_DOMAINS),
        "target_case_count": count,
        "agent_snapshot": {
            "capabilities": snapshot.capabilities,
            "tool_schemas": truncate_text(snapshot.tool_schemas, 1600),
            "evidence_features": [item.feature for item in snapshot.evidence_index[:40]],
        },
        "expected_json_schema": {
            "cases": [
                {
                    "attack_family": "one allowed risk domain",
                    "delivery_mode": "direct_input or environment_poisoning",
                    "setup": {},
                    "trigger": {},
                    "expected_signal": {},
                    "cleanup": {},
                    "rationale": "short reason",
                }
            ]
        },
    }
    try:
        payload = client.complete_json(system, request)
    except (LLMUnavailable, KeyError, TypeError, ValueError) as exc:
        return {
            "agent_ref": descriptor.agent_ref,
            "method": "direct_llm",
            "label_source": label_source,
            "status": "llm_failed",
            "error": str(exc)[:200],
            "metric_scope": "not_run",
        }

    cases = _coerce_direct_llm_cases(payload, snapshot)
    domains = sorted({case.attack_family for case in cases})
    seeds = _synthetic_seeds(snapshot, domains, "direct_llm")
    return {
        **compute_tool1_metrics(descriptor.agent_ref, "direct_llm", seeds, snapshot, truth, label_source),
        **compute_tool2_metrics(descriptor.agent_ref, "direct_llm", cases, snapshot),
        "status": "ok",
        "direct_llm_model": client.config.model,
    }


def _coerce_direct_llm_cases(payload: dict[str, Any], snapshot: AgentSnapshot) -> list[GeneratedCase]:
    """把 direct LLM 输出强制套回 GeneratedCase，并复用 Tool2 校验器。"""
    raw_cases = payload.get("cases", [])
    if not isinstance(raw_cases, list):
        return []
    validator = Tool2Generator(enable_llm_variants=False)
    evidence_id = snapshot.evidence_index[0].evidence_id if snapshot.evidence_index else "ev_direct_llm"
    cases: list[GeneratedCase] = []
    for idx, item in enumerate(raw_cases, start=1):
        if not isinstance(item, dict):
            continue
        attack_family = str(item.get("attack_family") or item.get("risk_domain") or "prompt_context_injection")
        if attack_family not in ALL_RISK_DOMAINS:
            attack_family = "prompt_context_injection"
        seed = RiskSeed(
            seed_id=f"seed_{snapshot.analysis_id}_direct_llm_{idx:02d}",
            analysis_id=snapshot.analysis_id,
            risk_domain=attack_family,
            entry_point="direct_llm_generated_entry",
            evidence_ids=[evidence_id],
            preconditions=["direct LLM baseline generated from snapshot summary"],
            attack_goal=str(item.get("rationale") or f"direct LLM baseline for {attack_family}"),
            recommended_executor=EXECUTOR_BY_DOMAIN.get(attack_family, "sandbox"),
            confidence=0.5,
            status="auto_generate",
            score_detail={"baseline": True, "rule_id": "direct_llm"},
        )
        candidate = {
            "template_id": "direct_llm_baseline",
            "delivery_mode": str(item.get("delivery_mode") or "direct_input"),
            "setup": item.get("setup") if isinstance(item.get("setup"), dict) else {},
            "trigger": item.get("trigger") if isinstance(item.get("trigger"), dict) else {},
            "expected_signal": item.get("expected_signal") if isinstance(item.get("expected_signal"), dict) else {},
            "cleanup": item.get("cleanup") if isinstance(item.get("cleanup"), dict) else {},
        }
        validation = validator._validate(candidate, snapshot, seed)
        cases.append(
            GeneratedCase(
                case_id=f"case_{seed.seed_id}_v01_direct",
                seed_id=seed.seed_id,
                attack_family=attack_family,
                delivery_mode=candidate["delivery_mode"],
                setup=dict(candidate["setup"]),
                trigger=dict(candidate["trigger"]),
                expected_signal=dict(candidate["expected_signal"]),
                cleanup=dict(candidate["cleanup"]),
                executor=seed.recommended_executor,
                quality_score=0.75 if validation.get("dry_run_valid") else 0.45,
                provenance={
                    "template_id": "direct_llm_baseline",
                    "mutation_strategy": "direct_llm",
                    "generator_version": "direct-llm-baseline-0.1",
                    "seed_confidence": seed.confidence,
                    "safe_marker": SAFE_MARKER,
                    "rationale": str(item.get("rationale", ""))[:500],
                },
                validation_result=validation,
            )
        )
    return cases


def _evaluate_ablations(
    descriptor: AgentAccessDescriptor,
    truth: set[str],
    label_source: str,
    count: int,
    profile: str,
    enable_llm_evidence: bool | None,
    enable_llm_runtime_events: bool | None,
    enable_llm_review: bool | None,
    enable_llm_variants: bool | None,
    use_siraj_prompts: bool,
    out_root: Path,
) -> list[dict[str, Any]]:
    """逐项关闭静态解析、动态 probe、语义证据、SIRAJ 等模块做消融。"""
    rows: list[dict[str, Any]] = []
    scenarios = [
        ("ours_full", descriptor, True, enable_llm_evidence, enable_llm_runtime_events, enable_llm_review, enable_llm_variants, use_siraj_prompts, False, False, True),
        ("w/o_static_parsing", _without_optional_artifacts(descriptor), True, enable_llm_evidence, enable_llm_runtime_events, enable_llm_review, enable_llm_variants, use_siraj_prompts, False, False, True),
        ("w/o_dynamic_probe", descriptor, False, enable_llm_evidence, enable_llm_runtime_events, enable_llm_review, enable_llm_variants, use_siraj_prompts, False, False, True),
        ("w/o_semantic_evidence", descriptor, True, False, enable_llm_runtime_events, enable_llm_review, enable_llm_variants, use_siraj_prompts, False, False, True),
        ("w/o_runtime_event_induction", descriptor, True, enable_llm_evidence, False, enable_llm_review, enable_llm_variants, use_siraj_prompts, False, False, True),
        ("w/o_llm_review", descriptor, True, enable_llm_evidence, enable_llm_runtime_events, False, enable_llm_variants, use_siraj_prompts, False, False, True),
        ("w/o_siraj_prompts", descriptor, True, enable_llm_evidence, enable_llm_runtime_events, enable_llm_review, enable_llm_variants, False, False, False, True),
        ("w/o_context_binding", descriptor, True, enable_llm_evidence, enable_llm_runtime_events, enable_llm_review, enable_llm_variants, use_siraj_prompts, True, False, True),
        ("w/o_dry_run", descriptor, True, enable_llm_evidence, enable_llm_runtime_events, enable_llm_review, enable_llm_variants, use_siraj_prompts, False, True, True),
        ("w/o_feedback", descriptor, True, enable_llm_evidence, enable_llm_runtime_events, enable_llm_review, enable_llm_variants, use_siraj_prompts, False, False, False),
    ]
    for method, scenario_descriptor, dynamic, llm_evidence, llm_runtime_events, llm_review, llm_variants, scenario_siraj_prompts, generic_context, ignore_dry_run, feedback in scenarios:
        agent_dir = ensure_dir(out_root / method.replace("/", "_") / _safe_name(descriptor.agent_ref))
        analyzer = Tool1Analyzer(
            enable_dynamic_probe=dynamic,
            enable_llm_evidence=llm_evidence,
            enable_llm_runtime_events=llm_runtime_events,
            enable_llm_review=llm_review,
        )
        start = time.perf_counter()
        session, snapshot, seeds = analyzer.analyze(scenario_descriptor, agent_dir)
        cost = round(time.perf_counter() - start, 4)
        generation_snapshot = _generic_snapshot(snapshot) if generic_context else snapshot
        cases = Tool2Generator(enable_llm_variants=llm_variants).generate(
            generation_snapshot,
            seeds,
            count=count,
            out_dir=agent_dir,
            profile=profile,
            use_siraj_prompts=scenario_siraj_prompts,
        )
        if feedback:
            results = DEFAULT_EXECUTOR_REGISTRY.run(session.analysis_id, cases)
            write_json(agent_dir / "run_result.json", results)
            apply_feedback_to_analysis(agent_dir)
            seeds = [RiskSeed.from_dict(item) for item in load_json(agent_dir / "risk_seeds.json")]
        rows.append(
            {
                **compute_tool1_metrics(descriptor.agent_ref, method, seeds, snapshot, truth, label_source, discovery_cost_s=cost),
                **_prefixed(compute_tool2_metrics(descriptor.agent_ref, method, cases, generation_snapshot, ignore_dry_run=ignore_dry_run), ""),
            }
        )
    return rows


def _synthetic_seeds(snapshot: AgentSnapshot, domains: list[str], method: str) -> list[RiskSeed]:
    """为 baseline 构造有固定证据引用的合成 seed。"""
    evidence_id = snapshot.evidence_index[0].evidence_id if snapshot.evidence_index else f"ev_{method}"
    return [
        RiskSeed(
            seed_id=f"seed_{snapshot.analysis_id}_{method}_{idx:02d}",
            analysis_id=snapshot.analysis_id,
            risk_domain=domain,
            entry_point=f"{method}_entry",
            evidence_ids=[evidence_id],
            preconditions=[f"{method} baseline selection"],
            attack_goal=f"{method} baseline case for {domain}",
            recommended_executor=EXECUTOR_BY_DOMAIN.get(domain, "sandbox"),
            confidence=0.75,
            status="auto_generate",
            score_detail={"rule_id": method, "baseline": True},
        )
        for idx, domain in enumerate(domains, start=1)
    ]


def _target_relevant(case: GeneratedCase, snapshot: AgentSnapshot) -> bool:
    """检查 case 的风险域是否与 snapshot 中观测到的能力匹配。"""
    capabilities = snapshot.capabilities
    if case.attack_family == "rag_poisoning":
        return bool(capabilities.get("rag"))
    if case.attack_family == "memory_poisoning":
        return bool(capabilities.get("memory"))
    if case.attack_family == "tool_output_injection":
        return bool(capabilities.get("tool")) and _case_tool_known(case, snapshot)
    if case.attack_family == "mcp_description_poisoning":
        return bool(capabilities.get("mcp")) and _case_tool_known(case, snapshot)
    if case.attack_family == "planning_poisoning":
        return bool(capabilities.get("planning"))
    if case.attack_family == "multi_agent_communication_poisoning":
        return bool(capabilities.get("multi_agent"))
    if case.attack_family == "search_narrative_poisoning":
        return bool(capabilities.get("search"))
    return case.attack_family == "prompt_context_injection"


def _case_tool_known(case: GeneratedCase, snapshot: AgentSnapshot) -> bool:
    known = {str(tool.get("name")) for tool in snapshot.tool_schemas if tool.get("name")}
    if not known:
        return bool(snapshot.capabilities.get("tool") or snapshot.capabilities.get("mcp"))
    tool_name = str(case.setup.get("tool_name", ""))
    return tool_name in known


def _has_provenance(case: GeneratedCase) -> bool:
    required = {"template_id", "mutation_strategy", "generator_version", "seed_confidence", "safe_marker"}
    return required.issubset(set(case.provenance))


def _without_optional_artifacts(descriptor: AgentAccessDescriptor) -> AgentAccessDescriptor:
    data = descriptor.__dict__.copy()
    data["optional_artifacts"] = []
    return AgentAccessDescriptor.from_dict(data)


def _generic_snapshot(snapshot: AgentSnapshot) -> AgentSnapshot:
    """消融 context binding 时移除 api_spec/tool_schemas，保留其他观测。"""
    return AgentSnapshot(
        analysis_id=snapshot.analysis_id,
        agent_ref=snapshot.agent_ref,
        connector_type=snapshot.connector_type,
        capabilities=dict(snapshot.capabilities),
        api_spec={},
        tool_schemas=[],
        runtime_observations=list(snapshot.runtime_observations),
        evidence_index=list(snapshot.evidence_index),
        created_at=snapshot.created_at,
    )


def _write_metric_bundle(out_dir: Path, stem: str, rows: list[dict[str, Any]]) -> None:
    write_json(out_dir / f"{stem}.json", rows)
    write_csv(out_dir / f"{stem}.csv", rows)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """用所有行的 key 并集写 CSV，保持不同方法字段可并排比较。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return target


def _read_records_or_mapping(path: str | Path) -> Any:
    """标签文件可以是映射，也可以是记录列表；上层按类型继续规范化。"""
    resolved = Path(path)
    if resolved.suffix.lower() == ".json":
        return load_json(resolved)
    return _read_records(resolved)


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSON/JSONL/CSV/XLSX 为记录列表。"""
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        data = load_json(resolved)
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        if isinstance(data, dict):
            return [dict(data)]
        return [dict(item) for item in data]
    if suffix == ".jsonl":
        with resolved.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".xlsx":
        try:
            import openpyxl  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("openpyxl is required to import .xlsx files; export to CSV or install openpyxl.") from exc
        workbook = openpyxl.load_workbook(resolved, read_only=True, data_only=True)
        sheet = workbook.active
        rows = list(sheet.iter_rows(values_only=True))
        if not rows:
            return []
        headers = [str(value) for value in rows[0]]
        return [dict(zip(headers, row)) for row in rows[1:]]
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _normalize_result_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = {str(key): value for key, value in record.items()}
    normalized.setdefault("source", "unspecified")
    normalized.setdefault("method", "unknown")
    return normalized


def _split_domains(value: str) -> list[str]:
    return sorted({item.strip() for item in value.replace(";", ",").replace("|", ",").split(",") if item.strip()})


def _markdown_table(title: str, rows: list[dict[str, Any]], columns: list[str] | None = None) -> list[str]:
    """渲染简单 Markdown 表；空表显式写 No rows。"""
    lines = [f"## {title}", ""]
    if not rows:
        return [*lines, "_No rows._", ""]
    columns = columns or sorted({key for row in rows for key in row})
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(col, "")) for col in columns) + " |")
    lines.append("")
    return lines


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value).replace("|", "\\|")


def _prefixed(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    if not prefix:
        return dict(row)
    return {f"{prefix}{key}": value for key, value in row.items()}


def _as_float(value: Any) -> float | None:
    """聚合时只对可解析数值列求平均。"""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return safe[:80] or "agent"
