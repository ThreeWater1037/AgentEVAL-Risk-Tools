"""AgentEVAL 命令行入口。

CLI 将 Tool1、Tool2、执行器、反馈、评估和报告串成可复现的本地流程。
默认 case 生成走 SIRAJ prompt 路径，`--legacy-prompts` 只作为显式回退/消融使用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .adapters import load_target_descriptors
from .evaluation import evaluate_tool12, import_paper_results, load_label_file
from .experiment import DEFAULT_EXECUTOR_REGISTRY, summarize_run_root
from .feedback import apply_feedback_to_analysis
from .io import ensure_dir, load_json, write_json
from .report import write_run_markdown
from .schemas import AgentAccessDescriptor, AgentSnapshot, GeneratedCase, RiskSeed
from .tool1 import Tool1Analyzer
from .tool2 import Tool2Generator


def main(argv: list[str] | None = None) -> int:
    """注册所有子命令并分发到对应处理函数。"""
    parser = argparse.ArgumentParser(prog="agenteval")
    sub = parser.add_subparsers(dest="command", required=True)

    # 单步 Tool1：从 descriptor 发现 evidence/snapshot/risk_seeds。
    analyze = sub.add_parser("analyze-agent", help="Run Tool1 against a descriptor.")
    analyze.add_argument("--descriptor", required=True)
    analyze.add_argument("--agent")
    analyze.add_argument("--out", required=True)
    analyze.add_argument("--no-dynamic-probe", action="store_true")
    _add_llm_evidence_flags(analyze)
    _add_llm_runtime_event_flags(analyze)
    _add_llm_review_flags(analyze)

    # 单步 Tool2：读取 Tool1 输出目录并生成 generated_cases.json。
    generate = sub.add_parser("generate-cases", help="Run Tool2 for one analysis directory.")
    generate.add_argument("--analysis-dir", required=True)
    generate.add_argument("--count", type=int, default=3)
    generate.add_argument("--profile", choices=["compact", "expanded"], default="compact")
    _add_case_prompt_flags(generate)
    _add_llm_variant_flags(generate)

    # 单步执行：默认注册表会回退到确定性 sandbox。
    run_cases = sub.add_parser("run-cases", help="Run deterministic sandbox execution for generated cases.")
    run_cases.add_argument("--analysis-dir", required=True)

    # 失败/低质量 case 的多轮 SIRAJ-style refinement。
    refine = sub.add_parser("refine-cases", help="Append SIRAJ-style refinements for failed or low-quality cases.")
    refine.add_argument("--analysis-dir", required=True)
    refine.add_argument("--rounds", type=int, default=1)
    refine.add_argument("--quality-threshold", type=float, default=0.80)
    _add_llm_variant_flags(refine)

    feedback = sub.add_parser("apply-feedback", help="Update risk seed confidence from run_result.json.")
    feedback.add_argument("--analysis-dir", required=True)

    # 全链路论文式代理评估，写 CSV/JSON/Markdown 表格。
    evaluate = sub.add_parser("evaluate-tool12", help="Run transparent Tool1/Tool2 paper-style proxy evaluation.")
    evaluate.add_argument("--descriptors", required=True)
    evaluate.add_argument("--labels")
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--count", type=int, default=3)
    evaluate.add_argument("--profile", choices=["compact", "expanded"], default="compact")
    evaluate.add_argument("--include-direct-llm", action="store_true")
    evaluate.add_argument("--random-seed", type=int, default=13)
    _add_case_prompt_flags(evaluate)
    _add_llm_evidence_flags(evaluate)
    _add_llm_runtime_event_flags(evaluate)
    _add_llm_review_flags(evaluate)
    _add_llm_variant_flags(evaluate)

    import_results = sub.add_parser("import-paper-results", help="Import explicit manual/real-executor result tables for paper formatting.")
    import_results.add_argument("--input", required=True)
    import_results.add_argument("--out", required=True)

    demo = sub.add_parser("run-demo", help="Run Tool1/Tool2 over current-framework mock descriptors.")
    demo.add_argument("--descriptors", default="examples/current_framework_agents.json")
    demo.add_argument("--out", required=True)
    demo.add_argument("--count", type=int, default=3)
    demo.add_argument("--profile", choices=["compact", "expanded"], default="compact")
    _add_case_prompt_flags(demo)
    _add_llm_evidence_flags(demo)
    _add_llm_runtime_event_flags(demo)
    _add_llm_review_flags(demo)
    _add_llm_variant_flags(demo)

    manifest = sub.add_parser("run-manifest", help="Run Tool1/Tool2 over a target manifest or registry.")
    manifest.add_argument("--manifest", required=True)
    manifest.add_argument("--out", required=True)
    manifest.add_argument("--count", type=int, default=3)
    manifest.add_argument("--profile", choices=["compact", "expanded"], default="compact")
    _add_case_prompt_flags(manifest)
    _add_llm_evidence_flags(manifest)
    _add_llm_runtime_event_flags(manifest)
    _add_llm_review_flags(manifest)
    _add_llm_variant_flags(manifest)

    summarize = sub.add_parser("summarize", help="Summarize a run root.")
    summarize.add_argument("--run-root", required=True)

    report = sub.add_parser("write-report", help="Write a Markdown report for a run root.")
    report.add_argument("--run-root", required=True)
    report.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.command == "analyze-agent":
        return _cmd_analyze(args)
    if args.command == "generate-cases":
        return _cmd_generate(args)
    if args.command == "run-cases":
        return _cmd_run_cases(args)
    if args.command == "refine-cases":
        return _cmd_refine_cases(args)
    if args.command == "apply-feedback":
        return _cmd_apply_feedback(args)
    if args.command == "evaluate-tool12":
        return _cmd_evaluate_tool12(args)
    if args.command == "import-paper-results":
        return _cmd_import_paper_results(args)
    if args.command == "run-demo":
        return _cmd_run_demo(args)
    if args.command == "run-manifest":
        return _cmd_run_manifest(args)
    if args.command == "summarize":
        summary = summarize_run_root(args.run_root)
        print(_brief_summary(summary))
        return 0
    if args.command == "write-report":
        path = write_run_markdown(args.run_root, args.out)
        print(f"wrote report -> {path}")
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


def _cmd_analyze(args: argparse.Namespace) -> int:
    """执行 Tool1，并按单目标/多目标选择输出目录。"""
    descriptors = _load_descriptors(args.descriptor)
    selected = _select_descriptors(descriptors, args.agent)
    print(f"【CLI】开始Tool1风险发现：输入={args.descriptor}，目标数量={len(selected)}，输出目录={args.out}")
    analyzer = Tool1Analyzer(
        enable_dynamic_probe=not args.no_dynamic_probe,
        enable_llm_evidence=args.llm_evidence,
        enable_llm_runtime_events=args.llm_runtime_events,
        enable_llm_review=args.llm_review,
    )
    out = ensure_dir(args.out)
    for descriptor in selected:
        target = out if len(selected) == 1 else out / _safe_name(descriptor.agent_ref)
        print(f"【Tool1】开始分析Agent：{descriptor.agent_ref}")
        analyzer.analyze(descriptor, target)
        print(f"【Tool1】完成分析Agent：{descriptor.agent_ref}，输出目录={target}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    """读取已有分析目录，执行 Tool2 case 生成。"""
    analysis_dir = Path(args.analysis_dir)
    print(f"【CLI】开始Tool2用例生成：analysis_dir={analysis_dir}，count={args.count}，profile={args.profile}")
    snapshot = AgentSnapshot.from_dict(load_json(analysis_dir / "agent_snapshot.json"))
    seeds = [RiskSeed.from_dict(item) for item in load_json(analysis_dir / "risk_seeds.json")]
    cases = Tool2Generator(enable_llm_variants=args.llm_variants).generate(
        snapshot,
        seeds,
        count=args.count,
        out_dir=analysis_dir,
        profile=args.profile,
        use_siraj_prompts=args.siraj_prompts,
    )
    print(f"【Tool2】完成用例生成：cases={len(cases)}，输出={analysis_dir / 'generated_cases.json'}")
    return 0


def _cmd_run_cases(args: argparse.Namespace) -> int:
    """执行 generated_cases.json，并写出 run_result.json。"""
    analysis_dir = Path(args.analysis_dir)
    print(f"【CLI】开始执行测试用例：analysis_dir={analysis_dir}")
    snapshot = AgentSnapshot.from_dict(load_json(analysis_dir / "agent_snapshot.json"))
    cases = [GeneratedCase.from_dict(item) for item in load_json(analysis_dir / "generated_cases.json")]
    results = DEFAULT_EXECUTOR_REGISTRY.run(snapshot.analysis_id, cases)
    write_json(analysis_dir / "run_result.json", results)
    print(f"【执行器】完成测试用例执行：results={len(results)}，输出={analysis_dir / 'run_result.json'}")
    return 0


def _cmd_refine_cases(args: argparse.Namespace) -> int:
    """根据 run_result.json 对低质量或失败 case 追加 refinement。"""
    analysis_dir = Path(args.analysis_dir)
    print(f"【CLI】开始多轮refinement：analysis_dir={analysis_dir}，rounds={args.rounds}，quality_threshold={args.quality_threshold}")
    snapshot = AgentSnapshot.from_dict(load_json(analysis_dir / "agent_snapshot.json"))
    seeds = [RiskSeed.from_dict(item) for item in load_json(analysis_dir / "risk_seeds.json")]
    cases = [GeneratedCase.from_dict(item) for item in load_json(analysis_dir / "generated_cases.json")]
    results = load_json(analysis_dir / "run_result.json")
    refined = Tool2Generator(enable_llm_variants=args.llm_variants).refine_cases(
        snapshot,
        seeds,
        cases,
        results,
        rounds=args.rounds,
        out_dir=analysis_dir,
        quality_threshold=args.quality_threshold,
    )
    print(f"【Tool2】完成refinement：总cases={len(refined)}，新增={len(refined) - len(cases)}，输出={analysis_dir / 'generated_cases.json'}")
    return 0


def _cmd_apply_feedback(args: argparse.Namespace) -> int:
    """把执行结果反馈回 risk_seeds.json 的 confidence/status。"""
    print(f"【CLI】开始反馈更新：analysis_dir={args.analysis_dir}")
    summary = apply_feedback_to_analysis(args.analysis_dir)
    print(f"【反馈】完成反馈更新：updated_seeds={summary['updated_seeds']}，输出={Path(args.analysis_dir) / 'risk_seeds.json'}")
    return 0


def _cmd_evaluate_tool12(args: argparse.Namespace) -> int:
    """运行主方法、baseline 和消融，生成评估汇总。"""
    print(f"【CLI】开始论文式评估：descriptors={args.descriptors}，输出目录={args.out}")
    descriptors = _load_descriptors(args.descriptors)
    labels = load_label_file(args.labels) if args.labels else None
    summary = evaluate_tool12(
        descriptors,
        args.out,
        labels=labels,
        count=args.count,
        profile=args.profile,
        enable_llm_evidence=args.llm_evidence,
        enable_llm_runtime_events=args.llm_runtime_events,
        enable_llm_review=args.llm_review,
        enable_llm_variants=args.llm_variants,
        use_siraj_prompts=args.siraj_prompts,
        include_direct_llm=args.include_direct_llm,
        random_seed=args.random_seed,
    )
    print(
        f"【实验】评估完成：agents={summary['agents']}，label_source={summary['label_source']}，"
        f"表格={Path(args.out) / 'paper_tables.md'}"
    )
    return 0


def _cmd_import_paper_results(args: argparse.Namespace) -> int:
    """导入外部实验结果并生成同格式表格。"""
    print(f"【CLI】开始导入论文结果：input={args.input}，输出目录={args.out}")
    summary = import_paper_results(args.input, args.out)
    print(f"【导入】完成论文结果导入：records={summary['records']}，表格={Path(args.out) / 'paper_tables.md'}")
    return 0


def _cmd_run_demo(args: argparse.Namespace) -> int:
    """运行示例 descriptor 的全链路 demo。"""
    descriptors = _load_descriptors(args.descriptors)
    root = ensure_dir(args.out)
    print(f"【CLI】开始demo全链路：agents={len(descriptors)}，输出目录={root}，count={args.count}，profile={args.profile}")
    analyzer = Tool1Analyzer(
        enable_dynamic_probe=True,
        enable_llm_evidence=args.llm_evidence,
        enable_llm_runtime_events=args.llm_runtime_events,
        enable_llm_review=args.llm_review,
    )
    generator = Tool2Generator(enable_llm_variants=args.llm_variants)
    per_agent_expected = []

    for descriptor in descriptors:
        agent_dir = ensure_dir(root / _safe_name(descriptor.agent_ref))
        print(f"【Demo】开始处理Agent：{descriptor.agent_ref}")
        session, snapshot, seeds = analyzer.analyze(descriptor, agent_dir)
        print(f"【Tool1】完成：agent={descriptor.agent_ref}，seeds={len(seeds)}，evidence={len(snapshot.evidence_index)}")
        cases = generator.generate(snapshot, seeds, count=args.count, out_dir=agent_dir, profile=args.profile, use_siraj_prompts=args.siraj_prompts)
        print(f"【Tool2】完成：agent={descriptor.agent_ref}，cases={len(cases)}")
        results = DEFAULT_EXECUTOR_REGISTRY.run(session.analysis_id, cases)
        write_json(agent_dir / "run_result.json", results)
        print(f"【执行器】完成：agent={descriptor.agent_ref}，results={len(results)}")
        detected_domains = {seed.risk_domain for seed in seeds}
        expected_domains = set(descriptor.expected_domains)
        hits = detected_domains & expected_domains
        per_agent_expected.append(
            {
                "agent_ref": descriptor.agent_ref,
                "expected_domains": sorted(expected_domains),
                "detected_domains": sorted(detected_domains),
                "hits": sorted(hits),
                "precision_proxy": round(len(hits) / max(1, len(detected_domains)), 3),
                "recall_proxy": round(len(hits) / max(1, len(expected_domains)), 3),
            }
        )
        print(f"【Demo】Agent处理完成：{descriptor.agent_ref}，seeds={len(seeds)}，cases={len(cases)}，results={len(results)}")

    summary = summarize_run_root(root)
    _augment_expected_summary(summary, per_agent_expected)
    write_json(root / "summary.json", summary)
    print(_brief_summary(summary))
    return 0


def _cmd_run_manifest(args: argparse.Namespace) -> int:
    """从目标清单/注册表读取 descriptor 并跑全链路。"""
    descriptors = load_target_descriptors(args.manifest)
    root = ensure_dir(args.out)
    print(f"【CLI】开始manifest全链路：targets={len(descriptors)}，manifest={args.manifest}，输出目录={root}")
    analyzer = Tool1Analyzer(
        enable_dynamic_probe=True,
        enable_llm_evidence=args.llm_evidence,
        enable_llm_runtime_events=args.llm_runtime_events,
        enable_llm_review=args.llm_review,
    )
    generator = Tool2Generator(enable_llm_variants=args.llm_variants)
    per_agent_expected = []
    for descriptor in descriptors:
        agent_dir = ensure_dir(root / _safe_name(descriptor.agent_ref))
        print(f"【Manifest】开始处理目标：{descriptor.agent_ref}")
        session, snapshot, seeds = analyzer.analyze(descriptor, agent_dir)
        print(f"【Tool1】完成：target={descriptor.agent_ref}，seeds={len(seeds)}，evidence={len(snapshot.evidence_index)}")
        cases = generator.generate(snapshot, seeds, count=args.count, out_dir=agent_dir, profile=args.profile, use_siraj_prompts=args.siraj_prompts)
        print(f"【Tool2】完成：target={descriptor.agent_ref}，cases={len(cases)}")
        results = DEFAULT_EXECUTOR_REGISTRY.run(session.analysis_id, cases)
        write_json(agent_dir / "run_result.json", results)
        print(f"【执行器】完成：target={descriptor.agent_ref}，results={len(results)}")
        detected_domains = {seed.risk_domain for seed in seeds}
        expected_domains = set(descriptor.expected_domains)
        hits = detected_domains & expected_domains
        per_agent_expected.append(
            {
                "agent_ref": descriptor.agent_ref,
                "expected_domains": sorted(expected_domains),
                "detected_domains": sorted(detected_domains),
                "hits": sorted(hits),
                "precision_proxy": round(len(hits) / max(1, len(detected_domains)), 3),
                "recall_proxy": round(len(hits) / max(1, len(expected_domains)), 3),
            }
        )
        print(f"【Manifest】目标处理完成：{descriptor.agent_ref}，seeds={len(seeds)}，cases={len(cases)}，results={len(results)}")
    summary = summarize_run_root(root)
    _augment_expected_summary(summary, per_agent_expected)
    write_json(root / "summary.json", summary)
    print(_brief_summary(summary))
    return 0


def _load_descriptors(path: str | Path) -> list[AgentAccessDescriptor]:
    """兼容单 descriptor、agents 列表和裸列表三种 JSON 形状。"""
    data = load_json(path)
    if isinstance(data, dict) and "agents" in data:
        items = data["agents"]
    elif isinstance(data, dict) and "agent_ref" in data:
        items = [data]
    else:
        items = data
    return [AgentAccessDescriptor.from_dict(item) for item in items]


def _add_llm_review_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--llm-review",
        "--enable-llm-review",
        dest="llm_review",
        action="store_true",
        default=None,
        help="Enable DeepSeek JSON review for low-confidence or natural-language Tool1 seeds.",
    )
    group.add_argument(
        "--no-llm-review",
        "--disable-llm-review",
        dest="llm_review",
        action="store_false",
        help="Disable Tool1 LLM review even when DEEPSEEK_API_KEY is set.",
    )


def _add_case_prompt_flags(parser: argparse.ArgumentParser) -> None:
    """SIRAJ 为默认生成路径；legacy 仅用于回退和消融。"""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--siraj-prompts",
        dest="siraj_prompts",
        action="store_true",
        default=True,
        help="Use SIRAJ-style Tool2 case prompts. This is the default path.",
    )
    group.add_argument(
        "--legacy-prompts",
        dest="siraj_prompts",
        action="store_false",
        help="Use the legacy Tool2 template path without SIRAJ case prompts.",
    )


def _add_llm_evidence_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--llm-evidence",
        "--enable-llm-evidence",
        dest="llm_evidence",
        action="store_true",
        default=None,
        help="Enable DeepSeek semantic evidence extraction for Tool1 text artifacts.",
    )
    group.add_argument(
        "--no-llm-evidence",
        "--disable-llm-evidence",
        dest="llm_evidence",
        action="store_false",
        help="Disable Tool1 semantic evidence extraction even when DEEPSEEK_API_KEY is set.",
    )


def _add_llm_runtime_event_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--llm-runtime-events",
        "--enable-llm-runtime-events",
        dest="llm_runtime_events",
        action="store_true",
        default=None,
        help="Enable DeepSeek runtime event induction from Tool1 probe responses.",
    )
    group.add_argument(
        "--no-llm-runtime-events",
        "--disable-llm-runtime-events",
        dest="llm_runtime_events",
        action="store_false",
        help="Disable Tool1 runtime event induction even when DEEPSEEK_API_KEY is set.",
    )


def _add_llm_variant_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--llm-variants",
        "--enable-llm-variants",
        dest="llm_variants",
        action="store_true",
        default=None,
        help="Enable DeepSeek JSON rewriting for Tool2 setup/trigger natural-language fields.",
    )
    group.add_argument(
        "--no-llm-variants",
        "--disable-llm-variants",
        dest="llm_variants",
        action="store_false",
        help="Disable Tool2 LLM rewriting even when DEEPSEEK_API_KEY is set.",
    )


def _select_descriptors(descriptors: list[AgentAccessDescriptor], agent_ref: str | None) -> list[AgentAccessDescriptor]:
    if not agent_ref:
        return descriptors
    selected = [item for item in descriptors if item.agent_ref == agent_ref]
    if not selected:
        available = ", ".join(item.agent_ref for item in descriptors)
        raise SystemExit(f"agent not found: {agent_ref}. Available: {available}")
    return selected


def _augment_expected_summary(summary: dict, per_agent_expected: list[dict]) -> None:
    """给 demo/manifest 汇总追加基于 expected_domains 的代理 precision/recall。"""
    hits = sum(len(item["hits"]) for item in per_agent_expected)
    detected = sum(len(item["detected_domains"]) for item in per_agent_expected)
    expected = sum(len(item["expected_domains"]) for item in per_agent_expected)
    full_domain_count = 8 * max(1, len(per_agent_expected))
    summary["tool1_precision_proxy"] = round(hits / max(1, detected), 3)
    summary["tool1_recall_proxy"] = round(hits / max(1, expected), 3)
    summary["invalid_test_reduction_vs_all_domains"] = round(1.0 - detected / full_domain_count, 3)
    summary["expected_domain_eval"] = per_agent_expected


def _brief_summary(summary: dict) -> str:
    return (
        f"agents={summary.get('agents', 0)} seeds={summary.get('seeds', 0)} "
        f"cases={summary.get('cases', 0)} schema_valid_rate={summary.get('schema_valid_rate', 0)} "
        f"executable_rate={summary.get('executable_rate', 0)} "
        f"tool1_precision_proxy={summary.get('tool1_precision_proxy', 'n/a')} "
        f"tool1_recall_proxy={summary.get('tool1_recall_proxy', 'n/a')}"
    )


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return safe[:80] or "agent"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
