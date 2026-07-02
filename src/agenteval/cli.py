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
    parser = argparse.ArgumentParser(prog="agenteval")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze-agent", help="Run Tool1 against a descriptor.")
    analyze.add_argument("--descriptor", required=True)
    analyze.add_argument("--agent")
    analyze.add_argument("--out", required=True)
    analyze.add_argument("--no-dynamic-probe", action="store_true")
    _add_llm_review_flags(analyze)

    generate = sub.add_parser("generate-cases", help="Run Tool2 for one analysis directory.")
    generate.add_argument("--analysis-dir", required=True)
    generate.add_argument("--count", type=int, default=3)
    generate.add_argument("--profile", choices=["compact", "expanded"], default="compact")
    _add_llm_variant_flags(generate)

    run_cases = sub.add_parser("run-cases", help="Run deterministic sandbox execution for generated cases.")
    run_cases.add_argument("--analysis-dir", required=True)

    feedback = sub.add_parser("apply-feedback", help="Update risk seed confidence from run_result.json.")
    feedback.add_argument("--analysis-dir", required=True)

    evaluate = sub.add_parser("evaluate-tool12", help="Run transparent Tool1/Tool2 paper-style proxy evaluation.")
    evaluate.add_argument("--descriptors", required=True)
    evaluate.add_argument("--labels")
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--count", type=int, default=3)
    evaluate.add_argument("--profile", choices=["compact", "expanded"], default="compact")
    evaluate.add_argument("--include-direct-llm", action="store_true")
    evaluate.add_argument("--random-seed", type=int, default=13)
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
    _add_llm_review_flags(demo)
    _add_llm_variant_flags(demo)

    manifest = sub.add_parser("run-manifest", help="Run Tool1/Tool2 over a target manifest or registry.")
    manifest.add_argument("--manifest", required=True)
    manifest.add_argument("--out", required=True)
    manifest.add_argument("--count", type=int, default=3)
    manifest.add_argument("--profile", choices=["compact", "expanded"], default="compact")
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
    descriptors = _load_descriptors(args.descriptor)
    selected = _select_descriptors(descriptors, args.agent)
    analyzer = Tool1Analyzer(enable_dynamic_probe=not args.no_dynamic_probe, enable_llm_review=args.llm_review)
    out = ensure_dir(args.out)
    for descriptor in selected:
        target = out if len(selected) == 1 else out / _safe_name(descriptor.agent_ref)
        analyzer.analyze(descriptor, target)
        print(f"analyzed {descriptor.agent_ref} -> {target}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    analysis_dir = Path(args.analysis_dir)
    snapshot = AgentSnapshot.from_dict(load_json(analysis_dir / "agent_snapshot.json"))
    seeds = [RiskSeed.from_dict(item) for item in load_json(analysis_dir / "risk_seeds.json")]
    cases = Tool2Generator(enable_llm_variants=args.llm_variants).generate(snapshot, seeds, count=args.count, out_dir=analysis_dir, profile=args.profile)
    print(f"generated {len(cases)} cases -> {analysis_dir / 'generated_cases.json'}")
    return 0


def _cmd_run_cases(args: argparse.Namespace) -> int:
    analysis_dir = Path(args.analysis_dir)
    snapshot = AgentSnapshot.from_dict(load_json(analysis_dir / "agent_snapshot.json"))
    cases = [GeneratedCase.from_dict(item) for item in load_json(analysis_dir / "generated_cases.json")]
    results = DEFAULT_EXECUTOR_REGISTRY.run(snapshot.analysis_id, cases)
    write_json(analysis_dir / "run_result.json", results)
    print(f"ran {len(results)} sandbox cases -> {analysis_dir / 'run_result.json'}")
    return 0


def _cmd_apply_feedback(args: argparse.Namespace) -> int:
    summary = apply_feedback_to_analysis(args.analysis_dir)
    print(f"updated {summary['updated_seeds']} seeds -> {Path(args.analysis_dir) / 'risk_seeds.json'}")
    return 0


def _cmd_evaluate_tool12(args: argparse.Namespace) -> int:
    descriptors = _load_descriptors(args.descriptors)
    labels = load_label_file(args.labels) if args.labels else None
    summary = evaluate_tool12(
        descriptors,
        args.out,
        labels=labels,
        count=args.count,
        profile=args.profile,
        enable_llm_review=args.llm_review,
        enable_llm_variants=args.llm_variants,
        include_direct_llm=args.include_direct_llm,
        random_seed=args.random_seed,
    )
    print(
        f"evaluated agents={summary['agents']} label_source={summary['label_source']} "
        f"-> {Path(args.out) / 'paper_tables.md'}"
    )
    return 0


def _cmd_import_paper_results(args: argparse.Namespace) -> int:
    summary = import_paper_results(args.input, args.out)
    print(f"imported records={summary['records']} -> {Path(args.out) / 'paper_tables.md'}")
    return 0


def _cmd_run_demo(args: argparse.Namespace) -> int:
    descriptors = _load_descriptors(args.descriptors)
    root = ensure_dir(args.out)
    analyzer = Tool1Analyzer(enable_dynamic_probe=True, enable_llm_review=args.llm_review)
    generator = Tool2Generator(enable_llm_variants=args.llm_variants)
    per_agent_expected = []

    for descriptor in descriptors:
        agent_dir = ensure_dir(root / _safe_name(descriptor.agent_ref))
        session, snapshot, seeds = analyzer.analyze(descriptor, agent_dir)
        cases = generator.generate(snapshot, seeds, count=args.count, out_dir=agent_dir, profile=args.profile)
        results = DEFAULT_EXECUTOR_REGISTRY.run(session.analysis_id, cases)
        write_json(agent_dir / "run_result.json", results)
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
        print(f"{descriptor.agent_ref}: seeds={len(seeds)} cases={len(cases)} results={len(results)}")

    summary = summarize_run_root(root)
    _augment_expected_summary(summary, per_agent_expected)
    write_json(root / "summary.json", summary)
    print(_brief_summary(summary))
    return 0


def _cmd_run_manifest(args: argparse.Namespace) -> int:
    descriptors = load_target_descriptors(args.manifest)
    root = ensure_dir(args.out)
    analyzer = Tool1Analyzer(enable_dynamic_probe=True, enable_llm_review=args.llm_review)
    generator = Tool2Generator(enable_llm_variants=args.llm_variants)
    per_agent_expected = []
    for descriptor in descriptors:
        agent_dir = ensure_dir(root / _safe_name(descriptor.agent_ref))
        session, snapshot, seeds = analyzer.analyze(descriptor, agent_dir)
        cases = generator.generate(snapshot, seeds, count=args.count, out_dir=agent_dir, profile=args.profile)
        results = DEFAULT_EXECUTOR_REGISTRY.run(session.analysis_id, cases)
        write_json(agent_dir / "run_result.json", results)
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
        print(f"{descriptor.agent_ref}: seeds={len(seeds)} cases={len(cases)} results={len(results)}")
    summary = summarize_run_root(root)
    _augment_expected_summary(summary, per_agent_expected)
    write_json(root / "summary.json", summary)
    print(_brief_summary(summary))
    return 0


def _load_descriptors(path: str | Path) -> list[AgentAccessDescriptor]:
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
