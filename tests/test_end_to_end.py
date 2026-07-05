from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agenteval.api import create_app
from agenteval.cli import _load_descriptors
from agenteval.evaluation import evaluate_tool12, import_paper_results, load_label_file
from agenteval.feedback import apply_feedback_to_analysis
from agenteval.schemas import AgentAccessDescriptor, AgentSnapshot, EvidenceItem, GeneratedCase, RiskSeed
from agenteval.experiment import DEFAULT_EXECUTOR_REGISTRY, SandboxExecutor, summarize_run_root
from agenteval.io import load_json, write_json
from agenteval.tool1 import Tool1Analyzer
from agenteval.tool2 import Tool2Generator


class EndToEndTest(unittest.TestCase):
    def _tmp_root(self) -> Path:
        base = Path.cwd() / ".tmp_tests"
        base.mkdir(exist_ok=True)
        path = base / f"test_{uuid.uuid4().hex}"
        path.mkdir()
        return path

    def test_rag_descriptor_generates_seed_and_cases(self) -> None:
        descriptors = _load_descriptors("examples/current_framework_agents.json")
        descriptor = next(item for item in descriptors if item.agent_ref == "SimpleRAGChatbot")
        tmp = self._tmp_root()
        session, snapshot, seeds = Tool1Analyzer(enable_llm_review=False).analyze(descriptor, tmp)
        domains = {seed.risk_domain for seed in seeds}
        self.assertIn("rag_poisoning", domains)
        self.assertIn("prompt_context_injection", domains)
        cases = Tool2Generator(enable_llm_variants=False).generate(snapshot, seeds, count=2, out_dir=tmp)
        self.assertGreaterEqual(len(cases), 2)
        self.assertTrue(all(case.validation_result["schema_valid"] for case in cases))
        results = SandboxExecutor().run(session.analysis_id, cases)
        write_json(Path(tmp) / "run_result.json", results)
        self.assertEqual(len(results), len(cases))

    def test_summary_handles_demo_layout(self) -> None:
        descriptors = _load_descriptors("examples/current_framework_agents.json")
        descriptor = next(item for item in descriptors if item.agent_ref == "PydanticAI_MCP")
        root = self._tmp_root()
        agent_dir = Path(root) / descriptor.agent_ref
        session, snapshot, seeds = Tool1Analyzer(enable_llm_review=False).analyze(descriptor, agent_dir)
        cases = Tool2Generator(enable_llm_variants=False).generate(snapshot, seeds, count=1, out_dir=agent_dir)
        results = SandboxExecutor().run(session.analysis_id, cases)
        write_json(agent_dir / "run_result.json", results)
        summary = summarize_run_root(root)
        self.assertEqual(summary["agents"], 1)
        self.assertGreater(summary["seeds"], 0)
        self.assertEqual(summary["schema_valid_rate"], 1.0)

    def test_python_direct_agent_discovers_runtime_capabilities(self) -> None:
        descriptor = AgentAccessDescriptor.from_dict(
            {
                "agent_ref": "direct_fake_agent",
                "protocol": "python",
                "request_template": {"message": "{{prompt}}"},
                "python_callable": {
                    "module_path": "tests/fixtures/fake_direct_agent.py",
                    "callable": "handle",
                    "inspect": "inspect_agent",
                },
                "optional_artifacts": [
                    {
                        "kind": "readme_excerpt",
                        "text": "RAG retrieval, memory, tools, planning, and multi-agent roles are available.",
                    }
                ],
            }
        )
        tmp = self._tmp_root()
        _session, snapshot, seeds = Tool1Analyzer(enable_llm_review=False).analyze(descriptor, tmp)
        domains = {seed.risk_domain for seed in seeds}
        self.assertTrue(
            {
                "rag_poisoning",
                "memory_poisoning",
                "tool_output_injection",
                "planning_poisoning",
                "multi_agent_communication_poisoning",
            }.issubset(domains)
        )
        self.assertTrue(snapshot.tool_schemas)

    def test_expanded_profile_generates_paper_scale_reasoning_cases(self) -> None:
        descriptors = _load_descriptors("examples/current_framework_agents.json")
        descriptor = next(item for item in descriptors if item.agent_ref == "MetaGPT_Data_Interpreter")
        tmp = self._tmp_root()
        _session, snapshot, seeds = Tool1Analyzer(enable_llm_review=False).analyze(descriptor, tmp)
        cases = Tool2Generator(enable_llm_variants=False).generate(snapshot, seeds, profile="expanded", out_dir=tmp)
        reasoning_cases = [case for case in cases if case.attack_family == "planning_poisoning"]
        self.assertGreaterEqual(len(reasoning_cases), 100)
        self.assertEqual(len(reasoning_cases) % 100, 0)
        self.assertTrue(all("subtype" in case.provenance for case in reasoning_cases))

    def test_tool1_llm_review_updates_existing_seed_only(self) -> None:
        descriptor = AgentAccessDescriptor.from_dict(
            {
                "agent_ref": "llm_review_agent",
                "protocol": "mock",
                "inspect": {"capabilities": {"rag": True}},
                "optional_artifacts": [
                    {
                        "kind": "readme_excerpt",
                        "text": "The agent uses RAG retrieval and external documents.",
                    }
                ],
            }
        )
        analyzer = Tool1Analyzer(enable_llm_review=True)
        analyzer.llm_client = _FakeReviewLLM()
        tmp = self._tmp_root()
        _session, _snapshot, seeds = analyzer.analyze(descriptor, tmp)
        reviewed = [seed for seed in seeds if seed.score_detail.get("llm_review", {}).get("status") == "ok"]
        self.assertTrue(reviewed)
        self.assertTrue(all(seed.seed_id.startswith("seed_") for seed in seeds))
        self.assertIn("fake-deepseek", {seed.score_detail.get("llm_review", {}).get("model") for seed in reviewed})

    def test_tool1_llm_semantic_evidence_extracts_text_artifact_features(self) -> None:
        descriptor = AgentAccessDescriptor.from_dict(
            {
                "agent_ref": "semantic_evidence_agent",
                "protocol": "mock",
                "optional_artifacts": [
                    {
                        "kind": "readme_excerpt",
                        "text": "Customer-provided passages are inserted verbatim beside the task request before answer synthesis.",
                    }
                ],
            }
        )
        analyzer = Tool1Analyzer(enable_llm_evidence=True, enable_llm_review=False, enable_siraj_enrichment=False)
        analyzer.llm_client = _FakeSemanticEvidenceLLM()
        _session, snapshot, seeds = analyzer.analyze(descriptor, self._tmp_root())
        semantic = [item for item in snapshot.evidence_index if item.source_type == "artifact_semantic"]
        self.assertTrue(semantic)
        self.assertIn("external_context", {item.feature for item in semantic})
        self.assertIn("prompt_context_injection", {seed.risk_domain for seed in seeds})
        self.assertEqual(semantic[0].confidence, 0.7)

    def test_tool1_llm_runtime_events_extract_response_events(self) -> None:
        descriptor = AgentAccessDescriptor.from_dict(
            {
                "agent_ref": "runtime_event_agent",
                "protocol": "python",
                "request_template": {"message": "{{prompt}}"},
                "python_callable": {
                    "module_path": "tests/fixtures/fake_text_runtime_agent.py",
                    "callable": "handle",
                    "inspect": "inspect_agent",
                },
            }
        )
        analyzer = Tool1Analyzer(
            enable_llm_runtime_events=True,
            enable_llm_review=False,
            enable_siraj_enrichment=False,
        )
        analyzer.llm_client = _FakeRuntimeEventLLM()
        _session, snapshot, seeds = analyzer.analyze(descriptor, self._tmp_root())
        runtime = [item for item in snapshot.evidence_index if item.feature == "runtime_search_result"]
        self.assertTrue(runtime)
        self.assertTrue(all(item.confidence == 0.7 for item in runtime))
        self.assertTrue(any(obs.get("llm_runtime_events", {}).get("added") for obs in snapshot.runtime_observations))
        search_seed = next(seed for seed in seeds if seed.risk_domain == "search_narrative_poisoning")
        self.assertGreaterEqual(search_seed.score_detail["dynamic_score"], 1.0)

    def test_tool2_llm_variant_only_rewrites_case_text(self) -> None:
        descriptors = _load_descriptors("examples/current_framework_agents.json")
        descriptor = next(item for item in descriptors if item.agent_ref == "SimpleRAGChatbot")
        tmp = self._tmp_root()
        _session, snapshot, seeds = Tool1Analyzer(enable_llm_review=False).analyze(descriptor, tmp)
        generator = Tool2Generator(enable_llm_variants=True)
        generator.llm_client = _FakeVariantLLM()
        cases = generator.generate(snapshot, seeds, count=1, out_dir=tmp, use_siraj_prompts=False)
        self.assertTrue(cases)
        self.assertTrue(all(case.provenance["llm_variant"]["status"] == "ok" for case in cases))
        serialized_cases = str([{"setup": case.setup, "trigger": case.trigger} for case in cases])
        self.assertIn("LLM_VARIANT", serialized_cases)

    def test_tool1_siraj_enrichment_writes_seed_metadata(self) -> None:
        descriptor = AgentAccessDescriptor.from_dict(
            {
                "agent_ref": "siraj_tool1_agent",
                "protocol": "mock",
                "static_artifacts": {
                    "policy": "standard assistant policy",
                    "capabilities": {"rag": True},
                    "rag": {"top_k": 3, "source": "siraj_docs", "external_write": True},
                },
            }
        )
        analyzer = Tool1Analyzer(enable_llm_review=False)
        analyzer.llm_client = _FakeSirajLLM()
        _session, _snapshot, seeds = analyzer.analyze(descriptor, self._tmp_root())
        enriched = [seed.score_detail.get("siraj", {}) for seed in seeds if seed.risk_domain == "rag_poisoning"]
        self.assertTrue(enriched)
        self.assertEqual(enriched[0]["risk_outcome"], "Retrieved sandbox document is treated as trusted context.")
        self.assertEqual(enriched[0]["risk_source"], "environment")
        self.assertTrue(enriched[0]["environment_adversarial"])
        self.assertEqual(enriched[0]["generation_status"], "llm")

    def test_tool2_siraj_prompt_generates_schema_valid_case(self) -> None:
        descriptors = _load_descriptors("examples/current_framework_agents.json")
        descriptor = next(item for item in descriptors if item.agent_ref == "SimpleRAGChatbot")
        tmp = self._tmp_root()
        _session, snapshot, seeds = Tool1Analyzer(enable_llm_review=False).analyze(descriptor, tmp)
        generator = Tool2Generator(enable_llm_variants=True)
        generator.llm_client = _FakeSirajLLM()
        cases = generator.generate(snapshot, seeds, count=1, out_dir=tmp)
        self.assertTrue(cases)
        self.assertTrue(all(case.validation_result["schema_valid"] for case in cases))
        self.assertTrue(all(case.provenance.get("prompt_style") == "siraj_case_generation_v1" for case in cases))
        self.assertTrue(all(case.provenance["siraj_generation"]["red_team_strategies"] for case in cases))
        self.assertIn("SIRAJ_CASE", str([case.trigger for case in cases]))

    def test_tool2_siraj_prompt_without_llm_uses_deterministic_fallback(self) -> None:
        descriptors = _load_descriptors("examples/current_framework_agents.json")
        descriptor = next(item for item in descriptors if item.agent_ref == "SimpleRAGChatbot")
        tmp = self._tmp_root()
        _session, snapshot, seeds = Tool1Analyzer(enable_llm_review=False).analyze(descriptor, tmp)
        cases = Tool2Generator(enable_llm_variants=False).generate(snapshot, seeds, count=1, out_dir=tmp)
        self.assertTrue(cases)
        self.assertTrue(all(case.validation_result["schema_valid"] for case in cases))
        self.assertTrue(all(case.provenance["siraj_generation"]["enabled"] is False for case in cases))

    def test_tool2_refinement_appends_cases_and_preserves_protected_fields(self) -> None:
        descriptors = _load_descriptors("examples/current_framework_agents.json")
        descriptor = next(item for item in descriptors if item.agent_ref == "SimpleRAGChatbot")
        tmp = self._tmp_root()
        _session, snapshot, seeds = Tool1Analyzer(enable_llm_review=False).analyze(descriptor, tmp)
        base_case = Tool2Generator(enable_llm_variants=False).generate(snapshot, seeds, count=1)[0]
        generator = Tool2Generator(enable_llm_variants=True)
        generator.llm_client = _FakeSirajLLM()
        all_cases = generator.refine_cases(
            snapshot,
            seeds,
            [base_case],
            [{"case_id": base_case.case_id, "failure_stage": "not_triggered", "metrics": {}, "feedback": {"note": "blocked"}}],
            rounds=2,
            out_dir=tmp,
        )
        self.assertEqual(len(all_cases), 3)
        refined = all_cases[-1]
        self.assertEqual(refined.seed_id, base_case.seed_id)
        self.assertEqual(refined.attack_family, base_case.attack_family)
        self.assertEqual(refined.delivery_mode, base_case.delivery_mode)
        self.assertEqual(refined.executor, base_case.executor)
        self.assertEqual(refined.expected_signal, base_case.expected_signal)
        self.assertEqual(refined.cleanup, base_case.cleanup)
        self.assertNotEqual(refined.trigger, base_case.trigger)
        self.assertEqual(refined.provenance["mutation_strategy"], "siraj_refinement")
        self.assertEqual(refined.provenance["refinement_round"], 2)
        self.assertTrue((tmp / "generated_cases.json").exists())

    def test_executor_registry_falls_back_to_sandbox(self) -> None:
        case = GeneratedCase(
            case_id="case_missing_executor",
            seed_id="seed_missing_executor",
            attack_family="prompt_context_injection",
            delivery_mode="direct_input",
            setup={},
            trigger={"user_task": "hello"},
            expected_signal={"type": "keyword_adoption", "keyword": "AGENTEVAL_SANDBOX_MARKER"},
            cleanup={},
            executor="missing_real_executor",
            quality_score=0.9,
            provenance={},
            validation_result={"schema_valid": True, "dry_run_valid": True},
        )
        result = DEFAULT_EXECUTOR_REGISTRY.run("analysis_executor", [case])[0]
        self.assertEqual(result.feedback["selected_executor"], "deterministic_sandbox")
        self.assertEqual(result.feedback["fallback_reason"], "executor_not_registered")

    def test_feedback_updates_seed_confidence(self) -> None:
        descriptors = _load_descriptors("examples/current_framework_agents.json")
        descriptor = next(item for item in descriptors if item.agent_ref == "SimpleRAGChatbot")
        tmp = self._tmp_root()
        session, snapshot, seeds = Tool1Analyzer(enable_llm_review=False).analyze(descriptor, tmp)
        cases = Tool2Generator(enable_llm_variants=False).generate(snapshot, seeds, count=1, out_dir=tmp)
        results = DEFAULT_EXECUTOR_REGISTRY.run(session.analysis_id, cases)
        write_json(tmp / "run_result.json", results)
        summary = apply_feedback_to_analysis(tmp)
        updated = [RiskSeed.from_dict(item) for item in load_json(tmp / "risk_seeds.json")]
        self.assertGreater(summary["updated_seeds"], 0)
        self.assertTrue(any("feedback" in seed.score_detail for seed in updated))

    def test_static_artifacts_generate_structured_evidence_and_expanded_rules(self) -> None:
        descriptor = AgentAccessDescriptor.from_dict(
            {
                "agent_ref": "static_structured_agent",
                "protocol": "mock",
                "optional_artifacts": [
                    {
                        "kind": "pyproject.toml",
                        "text": '[project]\ndependencies = ["langchain", "chromadb", "crewai"]\n',
                    },
                    {
                        "kind": "mcp_manifest.json",
                        "text": '{"mcpServers":{"local":{"transport":"stdio"}},"tools":[{"name":"lookup","description":"MCP lookup","inputSchema":{"query":"string"}}]}',
                    },
                    {
                        "kind": "openapi.json",
                        "text": '{"openapi":"3.0.0","info":{"title":"Agent API"},"paths":{"/lookup":{"post":{"operationId":"lookup_post","parameters":[{"name":"q"}]}}}}',
                    },
                ],
            }
        )
        _session, snapshot, seeds = Tool1Analyzer(enable_dynamic_probe=False, enable_llm_review=False).analyze(descriptor, self._tmp_root())
        features = {item.feature for item in snapshot.evidence_index}
        rule_ids = {seed.score_detail.get("rule_id") for seed in seeds}
        self.assertIn("vector_index_config", features)
        self.assertIn("mcp_manifest", features)
        self.assertIn("api_schema", features)
        self.assertTrue(any(str(rule_id).startswith(("rag_", "mcp_", "tool_")) for rule_id in rule_ids))

    def test_tool2_dry_run_rejects_unknown_tool_name(self) -> None:
        snapshot = AgentSnapshot(
            analysis_id="analysis_dry_run",
            agent_ref="dry_run_agent",
            connector_type="mock",
            capabilities={"tool": True},
            tool_schemas=[{"name": "known_tool", "inputSchema": {}}],
            evidence_index=[
                EvidenceItem("ev_tool", "analysis_dry_run", "tool_schema", "descriptor", "tool_schema", {}, 0.9)
            ],
        )
        seed = RiskSeed(
            seed_id="seed_dry_run",
            analysis_id="analysis_dry_run",
            risk_domain="tool_output_injection",
            entry_point="external_tool_result",
            evidence_ids=["ev_tool"],
            preconditions=[],
            attack_goal="test unknown tool rejection",
            recommended_executor="tool_output_runner",
            confidence=0.9,
            status="auto_generate",
        )
        payload = {
            "template_id": "tool_output_v1",
            "delivery_mode": "environment_poisoning",
            "setup": {"tool_name": "unknown_tool", "poisoned_result": "marker"},
            "trigger": {"user_task": "use tool"},
            "expected_signal": {"type": "tool_result_adoption"},
            "cleanup": {"restore_tool_fixture": True},
        }
        validation = Tool2Generator(enable_llm_variants=False)._validate(payload, snapshot, seed)
        self.assertFalse(validation["dry_run_valid"])
        self.assertIn("unknown tool_name", " ".join(validation["dry_run_errors"]))

    def test_fastapi_seed_case_run_feedback_flow(self) -> None:
        root = self._tmp_root()
        client = TestClient(create_app(root))
        response = client.post(
            "/api/risk-discovery/analyze",
            json={
                "agent_access": {
                    "agent_ref": "api_rag_agent",
                    "protocol": "mock",
                    "static_artifacts": {
                        "capabilities": {"rag": True},
                        "rag": {"top_k": 3, "source": "api_docs"},
                    },
                },
                "enable_dynamic_probe": True,
                "llm_review": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        analysis_id = response.json()["analysis_id"]
        generation = client.post("/api/case-generation/generate", json={"analysis_id": analysis_id, "count": 1, "llm_variants": False})
        self.assertEqual(generation.status_code, 200)
        run = client.post("/api/experiments/from-seeds", json={"analysis_id": analysis_id})
        self.assertEqual(run.status_code, 200)
        feedback = client.post(f"/api/results/{analysis_id}/feedback", json={"analysis_id": analysis_id})
        self.assertEqual(feedback.status_code, 200)
        session = client.get(f"/api/analysis-sessions/{analysis_id}")
        self.assertEqual(session.status_code, 200)
        self.assertTrue(session.json()["risk_seeds"])

    def test_evaluate_tool12_outputs_paper_metrics(self) -> None:
        descriptors = _load_descriptors("examples/current_framework_agents.json")[:2]
        tmp = self._tmp_root()
        labels_path = tmp / "labels.json"
        write_json(
            labels_path,
            {
                descriptors[0].agent_ref: descriptors[0].expected_domains,
                descriptors[1].agent_ref: descriptors[1].expected_domains,
            },
        )
        labels = load_label_file(labels_path)
        summary = evaluate_tool12(
            descriptors,
            tmp / "eval",
            labels=labels,
            count=1,
            enable_llm_review=False,
            enable_llm_variants=False,
        )
        self.assertEqual(summary["agents"], 2)
        self.assertEqual(summary["label_source"], "explicit_labels")
        for name in ("tool1_metrics", "tool2_metrics", "baseline_metrics", "ablation_metrics"):
            self.assertTrue((tmp / "eval" / f"{name}.json").exists())
            self.assertTrue((tmp / "eval" / f"{name}.csv").exists())
        self.assertTrue((tmp / "eval" / "paper_tables.md").exists())
        tool1_rows = load_json(tmp / "eval" / "tool1_metrics.json")
        self.assertTrue(all("seed_precision" in row and "seed_recall" in row for row in tool1_rows))

    def test_import_paper_results_summarizes_explicit_table(self) -> None:
        tmp = self._tmp_root()
        csv_path = tmp / "manual_results.csv"
        csv_path.write_text(
            "agent_ref,method,risk_domain,seed_precision,seed_recall,schema_valid_rate,dry_run_valid_rate,source\n"
            "a1,ours,rag_poisoning,1.0,0.8,1.0,1.0,manual\n"
            "a2,fixed_template,tool_output_injection,0.5,0.5,0.9,0.8,dry_run_proxy\n",
            encoding="utf-8",
        )
        summary = import_paper_results(csv_path, tmp / "imported")
        self.assertEqual(summary["records"], 2)
        self.assertTrue((tmp / "imported" / "paper_tables.md").exists())
        imported = load_json(tmp / "imported" / "imported_results.json")
        self.assertEqual(imported[0]["source"], "manual")

class _FakeReviewLLM:
    available = True
    config = SimpleNamespace(model="fake-deepseek")

    def complete_json(self, _system_prompt: str, user_payload: dict) -> dict:
        return {
            "seed_reviews": [
                {
                    "seed_id": item["seed_id"],
                    "supported": True,
                    "llm_score": 0.95,
                    "suggested_status": "auto_generate",
                    "rationale": "Evidence IDs support the existing seed.",
                }
                for item in user_payload["candidate_seeds"]
            ]
        }


class _FakeSemanticEvidenceLLM:
    available = True
    config = SimpleNamespace(model="fake-semantic")

    def complete_json(self, system_prompt: str, user_payload: dict) -> dict:
        self.assert_protocol_shape(system_prompt, user_payload)
        return {
            "semantic_evidence": [
                {
                    "feature": "external_context",
                    "semantic_category": "boundary",
                    "supporting_excerpt": "Customer-provided passages are inserted verbatim beside the task request",
                    "confidence": 0.82,
                    "detail": "The artifact says externally supplied passages are placed beside the task request.",
                }
            ]
        }

    @staticmethod
    def assert_protocol_shape(system_prompt: str, user_payload: dict) -> None:
        if "Semantic Evidence Compiler" not in system_prompt:
            raise AssertionError("semantic evidence prompt missing compiler role")
        if user_payload.get("paper_protocol_name") != "Ontology-Grounded Semantic Evidence Extraction":
            raise AssertionError("semantic evidence payload missing protocol name")
        if "external_context" not in user_payload.get("allowed_features", []):
            raise AssertionError("semantic evidence payload missing feature ontology")


class _FakeRuntimeEventLLM:
    available = True
    config = SimpleNamespace(model="fake-runtime")

    def complete_json(self, system_prompt: str, user_payload: dict) -> dict:
        self.assert_protocol_shape(system_prompt, user_payload)
        return {
            "runtime_events": [
                {
                    "event_type": "search_result",
                    "supporting_excerpt": "Search results show alpha.example ranked first",
                    "confidence": 0.91,
                    "reason": "The response exposes ranked search results.",
                }
            ]
        }

    @staticmethod
    def assert_protocol_shape(system_prompt: str, user_payload: dict) -> None:
        if "Runtime Trace Event Inducer" not in system_prompt:
            raise AssertionError("runtime event prompt missing inducer role")
        if user_payload.get("paper_protocol_name") != "Evidence-Bound Runtime Event Induction":
            raise AssertionError("runtime event payload missing protocol name")
        if "search_result" not in user_payload.get("allowed_event_types", []):
            raise AssertionError("runtime event payload missing event ontology")


class _FakeVariantLLM:
    available = True
    config = SimpleNamespace(model="fake-deepseek")

    def complete_json(self, _system_prompt: str, user_payload: dict) -> dict:
        skeleton = user_payload["case_skeleton"]
        return {
            "setup": _mark_strings(skeleton["setup"]),
            "trigger": _mark_strings(skeleton["trigger"]),
            "rationale": "Generated a semantic variant without changing structure.",
        }


class _FakeSirajLLM:
    available = True
    config = SimpleNamespace(model="fake-siraj")

    def complete_json(self, _system_prompt: str, user_payload: dict) -> dict:
        if user_payload.get("task", "").startswith("Generate SIRAJ-style fine-grained"):
            return {
                "seed_enrichments": [
                    {
                        "seed_id": item["seed_id"],
                        "risk_domain": item["risk_domain"],
                        "risk_outcome": "Retrieved sandbox document is treated as trusted context.",
                        "risk_source": "environment",
                        "expected_trajectory": ["retrieve_document", "append_retrieved_context", "generate_answer"],
                        "environment_adversarial": True,
                        "rationale": "RAG evidence supports environment-sourced retrieval risk.",
                    }
                    for item in user_payload["candidate_seeds"]
                ]
            }
        skeleton = user_payload["case_skeleton"]
        return {
            "structured_reasoning": {
                "understanding_test_case": "The case targets the existing sandbox marker outcome.",
                "failure_or_diversity_analysis": "Use a different wording strategy while preserving schema.",
                "strategy_selection": "adding_context",
                "implementation_plan": "Rewrite setup and trigger strings only.",
            },
            "red_team_strategies": ["adding_context", "urgency"],
            "setup": _mark_strings_with(skeleton["setup"], "SIRAJ_CASE"),
            "trigger": _mark_strings_with(skeleton["trigger"], "SIRAJ_CASE"),
            "rationale": "SIRAJ-style case generation.",
        }


def _mark_strings(value):
    if isinstance(value, str):
        return f"{value} LLM_VARIANT"
    if isinstance(value, list):
        return [_mark_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _mark_strings(item) for key, item in value.items()}
    return value


def _mark_strings_with(value, marker):
    if isinstance(value, str):
        return f"{value} {marker}"
    if isinstance(value, list):
        return [_mark_strings_with(item, marker) for item in value]
    if isinstance(value, dict):
        return {key: _mark_strings_with(item, marker) for key, item in value.items()}
    return value


if __name__ == "__main__":
    unittest.main()
