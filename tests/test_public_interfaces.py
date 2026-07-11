from __future__ import annotations

import os
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from agenteval.adapters import load_agent_descriptors
from agenteval.api import create_app
from agenteval.cli import main as cli_main
from agenteval.connectors import HttpAgentConnector, _join_url
from agenteval.experiment import CaseExecutor, ExecutorRegistry
from agenteval.io import load_json, write_json
from agenteval.pipeline import AgentEval
from agenteval.schemas import AgentAccessDescriptor, AgentSnapshot, ExecutionContext, GeneratedCase, RunResult
from agenteval.tool1 import Tool1Analyzer


def _mock_target(agent_ref: str = "public_rag_agent") -> dict:
    return {
        "agent_ref": agent_ref,
        "protocol": "mock",
        "static_artifacts": {
            "capabilities": {"rag": True},
            "rag": {"top_k": 2, "source": "test_docs", "external_write": True},
        },
    }


class _ContextExecutor(CaseExecutor):
    name = "context_executor"

    def __init__(self) -> None:
        self.context: ExecutionContext | None = None

    def run(self, analysis_id: str, cases: list[GeneratedCase]) -> list[RunResult]:
        raise AssertionError("new registry path should pass ExecutionContext")

    def run_with_context(self, context: ExecutionContext, cases: list[GeneratedCase]) -> list[RunResult]:
        self.context = context
        return [
            RunResult(
                run_id=f"run_{case.case_id}",
                analysis_id=context.analysis_id,
                seed_id=case.seed_id,
                case_id=case.case_id,
                failure_stage="action_blocked",
                metrics={"attack_success": False},
            )
            for case in cases
        ]


class PublicInterfaceTest(unittest.TestCase):
    def _temp_dir(self) -> Path:
        base = Path.cwd() / ".tmp_tests"
        base.mkdir(exist_ok=True)
        path = base / f"public_{uuid.uuid4().hex}"
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def test_pipeline_prepares_bundle_and_accepts_minimal_results(self) -> None:
        root = self._temp_dir()
        evaluation = AgentEval(root).prepare(
            _mock_target(),
            out_dir=root / "evaluation",
            count=1,
            llm="off",
            dynamic_probe=False,
        )
        bundle = load_json(evaluation.output_dir / "execution_bundle.json")
        self.assertEqual(bundle["schema_version"], "agenteval.execution.v1")
        self.assertEqual(bundle["evaluation_id"], evaluation.evaluation_id)
        self.assertEqual(bundle["context"]["agent_ref"], "public_rag_agent")
        self.assertEqual(len(bundle["cases"]), len(evaluation.cases))
        self.assertTrue(evaluation.cases)

        submission = AgentEval(root).submit_results(
            evaluation,
            {
                "evaluation_id": evaluation.evaluation_id,
                "results": [
                    {
                        "case_id": case.case_id,
                        "failure_stage": "action_blocked",
                        "metrics": {"attack_success": False, "defense_enabled": True},
                    }
                    for case in evaluation.cases
                ],
            },
        )
        self.assertEqual(submission["accepted"], len(evaluation.cases))
        canonical = load_json(evaluation.output_dir / "run_result.json")
        self.assertTrue(all(item["analysis_id"] == evaluation.evaluation_id for item in canonical))
        self.assertTrue(all(item["seed_id"] for item in canonical))
        confidence_after_first = [item["confidence"] for item in load_json(evaluation.output_dir / "risk_seeds.json")]
        AgentEval(root).submit_results(
            evaluation,
            {
                "results": [
                    {
                        "case_id": case.case_id,
                        "failure_stage": "action_blocked",
                        "metrics": {"attack_success": False, "defense_enabled": True},
                    }
                    for case in evaluation.cases
                ]
            },
        )
        confidence_after_retry = [item["confidence"] for item in load_json(evaluation.output_dir / "risk_seeds.json")]
        self.assertEqual(confidence_after_retry, confidence_after_first)
        rerun = AgentEval(root).prepare(
            _mock_target(),
            out_dir=evaluation.output_dir,
            count=1,
            llm="off",
            dynamic_probe=False,
        )
        self.assertNotEqual(rerun.evaluation_id, evaluation.evaluation_id)
        self.assertFalse((rerun.output_dir / "run_result.json").exists())

    def test_result_submission_rejects_partial_or_unknown_stages(self) -> None:
        root = self._temp_dir()
        evaluation = AgentEval(root).prepare(_mock_target(), out_dir=root / "evaluation", llm="off", dynamic_probe=False)
        first = evaluation.cases[0]
        with self.assertRaisesRegex(ValueError, "unsupported failure_stage"):
            AgentEval(root).submit_results(
                evaluation,
                [{"case_id": first.case_id, "failure_stage": "made_up", "metrics": {}}],
            )
        with self.assertRaisesRegex(ValueError, "missing results"):
            AgentEval(root).submit_results(evaluation, {"results": []})
        with self.assertRaisesRegex(ValueError, "unsupported result schema_version"):
            AgentEval(root).submit_results(
                evaluation,
                {"schema_version": "agenteval.results.v99", "results": []},
            )

    def test_registry_passes_context_and_strict_mode_fails_closed(self) -> None:
        root = self._temp_dir()
        executor = _ContextExecutor()
        registry = ExecutorRegistry()
        registry.register(executor.name, executor)
        registry.register("rag_poison_runner", executor)
        descriptor = AgentAccessDescriptor.from_dict(_mock_target("context_target"))
        snapshot = AgentSnapshot("analysis_context", "context_target", "mock", {})
        context = ExecutionContext("analysis_context", "runs/context", descriptor, snapshot)
        case = GeneratedCase(
            case_id="case_context",
            seed_id="seed_context",
            attack_family="prompt_context_injection",
            delivery_mode="direct_input",
            setup={},
            trigger={"message": "safe"},
            expected_signal={"type": "marker"},
            cleanup={},
            executor=executor.name,
            quality_score=0.9,
            provenance={},
            validation_result={"schema_valid": True},
        )
        results = registry.run(context, [case], allow_fallback=False)
        self.assertEqual(results[0].case_id, case.case_id)
        self.assertIs(executor.context, context)
        case.executor = "missing_executor"
        with self.assertRaisesRegex(ValueError, "executor not registered"):
            registry.run(context, [case], allow_fallback=False)

        prepared = AgentEval(root, executor_registry=registry).prepare(
            _mock_target("registry_target"),
            out_dir=root / "registry-evaluation",
            llm="off",
            dynamic_probe=False,
        )
        rag_cases = [item for item in prepared.cases if item.executor == "rag_poison_runner"]
        self.assertTrue(rag_cases)
        self.assertTrue(all(item.validation_result["executor_available"] for item in rag_cases))
        executor.context = None
        sandbox_results = AgentEval(root, executor_registry=registry).execute_sandbox(prepared)
        self.assertIsNone(executor.context)
        self.assertTrue(all(item.feedback["mode"] == "deterministic_sandbox" for item in sandbox_results))

    def test_versioned_api_prepares_and_imports_results(self) -> None:
        root = self._temp_dir()
        client = TestClient(create_app(root))
        self.assertEqual(client.get("/healthz").json()["status"], "ok")
        response = client.post(
            "/api/v1/evaluations",
            json={"target": _mock_target("api_v1_agent"), "count": 1, "llm": "off", "dynamic_probe": False},
        )
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        evaluation_id = body["evaluation_id"]
        self.assertEqual(body["status"], "ready_for_execution")
        cases = body["execution_bundle"]["cases"]
        imported = client.post(
            f"/api/v1/evaluations/{evaluation_id}/results",
            json={
                "results": [
                    {"case_id": case["case_id"], "failure_stage": "not_triggered", "metrics": {"attack_success": False}}
                    for case in cases
                ]
            },
        )
        self.assertEqual(imported.status_code, 200, imported.text)
        self.assertEqual(imported.json()["accepted"], len(cases))
        self.assertEqual(client.get(f"/api/v1/evaluations/{evaluation_id}").status_code, 200)

    def test_cli_run_and_import_results_use_the_public_pipeline(self) -> None:
        root = self._temp_dir()
        analysis_dir = root / "cli"
        self.assertEqual(
            cli_main(
                [
                    "run",
                    "--input",
                    "examples/demo.json",
                    "--out",
                    str(analysis_dir),
                    "--count",
                    "1",
                    "--llm",
                    "off",
                    "--no-dynamic-probe",
                ]
            ),
            0,
        )
        bundle = load_json(analysis_dir / "execution_bundle.json")
        result_path = root / "downstream-results.json"
        write_json(
            result_path,
            {
                "results": [
                    {"case_id": case["case_id"], "failure_stage": "not_triggered", "metrics": {}}
                    for case in bundle["cases"]
                ]
            },
        )
        self.assertEqual(
            cli_main(
                [
                    "import-results",
                    "--analysis-dir",
                    str(analysis_dir),
                    "--results",
                    str(result_path),
                ]
            ),
            0,
        )
        self.assertTrue((analysis_dir / "run_result.json").exists())

    def test_api_disables_local_code_targets_and_supports_optional_token(self) -> None:
        root = self._temp_dir()
        client = TestClient(create_app(root))
        blocked = client.post(
            "/api/v1/evaluations",
            json={
                "target": {
                    "agent_ref": "local_code",
                    "protocol": "python",
                    "python_callable": {"module_path": "examples/fake_direct_agent.py", "callable": "handle"},
                }
            },
        )
        self.assertEqual(blocked.status_code, 403)
        auth_blocked = client.post(
            "/api/v1/evaluations",
            json={
                "target": {
                    "agent_ref": "remote_with_server_secret",
                    "protocol": "http",
                    "endpoint": "http://agent.invalid/chat",
                    "auth_ref": "SERVER_SIDE_SECRET",
                }
            },
        )
        self.assertEqual(auth_blocked.status_code, 403)
        artifact_blocked = client.post(
            "/api/v1/evaluations",
            json={
                "target": {
                    "agent_ref": "remote_file_read",
                    "protocol": "mock",
                    "optional_artifacts": [{"path": "README.md"}],
                }
            },
        )
        self.assertEqual(artifact_blocked.status_code, 403)
        header_blocked = client.post(
            "/api/v1/evaluations",
            json={
                "target": {
                    "agent_ref": "remote_header_secret",
                    "protocol": "http",
                    "endpoint": "http://agent.invalid/chat",
                    "headers": {"Authorization": "Bearer should-not-be-accepted"},
                }
            },
        )
        self.assertEqual(header_blocked.status_code, 403)

        secured = TestClient(create_app(root / "secured", api_token="secret"))
        self.assertEqual(secured.get("/healthz").status_code, 200)
        self.assertEqual(secured.get("/api/v1/evaluations/analysis_missing").status_code, 401)
        authorized = secured.get(
            "/api/v1/evaluations/analysis_missing",
            headers={"X-API-Key": "secret"},
        )
        self.assertEqual(authorized.status_code, 404)

    def test_descriptor_loader_auth_and_url_join_are_consistent(self) -> None:
        descriptors = load_agent_descriptors("examples/current_framework_agents.json")
        self.assertTrue(descriptors)
        self.assertTrue(all(item.agent_ref != "None" for item in descriptors))
        descriptor = AgentAccessDescriptor.from_dict(
            {
                "name": "secured-http-agent",
                "url": "http://agent.local/chat",
                "input_key": "query",
                "output_key": "answer",
                "auth": {"env": "TEST_AGENT_TOKEN", "header": "X-Agent-Token", "scheme": ""},
            }
        )
        self.assertEqual(descriptor.protocol, "http")
        self.assertEqual(descriptor.request_template, {"query": "{{prompt}}"})
        self.assertEqual(_join_url(descriptor.endpoint or "", "/health"), "http://agent.local/health")
        with patch.dict(os.environ, {"TEST_AGENT_TOKEN": "token-value"}):
            headers = HttpAgentConnector(descriptor)._headers()
        self.assertEqual(headers["X-Agent-Token"], "token-value")

    def test_analysis_ids_do_not_collide_within_one_second(self) -> None:
        first = Tool1Analyzer._analysis_id("same-agent")
        second = Tool1Analyzer._analysis_id("same-agent")
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
