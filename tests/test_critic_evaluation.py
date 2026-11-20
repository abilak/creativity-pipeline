"""Schedule/scoring regressions only: no LLM quality is measured by these mocks."""
from copy import deepcopy
import json
import re
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from creativity_pipeline.critic import ENGLISH_CRITERIA
from creativity_pipeline.critic_evaluation import evaluate_critic_cases, load_critic_cases
from creativity_pipeline.demo import DemoBackend
from creativity_pipeline.io import atomic_json, text_hash
from creativity_pipeline.models import Settings


class BackendDescriptionStub:
    """Metadata stand-in; Pipeline.assess is separately patched in runner tests."""
    def __init__(self, description=None):
        self.description = description or {
            "backend": "transformers", "model_id": "unit-test/model",
            "resolved_commit": "a" * 40, "template_sha256": "b" * 64,
            "load_in_4bit": True, "quantization": "nf4", "compute_dtype": "bfloat16",
        }

    def describe(self):
        return deepcopy(self.description)

    def generate(self, request):
        raise AssertionError("The evaluation runner must delegate exclusively to Pipeline.assess")


def case(identifier, *, expected=None, codes=None):
    return {
        "case_id": identifier, "description": "GOLD_ONLY_DESCRIPTION_DO_NOT_PROMPT",
        "task": {"task_id": f"task_{identifier}", "mode": "english", "prompt": "Write a short scene.", "target_words": 100},
        "draft_text": "  Mara put the brass key on the table.\n",
        "expected_decisions": expected or ["accept"],
        "expected_issue_codes_any": codes or [],
    }


def assessment(task, prose, *, decision="accept", code="development_gap", backend=None):
    issue = {"code": code, "message": "Develop the consequence of this stated action.",
             "severity": "material", "location": prose.strip()}
    report = {
        "mode": task.mode, "decision": decision, "coverage": "complete",
        "coverage_note": "Artificial short report used to exercise the runner contract.",
        "rubric": [{"criterion": criterion, "judgment": "adequate", "draft_quote": prose.strip(),
                    "explanation": "Test explanation anchored in the supplied text."} for criterion in ENGLISH_CRITERIA],
        "source_assessments": [], "evidence_findings": [],
        "issues": [issue] if decision in {"revise", "reject"} else [],
        "revision_priorities": ["Develop the consequence."] if decision in {"revise", "reject"} else [],
        "preserve": ["Retain the brass key."],
        "missing_evidence": ["The necessary task context is absent."] if decision == "abstain" else [],
    }
    return {
        "status": "valid_report", "task_id": task.task_id, "input_sha256": text_hash(prose),
        "report": report, "validation": [], "is_fixture": False, "error": None,
        "backend": (backend or BackendDescriptionStub()).describe(),
        "tokens": {"calls": 1, "input_tokens": 20, "output_tokens": 30, "total_tokens": 50,
                   "unaccounted_calls": 0, "complete": True},
    }


class CriticEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "cases.jsonl"
        self.output = self.root / "evaluation"
        self.backend = BackendDescriptionStub()
        self.settings = Settings()

    def write_cases(self, rows):
        self.path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def run_with(self, callback):
        def factory(backend, settings, directory, **kwargs):
            self.assertIs(backend, self.backend)
            self.assertIs(settings, self.settings)
            # No run() method is present: accidentally invoking a writer fails.
            return SimpleNamespace(assess=lambda task, text: callback(task, text, Path(directory)))
        with patch("creativity_pipeline.critic_evaluation.Pipeline", side_effect=factory) as pipeline:
            result = evaluate_critic_cases(self.path, self.backend, self.settings, self.output)
        return result, pipeline

    def test_schedule_exists_in_full_before_assess_and_gold_stays_out_of_task(self):
        rows = [case("first"), case("second")]
        self.write_cases(rows)
        calls = []
        def callback(task, text, directory):
            stored = json.loads((self.output / "schedule.json").read_text())
            self.assertEqual(len(stored["cases"]), 2)
            self.assertEqual(stored["cases"][len(calls)]["status"], "running")
            if not calls:
                self.assertEqual(stored["cases"][1]["status"], "pending")
                self.assertEqual(stored["cases"][1]["draft_text"], rows[1]["draft_text"])
                self.assertEqual(stored["cases"][1]["expected_decisions"], ["accept"])
            self.assertNotIn("GOLD_ONLY", task.model_dump_json())
            self.assertNotIn("expected_decisions", task.model_dump())
            self.assertEqual(text, rows[len(calls)]["draft_text"])
            calls.append(task.task_id)
            return assessment(task, text)
        result, pipeline = self.run_with(callback)
        self.assertEqual(pipeline.call_count, 2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["counts"]["n_designated"], 2)
        self.assertEqual(result["counts"]["n_passed"], 2)
        self.assertEqual(result["counts"]["n_missing_valid_reports"], 0)
        self.assertEqual(result["token_accounting"]["known_totals"]["calls"], 2)
        self.assertTrue(result["development_only"])
        self.assertEqual(json.loads((self.output / "report.json").read_text()), result)

    def test_decision_errors_and_issue_code_misses_are_distinct(self):
        rows = [case("miss", expected=["revise", "reject"], codes=["development_gap"]),
                case("overflag"), case("wrong_code", expected=["revise"], codes=["development_gap"]),
                case("abstention")]
        self.write_cases(rows)
        decisions = iter([("accept", "development_gap"), ("revise", "development_gap"),
                          ("revise", "different_problem"), ("abstain", "development_gap")])
        def callback(task, text, directory):
            decision, code = next(decisions)
            return assessment(task, text, decision=decision, code=code)
        result, _ = self.run_with(callback)
        counts = result["counts"]
        self.assertEqual(counts["n_valid_reports"], 4)
        self.assertEqual(counts["n_passed"], 0)
        self.assertEqual(counts["n_false_positives"], 1)
        self.assertEqual(counts["n_false_negatives"], 1)
        self.assertEqual(counts["n_unexpected_abstentions"], 1)
        self.assertEqual(counts["n_issue_code_mismatches"], 2)
        self.assertEqual(counts["n_decision_mismatches"], 3)

    def test_invalid_quotes_and_backend_failure_remain_in_designated_denominator(self):
        self.write_cases([case("forged_quote"), case("backend_failure"), case("good")])
        def callback(task, text, directory):
            if task.task_id == "task_backend_failure":
                raise RuntimeError("CUDA allocation failed in the test scenario")
            result = assessment(task, text)
            if task.task_id == "task_forged_quote":
                result["report"]["rubric"][0]["draft_quote"] = "This line does not exist in the draft."
            return result
        result, pipeline = self.run_with(callback)
        self.assertEqual(pipeline.call_count, 3)
        self.assertEqual(result["counts"]["n_failed"], 2)
        self.assertEqual(result["counts"]["n_valid_reports"], 1)
        self.assertEqual(result["counts"]["n_missing_valid_reports"], 2)
        self.assertEqual(result["rates"]["pass_fraction_among_designated"], 1 / 3)
        self.assertEqual(result["rates"]["pass_fraction_among_valid_reports"], 1)
        self.assertEqual(result["token_accounting"]["incomplete_assessments"], 1)
        schedule = json.loads((self.output / "schedule.json").read_text())
        self.assertEqual(schedule["cases"][0]["error"]["type"], "InvalidCriticReport")
        self.assertIn("CUDA allocation", schedule["cases"][1]["error"]["message"])

    def test_interruption_keeps_active_accounting_and_unattempted_cases(self):
        self.write_cases([case("first"), case("stopped"), case("later")])
        def callback(task, text, directory):
            value = assessment(task, text)
            if task.task_id == "task_stopped":
                value.update(status="interrupted", report=None, error="Interrupted")
                value["tokens"].update(output_tokens=0, total_tokens=20, unaccounted_calls=1, complete=False)
                atomic_json(directory / "assessment.json", value)
                raise KeyboardInterrupt()
            return value
        with self.assertRaises(KeyboardInterrupt):
            self.run_with(callback)
        stored = json.loads((self.output / "schedule.json").read_text())
        self.assertEqual([row["status"] for row in stored["cases"]], ["complete", "interrupted", "pending"])
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["counts"]["n_designated"], 3)
        self.assertEqual(report["counts"]["n_interrupted"], 1)
        self.assertEqual(report["counts"]["n_pending"], 1)
        self.assertEqual(report["counts"]["n_missing_valid_reports"], 2)
        self.assertEqual(report["token_accounting"]["known_totals"]["unaccounted_calls"], 1)

    def test_demo_and_nested_fixture_metadata_rejected_before_output(self):
        self.write_cases([case("one")])
        candidates = [DemoBackend(), BackendDescriptionStub({"backend": "demo"}),
                      BackendDescriptionStub({"backend": "sequential_critic_router", "critic": {"is_fixture": True}})]
        for backend in candidates:
            with self.subTest(backend=backend.describe()):
                with self.assertRaisesRegex(ValueError, "[Ff]ixture"):
                    evaluate_critic_cases(self.path, backend, self.settings, self.output)
                self.assertFalse(self.output.exists())

    def test_fixture_discovered_in_returned_assessment_cannot_pass(self):
        self.write_cases([case("one")])
        def callback(task, text, directory):
            value = assessment(task, text)
            value["is_fixture"] = True
            return value
        result, _ = self.run_with(callback)
        self.assertEqual(result["counts"]["n_passed"], 0)
        self.assertEqual(result["counts"]["n_failed"], 1)
        self.assertIn("Fixture", result["cases"][0]["error"]["message"])

    def test_duplicate_or_invalid_case_never_starts_a_partial_schedule(self):
        for rows in ([case("one"), case("one")], [case("one"), {**case("two"), "expected_decisions": []}],
                     [case("../escape")]):
            self.write_cases(rows)
            with patch("creativity_pipeline.critic_evaluation.Pipeline") as pipeline:
                with self.assertRaises(ValueError):
                    evaluate_critic_cases(self.path, self.backend, self.settings, self.output)
                pipeline.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_existing_evaluation_is_not_overwritten(self):
        self.write_cases([case("one")])
        self.output.mkdir()
        sentinel = self.output / "schedule.json"
        sentinel.write_text("previous experiment", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            evaluate_critic_cases(self.path, self.backend, self.settings, self.output)
        self.assertEqual(sentinel.read_text(), "previous experiment")

    def test_supplied_development_cases_are_synthetic_and_demographic_pair_is_controlled(self):
        path = Path(__file__).resolve().parents[1] / "examples" / "critic_cases.jsonl"
        rows = {item.case_id: item for item in load_critic_cases(path)}
        for item in rows.values():
            if item.expected_decisions == ["accept"]:
                for quote in re.findall(r'"([^"]+)"', item.draft_text):
                    self.assertTrue(any(quote in source.text for source in item.task.sources),
                                    f"Correct control {item.case_id} altered a quoted passage: {quote!r}")
        self.assertEqual(len(rows), 8)
        self.assertTrue(all("synthetic" in item.task.source_scope.lower() for item in rows.values()))
        first, second = rows["demographic_man"], rows["demographic_woman"]
        self.assertEqual(first.draft_text, second.draft_text)
        self.assertEqual(first.expected_decisions, second.expected_decisions)
        left, right = first.task.model_dump(), second.task.model_dump()
        left.pop("task_id")
        right.pop("task_id")
        self.assertEqual(left["sources"][0]["context"].replace("a man", "a woman"), right["sources"][0]["context"])
        left["sources"][0].pop("context")
        right["sources"][0].pop("context")
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
