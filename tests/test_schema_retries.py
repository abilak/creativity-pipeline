"""A malformed critic report must recover explicitly or fail without approval.

These scripted responses exercise orchestration and contracts, not model quality.
"""
import json
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from creativity_pipeline.critic import CriticReport, critic_messages
from creativity_pipeline.io import load_json, text_hash
from creativity_pipeline.pipeline import Pipeline
from creativity_pipeline.recovery import load_saved_run
from test_pipeline import (Reply, ScriptedBackend, draft, english_task, history_task,
                           historical_draft, review, settings, valid_plan)
from test_pipeline_critic import report


class SchemaRetryRegressionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def run_script(self, task, replies, *, name="run", **changes):
        backend = ScriptedBackend(replies)
        config = settings(semantic_critic=True, critic_tokens=1024,
                          schema_retries=1, **changes)
        output = self.root / name
        result = Pipeline(backend, config, output).run(task)
        self.assertEqual(load_json(output / "summary.json"), result)
        return result, backend, output, config

    def feedback(self, first, retry):
        # A retry adds field-level correction to the original task, rather than
        # presenting the invalid report as an assistant answer to emulate.
        self.assertEqual(retry.messages[:-1], first.messages)
        self.assertEqual(retry.messages[-1]["role"], "user")
        content = retry.messages[-1]["content"]
        self.assertLess(len(content), 4000)
        structured = json.loads(content[content.index("{"):])
        return content, structured

    def test_missing_evidence_stays_required_with_explicit_empty_list_instructions(self):
        schema = CriticReport.model_json_schema()
        self.assertIn("missing_evidence", schema["required"])
        for task, item in ((english_task(), draft()), (history_task(), historical_draft())):
            with self.subTest(mode=task.mode):
                complete = report(task, item["text"])
                incomplete = {key: value for key, value in complete.items() if key != "missing_evidence"}
                with self.assertRaises(ValidationError) as raised:
                    CriticReport.model_validate(incomplete)
                self.assertIn((("missing_evidence",), "missing"),
                              [(error["loc"], error["type"]) for error in raised.exception.errors()])
                self.assertEqual(CriticReport.model_validate(complete).missing_evidence, [])
                instructions = critic_messages(task, item["text"])[0]["content"].split("JSON schema:", 1)[0]
                self.assertTrue(any("missing_evidence" in line and "[]" in line
                                    for line in instructions.splitlines()),
                                "The prose instructions must explicitly require the empty array when applicable")
                for field in schema["required"]:
                    self.assertIn(field, instructions)

    def test_missing_field_retry_completes_reviews_and_accounts_for_failed_tokens(self):
        task, item = english_task(), draft()
        complete = report(task, item["text"])
        incomplete = {key: value for key, value in complete.items() if key != "missing_evidence"}
        incomplete["coverage_note"] = "PREVIOUS_REPORT_CONTENT_MUST_NOT_BE_COPIED"
        result, backend, output, config = self.run_script(task, [
            Reply("draft", item, output_tokens=17),
            Reply("semantic_critic", incomplete, output_tokens=31, metadata={"complete": True}),
            Reply("semantic_critic", complete, output_tokens=43, metadata={"complete": True}),
            Reply("review_draft", review(), output_tokens=11),
        ])
        self.assertEqual(result["status"], "accepted_by_checks", result["error"])
        self.assertEqual([request.role for request in backend.requests],
                         ["draft", "semantic_critic", "semantic_critic", "review_draft"])
        self.assertEqual(result["tokens"]["calls"], 4)
        self.assertEqual(result["tokens"]["input_tokens"], 4 * 32)
        self.assertEqual(result["tokens"]["output_tokens"], 17 + 31 + 43 + 11)
        self.assertTrue(result["tokens"]["complete"])
        self.assertEqual(result["repairs"], 0)
        self.assertEqual(result["final_critic"]["missing_evidence"], [])
        self.assertEqual(result["critic_reviewed_output_sha256"], text_hash(item["text"]))
        self.assertEqual(result["reviewed_output_sha256"], text_hash(item["text"]))
        self.assertIn("W2", [entry["id"] for entry in load_json(output / "state.json")["world"]])
        self.assertEqual([request.max_new_tokens for request in backend.requests[1:3]],
                         [config.critic_tokens, config.critic_tokens])
        content, details = self.feedback(backend.requests[1], backend.requests[2])
        self.assertNotIn("PREVIOUS_REPORT_CONTENT_MUST_NOT_BE_COPIED", json.dumps(backend.requests[2].messages))
        self.assertNotIn("input_value", content)
        self.assertEqual(set(details["required_top_level_fields"]),
                         set(CriticReport.model_json_schema()["required"]))
        missing = next(error for error in details["errors"] if error["path"] == ["missing_evidence"])
        self.assertEqual(missing["type"], "missing")
        self.assertEqual(missing["expected_type"], "array")
        self.assertEqual(missing["item_type"], "string")
        self.assertIn("[]", missing["field_instruction"])
        self.assertIn("missing_evidence", missing["field_instruction"])

    def test_repeated_omission_cannot_approve_draft_or_commit_state(self):
        task, item = english_task(), draft()
        incomplete = report(task, item["text"])
        del incomplete["missing_evidence"]
        result, backend, output, _ = self.run_script(task, [
            Reply("draft", item, output_tokens=17),
            Reply("semantic_critic", incomplete, output_tokens=31, metadata={"complete": True}),
            Reply("semantic_critic", incomplete, output_tokens=47, metadata={"complete": True}),
        ], max_repairs=2)
        self.assertEqual(result["status"], "failed")
        self.assertIn("missing_evidence", result["error"])
        self.assertEqual([request.role for request in backend.requests],
                         ["draft", "semantic_critic", "semantic_critic"])
        self.assertEqual(result["tokens"]["calls"], 3)
        self.assertEqual(result["tokens"]["output_tokens"], 17 + 31 + 47)
        self.assertTrue(result["tokens"]["complete"])
        self.assertEqual(result["repairs"], 0)
        for field in ("final_critic", "final_review", "critic_reviewed_output_sha256", "reviewed_output_sha256"):
            self.assertIsNone(result[field])
        self.assertFalse((output / "critic_00.json").exists())
        self.assertEqual(load_json(output / "state.json"), task.initial_state.model_dump())
        self.assertEqual(load_json(output / "final.json")["text"], item["text"])
        self.assertEqual(result["output_sha256"], text_hash(item["text"]))

    def test_history_retry_can_abstain_without_erasing_missing_evidence(self):
        task, item = history_task(), historical_draft()
        abstention = report(task, item["text"], decision="abstain")
        abstention["missing_evidence"] = ["The supplied report does not establish why the assessment changed."]
        incomplete = {key: value for key, value in abstention.items() if key != "missing_evidence"}
        result, backend, output, _ = self.run_script(task, [
            Reply("draft", item), Reply("semantic_critic", incomplete),
            Reply("semantic_critic", abstention),
        ])
        self.assertEqual(result["status"], "needs_sources", result["error"])
        self.assertEqual(result["final_critic"]["decision"], "abstain")
        self.assertEqual(result["final_critic"]["missing_evidence"], abstention["missing_evidence"])
        self.assertEqual([request.role for request in backend.requests],
                         ["draft", "semantic_critic", "semantic_critic"])
        self.assertEqual(result["tokens"]["output_tokens"], 120)
        self.assertIsNone(result["final_review"])
        self.assertEqual(result["critic_reviewed_output_sha256"], text_hash(item["text"]))
        self.assertEqual(load_json(output / "final.json")["text"], item["text"])
        self.assertEqual(load_json(output / "state.json"), task.initial_state.model_dump())
        for request in backend.requests[1:]:
            self.assertEqual(request.metadata["task"]["sources"], task.model_dump()["sources"])

    def test_saved_failed_run_retries_critic_without_repeating_plan_or_draft(self):
        task, item = english_task(), draft()
        complete = report(task, item["text"])
        incomplete = {key: value for key, value in complete.items() if key != "missing_evidence"}
        parent_result, _, parent, config = self.run_script(task, [
            Reply("plan", valid_plan()), Reply("review_plan", review()), Reply("draft", item),
            Reply("semantic_critic", incomplete), Reply("semantic_critic", incomplete),
        ], name="parent", strategy="single_plan")
        self.assertEqual(parent_result["status"], "failed", parent_result["error"])
        self.assertEqual(parent_result["tokens"]["calls"], 5)
        before = {path.name: path.read_bytes() for path in parent.iterdir()}
        backend = ScriptedBackend([
            Reply("semantic_critic", incomplete), Reply("semantic_critic", complete),
            Reply("review_draft", review()),
        ])
        output = self.root / "recovered"
        result = Pipeline(backend, config, output).run(task, saved_run=load_saved_run(parent))
        self.assertEqual(result["status"], "accepted_by_checks", result["error"])
        self.assertEqual([request.role for request in backend.requests],
                         ["semantic_critic", "semantic_critic", "review_draft"])
        self.assertEqual(result["strategy"], "repair_run")
        self.assertTrue(result["reused_selected_plan"])
        self.assertEqual(result["accepted_plan_count"], 0)
        self.assertEqual(result["repairs"], 0)
        self.assertEqual(result["tokens"]["calls"], 3)
        self.assertEqual(result["tokens"]["output_tokens"], 120)
        self.assertEqual(result["continuation"]["parent_tokens"], parent_result["tokens"])
        self.assertEqual(load_json(output / "input_draft.json"), load_json(parent / "final.json"))
        self.assertEqual(result["output_sha256"], parent_result["output_sha256"])
        self.assertEqual(before, {path.name: path.read_bytes() for path in parent.iterdir()})

    def test_nested_schema_feedback_identifies_path_and_type_without_input_echo(self):
        task, item = english_task(), draft()
        complete = report(task, item["text"])
        malformed = report(task, item["text"])
        bogus = "NEVER_ECHO_THIS_INVALID_CRITERION_INPUT"
        malformed["rubric"][0]["criterion"] = {"private_input": bogus}
        result, backend, _, _ = self.run_script(task, [
            Reply("draft", item), Reply("semantic_critic", malformed),
            Reply("semantic_critic", complete), Reply("review_draft", review()),
        ])
        self.assertEqual(result["status"], "accepted_by_checks", result["error"])
        content, details = self.feedback(backend.requests[1], backend.requests[2])
        self.assertNotIn(bogus, json.dumps(backend.requests[2].messages))
        self.assertNotIn("input_value", content)
        nested = next(error for error in details["errors"] if error["path"] == ["rubric", 0, "criterion"])
        self.assertEqual(nested["type"], "string_type")
        self.assertTrue(nested["message"])
        self.assertNotIn("input", nested)
        self.assertNotIn("ctx", nested)
        self.assertNotIn("url", nested)
        self.assertEqual(result["tokens"]["calls"], 4)
        self.assertEqual(result["tokens"]["output_tokens"], 160)

    def test_many_schema_failures_bound_feedback_without_truncating_task_or_draft(self):
        task = history_task("The assessment rose in 1811.\nSOURCE_PACKET_END_MUST_REMAIN_PRESENT")
        item = historical_draft()
        complete = report(task, item["text"])
        malformed = {key: value for key, value in complete.items() if key != "missing_evidence"}
        # JSON-escaped long keys exercise the serialized-size limit, rather
        # than only the separate maximum number of reported failures.
        for index in range(12):
            malformed[f"unexpected_field_{index}_" + '"' * 300] = {"private_input": "DO_NOT_ECHO_PRIVATE_VALUE"}
        with self.assertRaises(ValidationError) as raised:
            CriticReport.model_validate(malformed)
        total_errors = len(raised.exception.errors())
        self.assertEqual(total_errors, 13)
        result, backend, _, _ = self.run_script(task, [
            Reply("draft", item), Reply("semantic_critic", malformed),
            Reply("semantic_critic", complete), Reply("review_draft", review()),
        ])
        self.assertEqual(result["status"], "accepted_by_checks", result["error"])
        content, details = self.feedback(backend.requests[1], backend.requests[2])
        self.assertLessEqual(len(json.dumps(details, ensure_ascii=False, separators=(",", ":"))), 2400)
        self.assertGreater(len(details["errors"]), 0)
        self.assertLess(len(details["errors"]), 8)
        self.assertEqual(details["omitted_error_count"], total_errors - len(details["errors"]))
        self.assertNotIn("DO_NOT_ECHO_PRIVATE_VALUE", content)
        original_context = json.loads(backend.requests[1].messages[1]["content"])
        retry_context = json.loads(backend.requests[2].messages[1]["content"])
        self.assertEqual(original_context, {"task": task.model_dump(), "prose": item["text"]})
        self.assertEqual(retry_context, original_context)
        self.assertEqual(backend.requests[2].metadata["task"], task.model_dump())

    def test_schema_repair_cannot_bypass_quote_or_history_coverage_integrity(self):
        for mode in ("english", "history"):
            with self.subTest(mode=mode):
                task, item = (english_task(), draft()) if mode == "english" else (history_task(), historical_draft())
                invalid_report = report(task, item["text"])
                if mode == "english":
                    invalid_report["rubric"][0]["draft_quote"] = "A fabricated sentence that is absent from the draft."
                    expected_codes = {"critic_draft_quote_mismatch"}
                else:
                    invalid_report["source_assessments"] = []
                    invalid_report["evidence_findings"] = []
                    expected_codes = {"critic_source_assessment_coverage", "critic_evidence_findings_missing"}
                incomplete = {key: value for key, value in invalid_report.items() if key != "missing_evidence"}
                # Adding the field satisfies the JSON/schema contract; the
                # critic's substantive report integrity must still be checked.
                self.assertEqual(CriticReport.model_validate(invalid_report).missing_evidence, [])
                result, backend, output, _ = self.run_script(task, [
                    Reply("draft", item), Reply("semantic_critic", incomplete),
                    Reply("semantic_critic", invalid_report),
                ], name=f"integrity_{mode}")
                self.assertEqual(result["status"], "needs_review", result["error"])
                self.assertEqual([request.role for request in backend.requests],
                                 ["draft", "semantic_critic", "semantic_critic"])
                self.assertEqual(result["final_critic"]["missing_evidence"], [])
                self.assertTrue(expected_codes <= {issue["code"] for issue in result["critic_validation"]})
                self.assertTrue(expected_codes <= {issue["code"] for issue in result["final_issues"]})
                self.assertIsNone(result["final_review"])
                self.assertIsNone(result["reviewed_output_sha256"])
                self.assertEqual(result["repairs"], 0)
                self.assertEqual(result["tokens"]["calls"], 3)
                self.assertEqual(load_json(output / "state.json"), task.initial_state.model_dump())
                self.assertEqual(load_json(output / "final.json")["text"], item["text"])


if __name__ == "__main__":
    unittest.main()
