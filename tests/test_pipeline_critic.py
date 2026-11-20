"""Adversarial orchestration tests; scripted critics are not quality evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from creativity_pipeline.critic import ENGLISH_CRITERIA, HISTORY_CRITERIA
from creativity_pipeline.io import load_json, text_hash
from creativity_pipeline.pipeline import Pipeline
from test_pipeline import Reply, ScriptedBackend, draft, english_task, history_task, historical_draft, review, settings


def report(task, prose, *, decision="accept", issues=None):
    history = task.mode == "history"
    value = {
        "mode": task.mode, "decision": decision, "coverage": "complete",
        "coverage_note": "Artificial test report covers the short supplied test draft.",
        "rubric": [{"criterion": criterion, "judgment": "adequate", "draft_quote": prose,
                    "explanation": "Artificial rubric evidence for the quoted test passage."}
                   for criterion in (HISTORY_CRITERIA if history else ENGLISH_CRITERIA)],
        "source_assessments": [], "evidence_findings": [], "issues": issues or [],
        "revision_priorities": ["Resolve the named material issue."] if issues else [],
        "preserve": ["Retain the concrete object in the current draft."], "missing_evidence": [],
    }
    if history:
        value["source_assessments"] = [
            {"source_id": source.id, "useful_for": "What this document states.",
             "limitations": "No external verification of truth or independence.", "context_basis": "unknown",
             "context_quotes": [], "related_source_ids": [], "independence": "unknown"}
            for source in task.sources]
        value["evidence_findings"] = [{"draft_quote": prose, "claim_kind": "source_assertion",
                                       "support_level": "supported_as_attributed", "evidence": [
                                           {"source_id": task.sources[0].id, "quote": task.sources[0].text, "relation": "supports"}],
                                       "explanation": "The draft attributes this statement to its document."}]
    return value


class CriticPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def run_script(self, replies, *, task=None, **changes):
        backend = ScriptedBackend(replies)
        config = settings(semantic_critic=True, critic_tokens=1024, **changes)
        pipeline = Pipeline(backend, config, self.root / "run")
        result = pipeline.run(task or english_task())
        return result, backend

    def test_acceptance_requires_critic_and_general_review_on_same_text(self):
        task, item = english_task(), draft()
        result, backend = self.run_script([
            Reply("draft", item), Reply("semantic_critic", report(task, item["text"])), Reply("review_draft", review())])
        self.assertEqual(result["status"], "accepted_by_checks", result["error"])
        self.assertEqual(result["critic_reviewed_output_sha256"], text_hash(item["text"]))
        self.assertEqual(result["reviewed_output_sha256"], result["critic_reviewed_output_sha256"])
        self.assertEqual(len(backend.requests), 3)
        self.assertEqual(result["tokens"]["output_tokens"], 120)

    def test_bad_history_annotations_are_repaired_before_any_semantic_review(self):
        task, final = history_task(), historical_draft()
        first = json.loads(json.dumps(final))
        first["text"] += " An unsupported addition. [S6]"
        first["claims"][0]["text"] = "This wording never appears in the draft."
        first["claims"][0]["support_quotes"] = [{"source_id": "S6", "quote": "An invented quotation."}]
        result, backend = self.run_script([
            Reply("draft", first), Reply("repair", final),
            Reply("semantic_critic", report(task, final["text"])), Reply("review_draft", review()),
        ], task=task)
        self.assertEqual(result["status"], "accepted_by_checks", result["error"])
        self.assertEqual([request.role for request in backend.requests],
                         ["draft", "repair", "semantic_critic", "review_draft"])
        self.assertEqual(result["repairs"], 1)
        repair = backend.requests[1]
        self.assertTrue({"claim_text_missing", "quote_source_not_cited", "unknown_quote_source"}
                        <= {issue["code"] for issue in repair.metadata["defects"]})
        self.assertEqual(backend.requests[2].metadata["draft"], {"text": final["text"]})
        self.assertEqual(result["critic_reviewed_output_sha256"], text_hash(final["text"]))
        self.assertFalse((self.root / "run/critic_00.json").exists())
        self.assertTrue((self.root / "run/critic_01.json").exists())
        events = [json.loads(line) for line in (self.root / "run/events.jsonl").read_text().splitlines()]
        skipped = next(event for event in events if event["kind"] == "reviews_skipped")
        self.assertEqual(skipped["reason"], "material_mechanical_defects")
        self.assertTrue(skipped["repair_available"])
        self.assertEqual(skipped["output_sha256"], text_hash(first["text"]))
        self.assertEqual([event["reason"] for event in events if event["kind"] == "repair_requested"],
                         ["mechanical_checks"])

    def test_repeated_mechanical_failures_use_only_the_bounded_repair_budget(self):
        task, invalid = english_task(), draft("W1", "Mara had two keys.")
        result, backend = self.run_script([
            Reply("draft", invalid), Reply("repair", invalid), Reply("repair", invalid),
        ], task=task, max_repairs=2)
        self.assertEqual(result["status"], "needs_review", result["error"])
        self.assertIn("repair limit is exhausted", result["error"])
        self.assertEqual([request.role for request in backend.requests], ["draft", "repair", "repair"])
        self.assertEqual(result["repairs"], 2)
        self.assertEqual(result["tokens"]["calls"], 3)
        self.assertIn("state_overwrite", {issue["code"] for issue in result["final_issues"]})
        self.assertIsNone(result["final_critic"])
        self.assertIsNone(result["final_review"])
        self.assertEqual(load_json(self.root / "run/final.json")["text"], invalid["text"])
        self.assertEqual(load_json(self.root / "run/state.json"), task.initial_state.model_dump())

    def test_zero_repair_limit_does_not_spend_a_critic_call_on_mechanical_failure(self):
        invalid = draft("W1", "Mara had two keys.")
        result, backend = self.run_script([Reply("draft", invalid)], max_repairs=0)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual([request.role for request in backend.requests], ["draft"])
        self.assertEqual(result["repairs"], 0)
        self.assertIsNone(result["final_critic"])
        self.assertIsNone(result["reviewed_output_sha256"])

    def test_minor_mechanical_findings_still_receive_semantic_reviews(self):
        task, item = english_task(), draft()
        result, backend = self.run_script([
            Reply("draft", item), Reply("semantic_critic", report(task, item["text"])),
            Reply("review_draft", review()),
        ], task=task)
        self.assertIn("target_length_deviation", {issue["code"] for issue in result["final_issues"]})
        self.assertEqual(result["status"], "accepted_by_checks", result["error"])
        self.assertEqual([request.role for request in backend.requests], ["draft", "semantic_critic", "review_draft"])

    def test_mechanical_and_editorial_repairs_share_the_same_limit(self):
        task, invalid, final = english_task(), draft("W1", "Mara had two keys."), draft()
        issue = {"code": "motivation_gap", "message": "This action needs a motive.",
                 "severity": "material", "location": final["text"]}
        result, backend = self.run_script([
            Reply("draft", invalid), Reply("repair", final),
            Reply("semantic_critic", report(task, final["text"], decision="revise", issues=[issue])),
            Reply("review_draft", review()),
        ], task=task, max_repairs=1)
        self.assertEqual(result["status"], "needs_review", result["error"])
        self.assertEqual(result["repairs"], 1)
        self.assertEqual([request.role for request in backend.requests].count("repair"), 1)
        self.assertIn(issue, result["final_issues"])
        self.assertEqual(load_json(self.root / "run/state.json"), task.initial_state.model_dump())

    def test_mechanically_invalid_editorial_repair_cannot_inherit_old_reviews(self):
        task, first, invalid = english_task(), draft(), draft("W1", "Mara had two keys.")
        result, backend = self.run_script([
            Reply("draft", first), Reply("semantic_critic", report(task, first["text"])),
            Reply("review_draft", review("revise")), Reply("repair", invalid),
        ], task=task, max_repairs=1)
        self.assertEqual(result["status"], "needs_review", result["error"])
        self.assertEqual(result["output_sha256"], text_hash(invalid["text"]))
        self.assertIsNone(result["final_critic"])
        self.assertIsNone(result["final_review"])
        self.assertIsNone(result["critic_reviewed_output_sha256"])
        self.assertIsNone(result["reviewed_output_sha256"])
        self.assertEqual([request.role for request in backend.requests],
                         ["draft", "semantic_critic", "review_draft", "repair"])
        self.assertEqual(load_json(self.root / "run/state.json"), task.initial_state.model_dump())

    def test_critic_material_issue_cannot_be_overridden_by_general_accept(self):
        task, item = english_task(), draft()
        issue = {"code": "motivation_gap", "message": "The locked door conflicts with the established purpose.",
                 "severity": "material", "location": item["text"]}
        result, _ = self.run_script([
            Reply("draft", item), Reply("semantic_critic", report(task, item["text"], decision="revise", issues=[issue])),
            Reply("review_draft", review())], max_repairs=0)
        self.assertEqual(result["status"], "needs_review", result["error"])
        self.assertIn(issue, result["final_issues"])
        self.assertEqual(load_json(self.root / "run/state.json"), task.initial_state.model_dump())

    def test_invalid_critic_quote_is_not_sent_to_writer_for_repair(self):
        task, item = english_task(), draft()
        critic = report(task, item["text"])
        critic["rubric"][0]["draft_quote"] = "A fabricated line that was never written."
        result, backend = self.run_script([Reply("draft", item), Reply("semantic_critic", critic)])
        self.assertEqual(result["status"], "needs_review")
        self.assertIn("critic's own report failed", result["error"])
        self.assertEqual([request.role for request in backend.requests], ["draft", "semantic_critic"])
        self.assertEqual(load_json(self.root / "run/state.json"), task.initial_state.model_dump())

    def test_repaired_text_gets_fresh_critic_and_preserve_guidance(self):
        task, first, final = english_task(), draft("discarded", "Mara put down the key."), draft()
        issue = {"code": "development_gap", "message": "The choice needs a consequence.", "severity": "material", "location": first["text"]}
        result, backend = self.run_script([
            Reply("draft", first), Reply("semantic_critic", report(task, first["text"], decision="revise", issues=[issue])),
            Reply("review_draft", review()), Reply("repair", final),
            Reply("semantic_critic", report(task, final["text"])), Reply("review_draft", review())])
        self.assertEqual(result["status"], "accepted_by_checks", result["error"])
        self.assertEqual(result["critic_reviewed_output_sha256"], text_hash(final["text"]))
        repair = next(request for request in backend.requests if request.role == "repair")
        self.assertIn("Retain the concrete object", json.dumps(repair.metadata["defects"]))
        self.assertNotIn("discarded", [entry["id"] for entry in load_json(self.root / "run/state.json")["world"]])
        self.assertTrue((self.root / "run/critic_00.json").exists())
        self.assertTrue((self.root / "run/critic_01.json").exists())

    def test_failed_critic_of_repair_cannot_inherit_old_approval(self):
        task, first, final = english_task(), draft("discarded", "Mara put down the key."), draft()
        result, _ = self.run_script([
            Reply("draft", first), Reply("semantic_critic", report(task, first["text"])),
            Reply("review_draft", review("revise")), Reply("repair", final),
            Reply("semantic_critic", "not JSON")])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["output_sha256"], text_hash(final["text"]))
        self.assertIsNone(result["critic_reviewed_output_sha256"])
        self.assertIsNone(result["final_critic"])
        self.assertIsNone(result["reviewed_output_sha256"])

    def test_history_critic_sees_only_prose_and_source_context_and_replaces_legacy_audit(self):
        task, item = history_task(), historical_draft()
        item["claims"][0]["id"] = "HIDDEN_ANNOTATION_ID"
        result, backend = self.run_script([
            Reply("draft", item), Reply("semantic_critic", report(task, item["text"])), Reply("review_draft", review())], task=task)
        self.assertEqual(result["status"], "accepted_by_checks", result["error"])
        request = backend.requests[1]
        self.assertEqual(request.metadata["draft"], {"text": item["text"]})
        self.assertFalse(request.metadata["use_adapter"])
        self.assertNotIn("HIDDEN_ANNOTATION_ID", json.dumps(request.messages))
        self.assertIn(task.sources[0].text, json.dumps(request.messages))
        self.assertNotIn("review_evidence", [request.role for request in backend.requests])

    def test_history_abstention_preserves_draft_and_requests_sources(self):
        task, item = history_task(), historical_draft()
        critic = report(task, item["text"], decision="abstain")
        critic["missing_evidence"] = ["The packet does not establish the proposed cause."]
        result, backend = self.run_script([Reply("draft", item), Reply("semantic_critic", critic)], task=task)
        self.assertEqual(result["status"], "needs_sources", result["error"])
        self.assertEqual(len(backend.requests), 2)
        self.assertTrue((self.root / "run/final.txt").exists())

    def test_standalone_assessment_preserves_input_without_misattributing_generation(self):
        task = english_task()
        prose = "  Mara put down the key.\n"
        backend = ScriptedBackend([Reply("semantic_critic", report(task, "Mara put down the key."))])
        config = settings(semantic_critic=True, critic_tokens=1024)
        output = self.root / "assessment"
        result = Pipeline(backend, config, output).assess(task, prose)
        self.assertEqual(result["status"], "valid_report", result.get("error"))
        self.assertEqual(result["input_sha256"], text_hash(prose))
        self.assertEqual(load_json(output / "input_draft.json")["text"], prose)
        self.assertFalse((output / "summary.json").exists())
        self.assertFalse((output / "final.txt").exists())
        self.assertFalse((output / "state.json").exists())


if __name__ == "__main__":
    unittest.main()
