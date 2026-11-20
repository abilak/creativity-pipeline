"""Offline integration tests for run integrity, review gates, and budget bounds."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest

from creativity_pipeline.demo import DemoBackend
from creativity_pipeline.interfaces import GenerationResult
from creativity_pipeline.io import load_json, text_hash
from creativity_pipeline.models import Settings, Task
from creativity_pipeline.pipeline import Pipeline


@dataclass
class Reply:
    role: str
    value: dict | str | Exception
    output_tokens: int = 40
    finish_reason: str = "stop"
    metadata: dict | None = None


class ScriptedBackend:
    """A deterministic response script with explicitly artificial token counts."""

    def __init__(self, replies, prompt_tokens=32):
        self.replies = list(replies)
        self.prompt_tokens = prompt_tokens
        self.requests = []
        self.counted_requests = []

    def describe(self):
        return {"backend": "test_script", "fixture": True}

    def count_input_tokens(self, request):
        self.counted_requests.append(request)
        return self.prompt_tokens

    def generate(self, request):
        self.requests.append(request)
        if not self.replies:
            raise AssertionError(f"Unexpected extra generation: {request.role}")
        reply = self.replies.pop(0)
        if reply.role != request.role:
            raise AssertionError(f"Expected {reply.role}, received {request.role}")
        if isinstance(reply.value, Exception):
            raise reply.value
        text = json.dumps(reply.value) if isinstance(reply.value, dict) else reply.value
        return GenerationResult(
            text=text, input_tokens=self.prompt_tokens, output_tokens=reply.output_tokens,
            finish_reason=reply.finish_reason, accounting="test_fixture_estimate",
            metadata=reply.metadata or {},
        )


def english_task():
    return Task.model_validate({
        "task_id": "english_test", "mode": "english", "prompt": "Continue Mara's story.",
        "target_words": 100,
        "initial_state": {
            "world": [{"id": "W1", "text": "Mara has one key.", "evidence": ""}],
            "reader": [],
        },
    })


def history_task(source_text="The assessment rose in 1811."):
    return Task.model_validate({
        "task_id": "history_test", "mode": "history", "prompt": "Assess this fictional source packet.",
        "target_words": 100,
        "sources": [{"id": "S1", "title": "Fictional report", "text": source_text}],
    })


def draft(identifier="W2", text="Mara locked the door."):
    return {
        "text": text, "claims": [],
        "state_updates": {
            "world": [{"id": identifier, "text": text, "evidence": text}],
            "reader": [],
        },
    }


def valid_plan(identifier="P1", *, history=False, concept="A practical discovery"):
    move = {
        "id": "M1", "description": "The assessment rose in 1811." if history else concept,
        "claim_status": "attested" if history else "fiction", "depends_on": [],
    }
    if history:
        move.update(source_ids=["S1"], support_quotes=[{"source_id": "S1", "quote": "The assessment rose in 1811."}])
    return {"id": identifier, "approach": concept, "intent": "Develop the consequence.",
            "moves": [move], "presentation": ["M1"], "open_options": []}


def historical_draft(*, ending="", claim_id="C1"):
    sentence = 'The supplied report states: "The assessment rose in 1811." [S1].'
    return {
        "text": sentence + ending,
        "claims": [{"id": claim_id, "text": sentence, "source_ids": ["S1"],
                    "support_quotes": [{"source_id": "S1", "quote": "The assessment rose in 1811."}],
                    "claim_status": "attested"}],
        "state_updates": {"world": [], "reader": []},
    }


def review(decision="accept", code="pacing"):
    return {
        "decision": decision,
        "issues": [] if decision == "accept" else [{
            "code": code, "message": "The last action needs a clearer consequence.",
            "severity": "minor", "location": "ending",
        }],
        "strengths": [],
    }


def settings(**changes):
    values = {
        "backend": "demo", "strategy": "direct", "max_repairs": 1, "semantic_critic": False,
        "schema_retries": 0, "max_context_tokens": 8192,
        "max_run_tokens": 60000, "max_output_tokens": 16000,
        "draft_tokens": 512, "review_tokens": 128,
    }
    values.update(changes)
    return Settings.model_validate(values)


class PipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def run_script(self, replies, *, task=None, config=None, name="run", prompt_tokens=32):
        backend = ScriptedBackend(replies, prompt_tokens=prompt_tokens)
        output = self.root / name
        pipeline = Pipeline(backend, config or settings(), output)
        summary = pipeline.run(task or english_task())
        self.assertEqual(load_json(output / "summary.json"), summary)
        return summary, backend, output

    def events(self, output):
        path = output / "events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_final_repair_is_reviewed_and_only_accepted_state_is_committed(self):
        first = draft("discarded", "Mara placed the key beside the window.")
        final = draft("accepted", "Mara locked the door.")
        summary, backend, output = self.run_script([
            Reply("draft", first), Reply("review_draft", review("revise")),
            Reply("repair", final), Reply("review_draft", review()),
        ])
        self.assertEqual(summary["status"], "accepted_by_checks")
        self.assertEqual([r.role for r in backend.requests], ["draft", "review_draft", "repair", "review_draft"])
        self.assertEqual(backend.requests[-1].metadata["draft"]["text"], final["text"])
        self.assertFalse(backend.requests[-1].metadata["use_adapter"])
        self.assertTrue(backend.requests[-2].metadata["use_adapter"])
        self.assertEqual((output / "final.txt").read_text(), final["text"])
        self.assertEqual(summary["draft_revision"], 1)
        self.assertEqual(summary["reviewed_output_sha256"], text_hash(final["text"]))
        self.assertEqual(summary["reviewed_output_sha256"], summary["output_sha256"])
        state = load_json(output / "state.json")
        self.assertEqual([entry["id"] for entry in state["world"]], ["W1", "accepted"])
        self.assertEqual(len([e for e in self.events(output) if e["kind"] == "draft_review"]), 2)
        self.assertFalse(backend.replies)

    def test_rejected_final_repair_does_not_commit_any_proposed_state(self):
        task = english_task()
        summary, backend, output = self.run_script([
            Reply("draft", draft("discarded")), Reply("review_draft", review("revise")),
            Reply("repair", draft("still_unaccepted")), Reply("review_draft", review("reject")),
        ], task=task)
        self.assertEqual(summary["status"], "needs_review")
        self.assertEqual(summary["final_review"]["decision"], "reject")
        self.assertEqual(load_json(output / "state.json"), task.initial_state.model_dump())
        self.assertEqual([r.role for r in backend.requests].count("review_draft"), 2)

    def test_accept_cannot_override_material_mechanical_findings(self):
        task = english_task()
        summary, _, output = self.run_script([
            Reply("draft", draft("W1", "Mara had two keys.")),
            Reply("review_draft", review()),
        ], task=task, config=settings(max_repairs=0))
        self.assertEqual(summary["status"], "needs_review")
        self.assertEqual(summary["final_review"]["decision"], "accept")
        findings = [i for i in summary["final_issues"] if i["code"] == "state_overwrite"]
        self.assertEqual(len(findings), 1, "A single mechanical finding must not be duplicated by the runner")
        self.assertEqual(findings[0]["severity"], "material")
        self.assertEqual(load_json(output / "state.json"), task.initial_state.model_dump())

    def test_material_plan_failure_is_rejected_before_model_acceptance(self):
        invalid_plan = {
            "id": "P1", "approach": "Follow a consequence", "intent": "A coherent story",
            "moves": [{"id": "M1", "description": "Mara locks the door.",
                       "claim_status": "fiction", "depends_on": ["missing_move"]}],
            "presentation": ["M1"], "open_options": [],
        }
        summary, backend, output = self.run_script([
            Reply("plan", invalid_plan),
        ], config=settings(strategy="single_plan"))
        self.assertEqual(summary["status"], "failed")
        self.assertEqual([r.role for r in backend.requests], ["plan"])
        self.assertEqual(load_json(output / "accepted_plans.json"), [])
        self.assertFalse((output / "final.txt").exists())
        rejected = [e for e in self.events(output) if e["kind"] == "plan_rejected"]
        self.assertEqual(rejected[0]["reason"], "deterministic_checks")

    def test_accepted_candidate_survives_malformed_later_plan_or_review(self):
        for failed_stage in ("plan", "review_plan"):
            with self.subTest(failed_stage=failed_stage):
                replies = [Reply("plan", valid_plan()), Reply("review_plan", review())]
                if failed_stage == "review_plan":
                    replies.append(Reply("plan", valid_plan("P2", concept="A different action")))
                replies.extend([
                    Reply(failed_stage, "not valid JSON", output_tokens=17),
                    Reply("draft", draft()), Reply("review_draft", review()),
                ])
                summary, backend, output = self.run_script(
                    replies, config=settings(strategy="multi_plan", candidates=2, max_repairs=0),
                    name=f"candidate_{failed_stage}",
                )
                self.assertEqual(summary["status"], "accepted_by_checks", summary.get("error"))
                self.assertEqual(summary["accepted_plan_count"], 1)
                self.assertEqual(load_json(output / "selected_plan.json")["id"], "P1")
                self.assertEqual([row["plan"]["id"] for row in load_json(output / "accepted_plans.json")], ["P1"])
                failures = [event for event in self.events(output) if event["kind"] == "candidate_format_failure"]
                self.assertEqual(len(failures), 1)
                self.assertEqual((failures[0]["plan_id"], failures[0]["stage"]), ("P2", failed_stage))
                self.assertEqual(summary["tokens"]["output_tokens"], 40 * (len(replies) - 1) + 17)
                self.assertFalse(backend.replies)

    def test_backend_failure_is_fatal_but_preserves_accepted_candidates(self):
        summary, backend, output = self.run_script([
            Reply("plan", valid_plan()), Reply("review_plan", review()),
            Reply("plan", RuntimeError("simulated GPU failure")),
        ], config=settings(strategy="multi_plan", candidates=2))
        self.assertEqual(summary["status"], "failed")
        self.assertIn("simulated GPU failure", summary["error"])
        self.assertEqual([row["plan"]["id"] for row in load_json(output / "accepted_plans.json")], ["P1"])
        self.assertEqual([request.role for request in backend.requests], ["plan", "review_plan", "plan"])
        self.assertEqual(summary["tokens"]["unaccounted_calls"], 1)
        self.assertFalse((output / "final.txt").exists())
        self.assertFalse(any(event["kind"] == "candidate_format_failure" for event in self.events(output)))

    def test_candidate_budget_exhaustion_is_not_a_recoverable_format_error(self):
        summary, backend, output = self.run_script([
            Reply("plan", valid_plan()), Reply("review_plan", review()),
        ], config=settings(strategy="multi_plan", candidates=2, max_calls=2))
        self.assertEqual(summary["status"], "budget_exhausted")
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(load_json(output / "accepted_plans.json")[0]["plan"]["id"], "P1")
        self.assertFalse((output / "final.txt").exists())

    def test_history_audit_sees_prose_and_sources_without_writer_judgments(self):
        candidate = valid_plan(history=True)
        candidate["intent"] = "WRITER_PLAN_SENTINEL"
        previous_review = review()
        previous_review["strengths"] = ["PREVIOUS_REVIEW_SENTINEL"]
        final = historical_draft(claim_id="WRITER_CLAIM_SENTINEL")
        task = history_task()
        summary, backend, _ = self.run_script([
            Reply("plan", candidate), Reply("review_plan", previous_review),
            Reply("draft", final), Reply("review_evidence", review()),
            Reply("review_draft", review()),
        ], task=task, config=settings(strategy="single_plan", max_repairs=0))
        self.assertEqual(summary["status"], "accepted_by_checks", summary.get("error"))
        audit = next(request for request in backend.requests if request.role == "review_evidence")
        rendered = "\n".join(message["content"] for message in audit.messages)
        for sentinel in ("WRITER_PLAN_SENTINEL", "PREVIOUS_REVIEW_SENTINEL", "WRITER_CLAIM_SENTINEL"):
            self.assertNotIn(sentinel, rendered)
        self.assertEqual(audit.metadata["draft"], {"text": final["text"]})
        self.assertNotIn("plan", audit.metadata)
        self.assertFalse(audit.metadata["use_adapter"])
        payload = json.loads(audit.messages[-1]["content"].split("\nStage inputs:\n", 1)[1])
        self.assertEqual(payload, {"prose": final["text"]})
        self.assertIn(task.sources[0].text, rendered)
        self.assertEqual(summary["evidence_reviewed_output_sha256"], summary["output_sha256"])

    def test_history_audit_rejection_cannot_be_overridden_by_general_accept(self):
        for with_issue in (False, True):
            with self.subTest(with_issue=with_issue):
                rejection = {"decision": "reject", "issues": [], "strengths": []}
                if with_issue:
                    rejection["issues"] = [{"code": "unsupported_claim", "message": "The conclusion is not supported by S1.", "severity": "material"}]
                task = history_task()
                summary, _, output = self.run_script([
                    Reply("draft", historical_draft()), Reply("review_evidence", rejection),
                    Reply("review_draft", review()),
                ], task=task, config=settings(max_repairs=0), name=f"audit_reject_{with_issue}")
                self.assertEqual(summary["status"], "needs_review")
                self.assertEqual(summary["final_review"]["decision"], "accept")
                self.assertEqual(summary["final_evidence_review"]["decision"], "reject")
                self.assertEqual(load_json(output / "state.json"), task.initial_state.model_dump())

    def test_history_evidence_audit_is_rerun_after_repair(self):
        first = historical_draft(ending=" The increase certainly caused every protest.")
        final = historical_draft(ending=" This report alone cannot establish the cause.")
        rejection = {"decision": "revise", "issues": [{
            "code": "causal_overreach", "message": "'certainly caused every protest' is not established by S1.",
            "severity": "material",
        }], "strengths": []}
        summary, backend, output = self.run_script([
            Reply("draft", first), Reply("review_evidence", rejection), Reply("review_draft", review()),
            Reply("repair", final), Reply("review_evidence", review()), Reply("review_draft", review()),
        ], task=history_task())
        self.assertEqual(summary["status"], "accepted_by_checks", summary.get("error"))
        audits = [request for request in backend.requests if request.role == "review_evidence"]
        self.assertEqual([request.metadata["draft"]["text"] for request in audits], [first["text"], final["text"]])
        self.assertTrue(all(not request.metadata["use_adapter"] for request in audits))
        repair = next(request for request in backend.requests if request.role == "repair")
        self.assertIn("causal_overreach", [issue["code"] for issue in repair.metadata["defects"]])
        self.assertEqual(summary["evidence_reviewed_output_sha256"], text_hash(final["text"]))
        evidence_events = [event for event in self.events(output) if event["kind"] == "evidence_review"]
        self.assertEqual([event["output_sha256"] for event in evidence_events], [text_hash(first["text"]), text_hash(final["text"])])

    def test_failed_audit_after_repair_cannot_inherit_previous_review_hashes(self):
        task = Task.model_validate({
            **history_task().model_dump(),
            "initial_state": {"world": [{"id": "existing_context", "text": "Preserve this supplied historical context."}], "reader": []},
        })
        first = historical_draft()
        final = historical_draft(ending=" A broader conclusion remains unresolved.")
        summary, backend, output = self.run_script([
            Reply("draft", first), Reply("review_evidence", review()),
            Reply("review_draft", review("revise", code="old_review_issue")),
            Reply("repair", final), Reply("review_evidence", RuntimeError("new audit failed")),
        ], task=task)
        self.assertEqual(summary["status"], "failed")
        for field in ("final_review", "reviewed_output_sha256", "final_evidence_review", "evidence_reviewed_output_sha256"):
            self.assertIsNone(summary[field], field)
        self.assertEqual(summary["output_sha256"], text_hash(final["text"]))
        self.assertNotIn("old_review_issue", [issue["code"] for issue in summary["final_issues"]])
        self.assertEqual(load_json(output / "state.json"), task.initial_state.model_dump())
        self.assertEqual([request.role for request in backend.requests].count("review_draft"), 1)
        self.assertFalse(summary["tokens"]["complete"])

    def test_history_audit_can_be_disabled_only_by_explicit_configuration(self):
        summary, backend, output = self.run_script([
            Reply("draft", historical_draft()), Reply("review_draft", review()),
        ], task=history_task(), config=settings(history_evidence_audit=False, max_repairs=0))
        self.assertEqual(summary["status"], "accepted_by_checks")
        self.assertIsNone(summary["final_evidence_review"])
        self.assertNotIn("review_evidence", [request.role for request in backend.requests])
        self.assertFalse(load_json(output / "config.json")["history_evidence_audit"])

    def test_selector_gets_aliased_shuffled_plans_without_prior_praise(self):
        replies = []
        for index in range(1, 4):
            judgment = review()
            judgment["strengths"] = [f"PRIOR_PRAISE_SENTINEL_{index}"]
            replies.extend([Reply("plan", valid_plan(f"P{index}", concept=f"Concept {index}")), Reply("review_plan", judgment)])
        replies.extend([Reply("select", {"plan_id": "C1", "reason": "Choose the first shown option."}),
                        Reply("draft", draft()), Reply("review_draft", review())])
        summary, backend, output = self.run_script(replies, config=settings(strategy="multi_plan", candidates=3, max_repairs=0))
        self.assertEqual(summary["status"], "accepted_by_checks", summary.get("error"))
        selection = next(request for request in backend.requests if request.role == "select")
        shown = selection.metadata["candidates"]
        self.assertEqual([row["plan"]["id"] for row in shown], ["C1", "C2", "C3"])
        self.assertTrue(all(set(row) == {"plan"} for row in shown))
        rendered = "\n".join(message["content"] for message in selection.messages)
        self.assertNotIn("PRIOR_PRAISE_SENTINEL", rendered)
        for original_id in ("P1", "P2", "P3"):
            self.assertNotIn(f'"id": "{original_id}"', rendered)
        selection_event = next(event for event in self.events(output) if event["kind"] == "selection_context")
        aliases = selection_event["aliases"]
        self.assertNotEqual(list(aliases.values()), ["P1", "P2", "P3"], "The known test seed must actually permute these candidates")
        self.assertEqual(load_json(output / "selected_plan.json")["id"], aliases["C1"])
        for row in shown:
            original = aliases[row["plan"]["id"]]
            self.assertEqual(row["plan"]["approach"], f"Concept {original[1:]}")

    def test_uniform_accepted_selection_is_seeded_and_does_not_call_a_judge(self):
        selected = []
        for run in range(2):
            replies = []
            for index in range(1, 4):
                replies.extend([Reply("plan", valid_plan(f"P{index}", concept=f"Concept {index}")), Reply("review_plan", review())])
            replies.extend([Reply("draft", draft()), Reply("review_draft", review())])
            summary, backend, output = self.run_script(
                replies, config=settings(strategy="multi_plan", candidates=3, selection_policy="sample", max_repairs=0),
                name=f"sample_{run}",
            )
            self.assertEqual(summary["status"], "accepted_by_checks", summary.get("error"))
            self.assertNotIn("select", [request.role for request in backend.requests])
            selected.append(load_json(output / "selected_plan.json")["id"])
            policy = next(event for event in self.events(output) if event["kind"] == "selection_policy")
            self.assertEqual(policy["policy"], "sample")
            self.assertIsInstance(policy["seed"], int)
        self.assertEqual(selected[0], selected[1])
        self.assertIn(selected[0], {"P1", "P2", "P3"})

    def test_explicit_length_minimum_cannot_be_overridden_by_accept(self):
        task = Task.model_validate({**english_task().model_dump(), "min_words": 80})
        summary, _, output = self.run_script([
            Reply("draft", draft()), Reply("review_draft", review()),
        ], task=task, config=settings(max_repairs=0))
        self.assertEqual(summary["word_count"], 4)
        self.assertEqual(summary["status"], "needs_review")
        self.assertEqual(summary["final_review"]["decision"], "accept")
        self.assertTrue(any(issue["code"] == "below_min_words" and issue["severity"] == "material" for issue in summary["final_issues"]))
        self.assertEqual(load_json(output / "state.json"), task.initial_state.model_dump())

    def test_role_specific_sampling_settings_reach_the_backend(self):
        summary, backend, _ = self.run_script([
            Reply("plan", valid_plan()), Reply("review_plan", review()),
            Reply("draft", draft()), Reply("review_draft", review()),
        ], config=settings(strategy="single_plan", thinking_plan=True, thinking_review=True,
                           thinking_top_p=0.91, nonthinking_top_p=0.63, top_k=7, min_p=0.1))
        self.assertEqual(summary["status"], "accepted_by_checks", summary.get("error"))
        for request in backend.requests:
            self.assertEqual(request.top_k, 7)
            self.assertEqual(request.min_p, 0.1)
            self.assertEqual(request.top_p, 0.91 if request.thinking else 0.63)
        self.assertFalse(next(request for request in backend.requests if request.role == "draft").thinking)
        self.assertTrue(next(request for request in backend.requests if request.role == "plan").thinking)

    def test_backend_failure_preserves_initial_state_and_marks_accounting_unknown(self):
        task = english_task()
        summary, _, output = self.run_script([
            Reply("draft", draft("uncommitted")),
            Reply("review_draft", RuntimeError("simulated device failure")),
        ], task=task)
        self.assertEqual(summary["status"], "failed")
        self.assertIn("simulated device failure", summary["error"])
        self.assertEqual(load_json(output / "state.json"), task.initial_state.model_dump())
        self.assertEqual(summary["tokens"]["calls"], 2)
        self.assertEqual(summary["tokens"]["unaccounted_calls"], 1)
        self.assertFalse(summary["tokens"]["complete"])

    def test_failed_review_of_repair_cannot_inherit_previous_review(self):
        task = english_task()
        final = draft("uncommitted_final", "Mara locked the cupboard.")
        summary, _, output = self.run_script([
            Reply("draft", draft("discarded")),
            Reply("review_draft", review("revise", code="old_draft_problem")),
            Reply("repair", final),
            Reply("review_draft", RuntimeError("review did not complete")),
        ], task=task)
        self.assertEqual(summary["status"], "failed")
        self.assertIsNone(summary["final_review"])
        self.assertIsNone(summary["reviewed_output_sha256"])
        self.assertEqual(summary["output_sha256"], text_hash(final["text"]))
        self.assertEqual((output / "final.txt").read_text(), final["text"])
        self.assertNotIn("old_draft_problem", [i["code"] for i in summary["final_issues"]])
        self.assertEqual(load_json(output / "state.json"), task.initial_state.model_dump())

    def test_schema_retry_limit_is_bounded_and_all_malformed_tokens_are_counted(self):
        for retries in (0, 2):
            with self.subTest(retries=retries):
                attempts = retries + 1
                summary, backend, output = self.run_script(
                    [Reply("draft", '{"text":', output_tokens=17) for _ in range(attempts)],
                    config=settings(schema_retries=retries), name=f"retry_{retries}",
                )
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(len(backend.requests), attempts)
                self.assertEqual(summary["tokens"]["calls"], attempts)
                self.assertEqual(summary["tokens"]["output_tokens"], 17 * attempts)
                self.assertEqual(summary["tokens"]["input_tokens"], 32 * attempts)
                self.assertEqual(len([e for e in self.events(output) if e["kind"] == "schema_error"]), attempts)
                self.assertFalse((output / "final.json").exists())
                if retries:
                    self.assertIn("previous attempt failed validation", backend.requests[-1].messages[-1]["content"])

    def test_successful_schema_retry_keeps_failed_attempt_in_ledger(self):
        summary, backend, _ = self.run_script([
            Reply("draft", "not JSON", output_tokens=17),
            Reply("draft", draft(), output_tokens=31),
            Reply("review_draft", review(), output_tokens=13),
        ], config=settings(schema_retries=1))
        self.assertEqual(summary["status"], "accepted_by_checks")
        self.assertEqual(summary["tokens"]["output_tokens"], 61)
        self.assertEqual(summary["tokens"]["calls"], 3)
        self.assertEqual([r.seed for r in backend.requests], [42, 43, 44])

    def test_malformed_generations_consume_budget_before_next_retry(self):
        summary, backend, output = self.run_script([
            Reply("draft", "bad JSON", output_tokens=100),
            Reply("draft", "bad JSON again", output_tokens=100),
            Reply("draft", draft()),
        ], config=settings(schema_retries=2, draft_tokens=128, max_output_tokens=256))
        self.assertEqual(summary["status"], "budget_exhausted")
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(summary["tokens"]["output_tokens"], 200)
        self.assertEqual(summary["tokens"]["input_tokens"], 64)
        self.assertEqual(summary["tokens"]["calls"], 2)
        self.assertEqual(len(backend.replies), 1)
        self.assertFalse((output / "final.txt").exists())

    def test_context_overflow_never_truncates_evidence_or_calls_model(self):
        source = "SOURCE_BEGIN " + "unchanged evidence " * 100 + " SOURCE_END"
        task = history_task(source)
        summary, backend, output = self.run_script([], task=task, prompt_tokens=600,
            config=settings(max_context_tokens=512, draft_tokens=128))
        self.assertEqual(summary["status"], "failed")
        self.assertIn("will not silently truncate evidence", summary["error"])
        self.assertEqual(backend.requests, [])
        self.assertEqual(len(backend.counted_requests), 1)
        rendered = "\n".join(m["content"] for m in backend.counted_requests[0].messages)
        self.assertIn(source, rendered)
        self.assertEqual(load_json(output / "task.json")["sources"][0]["text"], source)
        self.assertEqual(summary["tokens"]["calls"], 0)

    def test_existing_run_directory_refuses_overwrite_before_model_calls(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "summary.json"
        sentinel.write_bytes(b'{"preserve":"original experiment"}\n')
        backend = ScriptedBackend([])
        with self.assertRaises(FileExistsError):
            Pipeline(backend, settings(), output).run(english_task())
        self.assertEqual(sentinel.read_bytes(), b'{"preserve":"original experiment"}\n')
        self.assertEqual(sorted(p.name for p in output.iterdir()), ["summary.json"])
        self.assertEqual(backend.requests, [])
        self.assertEqual(backend.counted_requests, [])

    def test_valid_json_is_not_accepted_when_generation_is_incomplete(self):
        cases = [
            ("length", {}), ("max_tokens", {}), ("incomplete", {}),
            ("stop", {"complete": False}),
        ]
        for index, (finish, metadata) in enumerate(cases):
            with self.subTest(finish=finish, metadata=metadata):
                task = english_task()
                summary, backend, output = self.run_script([
                    Reply("draft", draft(), output_tokens=51, finish_reason=finish, metadata=metadata),
                ], task=task, name=f"incomplete_{index}")
                self.assertEqual(summary["status"], "failed")
                self.assertEqual([r.role for r in backend.requests], ["draft"])
                self.assertEqual(summary["tokens"]["output_tokens"], 51)
                self.assertFalse((output / "final.json").exists())
                self.assertEqual(load_json(output / "state.json"), task.initial_state.model_dump())

    def test_demo_runs_end_to_end_for_both_modes(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        for mode in ("english", "history"):
            with self.subTest(mode=mode):
                task = Task.model_validate(load_json(examples / f"{mode}.json"))
                output = self.root / f"demo_{mode}"
                summary = Pipeline(DemoBackend(), Settings(backend="demo", candidates=2, schema_retries=0), output).run(task)
                self.assertEqual(summary["status"], "accepted_by_checks", summary.get("error"))
                self.assertTrue(summary["is_fixture"])
                self.assertEqual(summary["accepted_plan_count"], 2)
                self.assertEqual(summary["tokens"]["accounting"], ["fixture_estimate"])
                self.assertEqual(summary["output_sha256"], text_hash((output / "final.txt").read_text()))
                self.assertEqual(summary["reviewed_output_sha256"], summary["output_sha256"])
                events = self.events(output)
                self.assertEqual([e["sequence"] for e in events], list(range(1, len(events) + 1)))
                selection_context = next(event for event in events if event["kind"] == "selection_context")
                self.assertEqual(load_json(output / "selected_plan.json")["id"], selection_context["aliases"]["C1"])
                if mode == "history":
                    self.assertEqual(load_json(output / "state.json"), task.initial_state.model_dump())
                    self.assertIn("[S1]", (output / "final.txt").read_text())
                    self.assertEqual(summary["critic_reviewed_output_sha256"], summary["output_sha256"])
                    self.assertIsNone(summary["evidence_reviewed_output_sha256"])
                    self.assertEqual(len([event for event in events if event["kind"] == "semantic_critic"]), 1)
                    self.assertFalse(any(event["kind"] == "evidence_review" for event in events))
                else:
                    self.assertIn("mara_hinge", [entry["id"] for entry in load_json(output / "state.json")["world"]])


if __name__ == "__main__":
    unittest.main()
