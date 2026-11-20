"""Prompt boundary checks; these do not establish model compliance or quality."""
import json
import unittest

from creativity_pipeline.models import Task
from creativity_pipeline.prompts import messages


def history_task():
    return Task.model_validate({
        "task_id": "prompt_history", "mode": "history", "prompt": "Assess the supplied account.",
        "sources": [{
            "id": "S1", "title": "Source packet title", "author": "SOURCE_AUTHOR_SENTINEL",
            "date": "1812", "perspective": "SOURCE_PERSPECTIVE_SENTINEL",
            "text": "SOURCE_BEGIN The levy changed in 1811. SOURCE_END",
        }],
    })


def user_payload(rendered):
    user = rendered[-1]["content"]
    context, stage = user.split("\nStage inputs:\n", 1)
    task_context = json.loads(context.removeprefix("Task and immutable context:\n"))
    return task_context, json.loads(stage)


class PromptBoundaryTests(unittest.TestCase):
    def test_evidence_audit_ignores_writer_annotations_plan_and_previous_judgments(self):
        task = history_task()
        prose = "The report states that the levy changed in 1811. [S1]"
        rendered = messages(
            "review_evidence", task,
            draft={"text": prose, "claims": [{"id": "WRITER_CLAIM_SENTINEL"}],
                   "state_updates": {"world": [{"text": "WRITER_STATE_SENTINEL"}]}},
            plan={"intent": "WRITER_PLAN_SENTINEL"},
            defects=[{"message": "PREVIOUS_REVIEW_SENTINEL"}],
            previous=["PREVIOUS_APPROACH_SENTINEL"],
            candidates=[{"review": {"strengths": ["CANDIDATE_PRAISE_SENTINEL"]}}],
        )
        context, stage = user_payload(rendered)
        self.assertEqual(stage, {"prose": prose})
        self.assertEqual(context["sources"], task.model_dump()["sources"])
        text = "\n".join(message["content"] for message in rendered)
        for sentinel in ("WRITER_CLAIM_SENTINEL", "WRITER_STATE_SENTINEL", "WRITER_PLAN_SENTINEL",
                         "PREVIOUS_REVIEW_SENTINEL", "PREVIOUS_APPROACH_SENTINEL", "CANDIDATE_PRAISE_SENTINEL"):
            self.assertNotIn(sentinel, text)
        self.assertIn("SOURCE_AUTHOR_SENTINEL", text)
        self.assertIn("SOURCE_PERSPECTIVE_SENTINEL", text)

    def test_evidence_audit_keeps_entire_source_and_raw_prose(self):
        source_text = "START_OF_SOURCE " + "relevant evidence " * 2000 + " END_OF_SOURCE"
        task = history_task()
        task.sources[0].text = source_text
        prose = "Beginning.\n\n" + "A sentence with a claimed implication. " * 200 + "Final sentence."
        context, stage = user_payload(messages("review_evidence", task, draft={"text": prose}))
        self.assertEqual(context["sources"][0]["text"], source_text)
        self.assertEqual(stage["prose"], prose)

    def test_writer_receives_explicit_length_and_stable_event_constraints(self):
        task = Task.model_validate({
            **history_task().model_dump(), "min_words": 500, "max_words": 900,
            "date_constraints": [{"before": "levy_change", "after": "protest"}],
        })
        context, stage = user_payload(messages("draft", task, plan=None))
        self.assertEqual((context["min_words"], context["max_words"]), (500, 900))
        self.assertEqual(context["date_constraints"], [{"before": "levy_change", "after": "protest"}])
        self.assertIsNone(stage["plan"], "Direct generation still receives the task's explicit constraints")


if __name__ == "__main__":
    unittest.main()
