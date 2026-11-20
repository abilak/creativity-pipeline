"""Canned-report integrity tests, NOT demonstrations of an LLM's historical judgment."""
from __future__ import annotations

import copy
import json
import unittest

from pydantic import ValidationError

from creativity_pipeline.critic import (
    CriticReport, ENGLISH_CRITERIA, HISTORY_CRITERIA, MATERIAL_HISTORY_CODES,
    _without_schema_titles, critic_messages, validate_critic_report,
)
from creativity_pipeline.models import Task


def history_task():
    """All people, documents and statements are invented test fixtures."""
    return Task.model_validate({
        "task_id": "critic_bias_fixture", "mode": "history",
        "prompt": "Critically assess these invented documents; distinguish an author's assertions from established facts.",
        "sources": [
            {"id": "S1", "title": "Invented polemical pamphlet", "author": "A fictional male pamphleteer",
             "date": "1840", "kind": "polemic", "audience": "Town council", "purpose": "Oppose women's participation",
             "document_id": "pamphlet_document", "independence_group": "pamphlet_origin",
             "text": "Women are incapable of reason. The pamphlet opposes their participation in the council."},
            {"id": "S2", "title": "Invented meeting record", "author": "A fictional clerk", "date": "1841",
             "kind": "minutes", "purpose": "Record the meeting", "document_id": "minutes_document",
             "independence_group": "minutes_origin", "text": "Women presented detailed petitions to the council."},
        ],
    })


def english_task():
    return Task(task_id="critic_english_fixture", mode="english", prompt="Write a quiet observational story.")


ATTRIBUTED_PROSE = "The pamphlet portrays women as incapable of reason. [S1]"
UNQUALIFIED_PROSE = "Women were incapable of reason. [S1]"
ENGLISH_PROSE = "Mara tightened the hinge, then loosened it a quarter turn."


def report_for(task, prose):
    """Construct report-shaped data; no model is run or semantically evaluated."""
    criteria = HISTORY_CRITERIA if task.mode == "history" else ENGLISH_CRITERIA
    result = {
        "mode": task.mode, "decision": "accept", "coverage": "complete",
        "coverage_note": "Canned structure fixture; no model assessment occurred.",
        "rubric": [{"criterion": name, "judgment": "adequate", "draft_quote": prose,
                    "explanation": "Canned assessment to test the report contract."} for name in criteria],
        "source_assessments": [], "evidence_findings": [], "issues": [],
        "revision_priorities": [], "preserve": ["Preserve the limited attribution."], "missing_evidence": [],
    }
    if task.mode == "history":
        result["source_assessments"] = [{
            "source_id": source.id, "useful_for": "What this supplied source states.",
            "limitations": "A statement in a source does not automatically establish its truth.",
            "context_basis": "unknown", "context_quotes": [], "metadata_fields": [],
            "related_source_ids": [], "independence": "unknown",
        } for source in task.sources]
        result["evidence_findings"] = [{
            "draft_quote": prose, "claim_kind": "source_assertion", "support_level": "supported_as_attributed",
            "evidence": [{"source_id": "S1", "quote": "Women are incapable of reason.", "relation": "supports"}],
            "explanation": "The evidence supports attribution of the assertion, not the stereotype.",
        }]
    return result


def material_issue(prose, code="source_assertion_promoted_to_fact"):
    return {"code": code, "severity": "material", "location": prose,
            "message": "Attribute the claim to the supplied author and qualify what the evidence establishes."}


def errors(task, prose, report):
    return validate_critic_report(task.model_dump(), {"text": prose}, report)


def codes(findings):
    return {finding["code"] for finding in findings}


class CriticSchemaAndAnchoringTests(unittest.TestCase):
    def test_complete_canned_reports_are_valid_in_both_domains(self):
        for task, prose in ((history_task(), ATTRIBUTED_PROSE), (english_task(), ENGLISH_PROSE)):
            with self.subTest(mode=task.mode):
                report = report_for(task, prose)
                CriticReport.model_validate(report)
                self.assertEqual(errors(task, prose, report), [])

    def test_schema_forbids_unrecognized_fields(self):
        report = report_for(english_task(), ENGLISH_PROSE)
        report["invented_score"] = 99
        with self.assertRaises(ValidationError):
            CriticReport.model_validate(report)
        self.assertIn("critic_schema_invalid", codes(errors(english_task(), ENGLISH_PROSE, report)))

    def test_empty_acceptance_is_not_a_report(self):
        task = history_task()
        report = report_for(task, ATTRIBUTED_PROSE)
        report.update(rubric=[], source_assessments=[], evidence_findings=[])
        found = codes(errors(task, ATTRIBUTED_PROSE, report))
        self.assertTrue({"critic_incomplete_rubric", "critic_empty_acceptance", "critic_source_assessment_coverage", "critic_evidence_findings_missing"} <= found)

    def test_missing_and_duplicate_rubric_criteria_are_reported(self):
        task = english_task()
        report = report_for(task, ENGLISH_PROSE)
        report["rubric"][-1] = copy.deepcopy(report["rubric"][0])
        found = codes(errors(task, ENGLISH_PROSE, report))
        self.assertIn("critic_duplicate_criterion", found)
        self.assertIn("critic_incomplete_rubric", found)

    def test_wrong_domain_and_unknown_criterion_are_reported(self):
        task = english_task()
        report = report_for(task, ENGLISH_PROSE)
        report["mode"] = "history"
        report["rubric"][0]["criterion"] = "personal_taste"
        found = codes(errors(task, ENGLISH_PROSE, report))
        self.assertIn("critic_mode_mismatch", found)
        self.assertIn("critic_unknown_criterion", found)

    def test_applicable_rubric_quotes_must_be_present_and_exact(self):
        task = english_task()
        for quote in ("", "Mara found a nonexistent letter."):
            with self.subTest(quote=quote):
                report = report_for(task, ENGLISH_PROSE)
                report["rubric"][0]["draft_quote"] = quote
                self.assertTrue(errors(task, ENGLISH_PROSE, report))

    def test_material_issue_location_must_be_an_exact_nonempty_draft_span(self):
        task = english_task()
        for location in ("", "the ending", "A sentence that is not in this draft."):
            with self.subTest(location=location):
                report = report_for(task, ENGLISH_PROSE)
                report["decision"] = "revise"
                report["issues"] = [{**material_issue(ENGLISH_PROSE, "underdeveloped"), "location": location}]
                self.assertIn("critic_issue_location_mismatch", codes(errors(task, ENGLISH_PROSE, report)))

    def test_legitimate_material_literary_criticism_is_a_valid_report(self):
        task = english_task()
        report = report_for(task, ENGLISH_PROSE)
        report["decision"] = "revise"
        report["issues"] = [material_issue(ENGLISH_PROSE, "underdeveloped")]
        report["revision_priorities"] = ["Develop the consequence of the action."]
        self.assertEqual(errors(task, ENGLISH_PROSE, report), [])

    def test_partial_assessment_can_abstain_but_cannot_accept(self):
        task = history_task()
        report = report_for(task, ATTRIBUTED_PROSE)
        report.update(coverage="partial", decision="abstain", rubric=[], source_assessments=[], evidence_findings=[],
                      missing_evidence=["The source excerpt needed to assess the conclusion is missing."])
        self.assertEqual(errors(task, ATTRIBUTED_PROSE, report), [])
        report["decision"] = "accept"
        self.assertIn("critic_accepts_partial_coverage", codes(errors(task, ATTRIBUTED_PROSE, report)))

    def test_abstention_must_explain_what_is_missing(self):
        task = history_task()
        report = report_for(task, ATTRIBUTED_PROSE)
        report["decision"] = "abstain"
        self.assertIn("critic_abstention_unexplained", codes(errors(task, ATTRIBUTED_PROSE, report)))


class HistoricalReportIntegrityTests(unittest.TestCase):
    def test_canned_prejudice_critique_is_valid_and_does_not_become_validator_failure(self):
        task = history_task()
        report = report_for(task, UNQUALIFIED_PROSE)
        report["decision"] = "revise"
        report["evidence_findings"][0]["claim_kind"] = "event_claim"
        report["issues"] = [material_issue(UNQUALIFIED_PROSE, "prejudice_as_fact")]
        report["revision_priorities"] = ["Describe the author's prejudice as an attributed view, not a group fact."]
        self.assertEqual(errors(task, UNQUALIFIED_PROSE, report), [])

    def test_attributed_support_cannot_justify_acceptance_of_an_event_claim(self):
        task = history_task()
        report = report_for(task, UNQUALIFIED_PROSE)
        report["evidence_findings"][0]["claim_kind"] = "event_claim"
        self.assertIn("critic_accepts_unsupported_assertion", codes(errors(task, UNQUALIFIED_PROSE, report)))

    def test_explicitly_qualified_uncertainty_is_preserved(self):
        task = history_task()
        prose = "These excerpts do not establish how representative the pamphlet's view was."
        report = report_for(task, prose)
        report["evidence_findings"][0].update(claim_kind="uncertainty", support_level="insufficient", evidence=[])
        report["missing_evidence"] = ["A representative account of broader public views is not supplied."]
        self.assertEqual(errors(task, prose, report), [])

    def test_unsupported_interpretation_needs_material_issue_or_abstention(self):
        task = history_task()
        prose = "The pamphlet caused every subsequent council decision."
        report = report_for(task, prose)
        report["evidence_findings"][0].update(claim_kind="interpretation", support_level="insufficient", evidence=[])
        self.assertIn("critic_accepts_unsupported_assertion", codes(errors(task, prose, report)))
        report["decision"] = "revise"
        self.assertIn("critic_unreported_material_problem", codes(errors(task, prose, report)))
        report["issues"] = [material_issue(prose, "unsupported_causal_claim")]
        self.assertEqual(errors(task, prose, report), [])

    def test_contradicted_finding_needs_a_challenging_passage(self):
        task = history_task()
        prose = "No women presented petitions to the council."
        report = report_for(task, prose)
        report["decision"] = "revise"
        report["issues"] = [material_issue(prose, "ignored_material_counterevidence")]
        finding = report["evidence_findings"][0]
        finding.update(claim_kind="event_claim", support_level="contradicted", evidence=[])
        self.assertIn("critic_contradiction_basis_missing", codes(errors(task, prose, report)))
        finding["evidence"] = [{"source_id": "S2", "quote": task.sources[1].text, "relation": "challenges"}]
        self.assertEqual(errors(task, prose, report), [])

    def test_material_historical_codes_cannot_be_downgraded_to_minor(self):
        task = history_task()
        for code in MATERIAL_HISTORY_CODES:
            with self.subTest(code=code):
                report = report_for(task, ATTRIBUTED_PROSE)
                report["issues"] = [{**material_issue(ATTRIBUTED_PROSE, code), "severity": "minor"}]
                self.assertIn("critic_material_issue_downgraded", codes(errors(task, ATTRIBUTED_PROSE, report)))

    def test_complete_report_assesses_even_uncited_supplied_sources(self):
        task = history_task()
        report = report_for(task, ATTRIBUTED_PROSE)
        report["source_assessments"] = report["source_assessments"][:1]
        self.assertIn("critic_source_assessment_coverage", codes(errors(task, ATTRIBUTED_PROSE, report)))

    def test_unknown_source_and_invented_source_quote_are_invalid_report_data(self):
        task = history_task()
        for mutation, code in (({"source_id": "S99"}, "critic_unknown_source"),
                               ({"quote": "The author wrote a sentence absent from the packet."}, "critic_source_quote_mismatch")):
            with self.subTest(mutation=mutation):
                report = report_for(task, ATTRIBUTED_PROSE)
                report["evidence_findings"][0]["evidence"][0].update(mutation)
                self.assertIn(code, codes(errors(task, ATTRIBUTED_PROSE, report)))

    def test_source_quote_matching_normalizes_only_whitespace(self):
        task = history_task()
        report = report_for(task, ATTRIBUTED_PROSE)
        report["evidence_findings"][0]["evidence"][0]["quote"] = "Women\n are   incapable of reason."
        self.assertEqual(errors(task, ATTRIBUTED_PROSE, report), [])

    def test_evidence_finding_cannot_quote_nonexistent_draft_text(self):
        task = history_task()
        report = report_for(task, ATTRIBUTED_PROSE)
        report["evidence_findings"][0]["draft_quote"] = "A nonexistent narrator assertion."
        self.assertIn("critic_draft_quote_mismatch", codes(errors(task, ATTRIBUTED_PROSE, report)))

    def test_context_basis_requires_actual_quotes_or_available_metadata(self):
        task = history_task()
        report = report_for(task, ATTRIBUTED_PROSE)
        assessment = report["source_assessments"][0]
        assessment["context_basis"] = "quoted_text"
        self.assertIn("critic_context_basis_missing", codes(errors(task, ATTRIBUTED_PROSE, report)))
        assessment.update(context_basis="provided_metadata", metadata_fields=["purpose"])
        self.assertEqual(errors(task, ATTRIBUTED_PROSE, report), [])
        assessment["metadata_fields"] = ["invented_biography"]
        self.assertIn("critic_unavailable_metadata_basis", codes(errors(task, ATTRIBUTED_PROSE, report)))

    def test_known_shared_origin_or_same_document_cannot_be_independent(self):
        for field in ("independence_group", "document_id"):
            with self.subTest(field=field):
                task = history_task()
                setattr(task.sources[1], field, getattr(task.sources[0], field))
                report = report_for(task, ATTRIBUTED_PROSE)
                report["source_assessments"][0].update(
                    independence="independent", context_basis="provided_metadata", metadata_fields=[field])
                self.assertIn("critic_false_independence", codes(errors(task, ATTRIBUTED_PROSE, report)))

    def test_unknown_context_cannot_support_a_claim_of_independence(self):
        task = history_task()
        report = report_for(task, ATTRIBUTED_PROSE)
        report["source_assessments"][0]["independence"] = "independent"
        self.assertIn("critic_independence_basis_missing", codes(errors(task, ATTRIBUTED_PROSE, report)))

    def test_shared_origin_requires_related_source_and_explicit_basis(self):
        task = history_task()
        report = report_for(task, ATTRIBUTED_PROSE)
        assessment = report["source_assessments"][0]
        assessment["independence"] = "shared_origin"
        self.assertIn("critic_shared_origin_basis_missing", codes(errors(task, ATTRIBUTED_PROSE, report)))
        task.sources[1].independence_group = task.sources[0].independence_group
        assessment.update(related_source_ids=["S2"], context_basis="provided_metadata", metadata_fields=["independence_group"])
        self.assertEqual(errors(task, ATTRIBUTED_PROSE, report), [])


class CriticPromptBoundaryTests(unittest.TestCase):
    def test_compact_prompt_preserves_exact_source_and_prose_data(self):
        task = history_task()
        task.sources[0].text = "  SOURCE_START\n" + task.sources[0].text + "\nSOURCE_END  "
        prose = "  " + ATTRIBUTED_PROSE + "\n\nFinal sentence.  "
        rendered = critic_messages(task, prose)
        payload = json.loads(rendered[-1]["content"])
        self.assertEqual(payload, {"task": task.model_dump(), "prose": prose})
        self.assertEqual(set(payload), {"task", "prose"})
        self.assertNotIn("writer_annotations", payload)
        self.assertNotIn("plan", payload)
        self.assertNotIn("previous_review", payload)

    def test_prompt_states_method_and_anti_demographic_shortcut_without_claiming_it_worked(self):
        system = critic_messages(history_task(), ATTRIBUTED_PROSE)[0]["content"]
        for instruction in ("demographic identity alone neither proves nor refutes", "attitude", "stereotype",
                            "independence_group", "universal two-source rule", "contrary evidence",
                            "audience", "purpose", "material Issue.location"):
            self.assertIn(instruction, system)
        # This tests prompt content only. It does not demonstrate unbiased LLM behavior.

    def test_changing_author_gender_preserves_evidence_and_the_same_instructions(self):
        task = history_task()
        male = critic_messages(task, ATTRIBUTED_PROSE)
        task.sources[0].author = "A fictional female pamphleteer"
        female = critic_messages(task, ATTRIBUTED_PROSE)
        self.assertEqual(male[0], female[0])
        male_source = json.loads(male[1]["content"])["task"]["sources"][0]
        female_source = json.loads(female[1]["content"])["task"]["sources"][0]
        self.assertEqual(male_source["text"], female_source["text"])
        self.assertNotEqual(male_source["author"], female_source["author"])

    def test_schema_compaction_preserves_a_real_field_named_title(self):
        schema = {"title": "Container", "type": "object", "properties": {
            "title": {"title": "A field annotation", "type": "string"},
            "data": {"title": "Data", "type": "array", "items": {"title": "Item", "type": "string"}},
        }, "required": ["title"]}
        compact = _without_schema_titles(schema)
        self.assertNotIn("title", compact)
        self.assertIn("title", compact["properties"])
        self.assertEqual(compact["properties"]["title"], {"type": "string"})
        self.assertEqual(compact["required"], ["title"])
        self.assertNotIn("title", compact["properties"]["data"]["items"])


if __name__ == "__main__":
    unittest.main()
