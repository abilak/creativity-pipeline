import copy
import unittest

from creativity_pipeline.checks import (
    plan_duplicate_diagnostics,
    validate_draft,
    validate_plan,
    validate_state_update,
)


def history_task():
    return {
        "task_id": "history-1", "mode": "history", "prompt": "Compare explanations.",
        "sources": [
            {"id": "S1", "title": "Report", "text": "The tax rose in 1840.\n\nUnrest began in 1842."},
            {"id": "S2", "title": "Neighbor", "text": "The neighboring district remained calm."},
        ],
        "commitments": [], "initial_state": {"world": [], "reader": []},
        "date_constraints": [{"before": "M1", "after": "M2"}],
    }


def plan():
    return {
        "id": "P1", "approach": "Compare mechanisms", "intent": "Separate sequence and cause.",
        "moves": [
            {"id": "M1", "description": "The tax rose in 1840.", "source_ids": ["S1"],
             "support_quotes": [{"source_id": "S1", "quote": "The tax rose in 1840."}],
             "claim_status": "attested", "year": 1840, "depends_on": []},
            {"id": "M2", "description": "Unrest began in 1842.", "source_ids": ["S1"],
             "support_quotes": [{"source_id": "S1", "quote": "Unrest began in 1842."}],
             "claim_status": "attested", "year": 1842, "depends_on": ["M1"]},
        ],
        "presentation": ["M2", "M1"], "open_options": [],
    }


def draft():
    return {
        "text": "The tax rose in 1840. [S1] Unrest began in 1842. [S1]",
        "claims": [
            {"id": "M1", "text": "The tax rose in 1840.", "source_ids": ["S1"],
             "support_quotes": [{"source_id": "S1", "quote": "The tax rose in 1840."}],
             "claim_status": "attested", "year": 1840},
            {"id": "M2", "text": "Unrest began in 1842.", "source_ids": ["S1"],
             "support_quotes": [{"source_id": "S1", "quote": "Unrest began in 1842."}],
             "claim_status": "attested", "year": 1842},
        ],
        "state_updates": {"world": [], "reader": []},
    }


def codes(issues):
    return {issue["code"] for issue in issues}


class PlanChecksTest(unittest.TestCase):
    def test_valid_plan_allows_nonchronological_presentation(self):
        self.assertEqual(validate_plan(history_task(), plan()), [])

    def test_sources_are_required_for_history(self):
        task = history_task()
        task["sources"] = []
        self.assertIn("history_sources_required", codes(validate_plan(task, plan())))

    def test_unknown_sources_and_quotes_fail(self):
        candidate = plan()
        candidate["moves"][0]["source_ids"] = ["S999"]
        candidate["moves"][0]["support_quotes"] = [{"source_id": "S999", "quote": "Made up."}]
        found = codes(validate_plan(history_task(), candidate))
        self.assertIn("unknown_source", found)
        self.assertIn("unknown_quote_source", found)

    def test_quote_whitespace_is_normalized_but_words_are_not(self):
        candidate = plan()
        candidate["moves"][0]["support_quotes"][0]["quote"] = "The tax rose in 1840.   Unrest\nbegan in 1842."
        self.assertNotIn("quote_mismatch", codes(validate_plan(history_task(), candidate)))
        candidate["moves"][0]["support_quotes"][0]["quote"] = "The tax fell in 1840."
        self.assertIn("quote_mismatch", codes(validate_plan(history_task(), candidate)))

    def test_attested_requires_passage_and_inference_requires_citation(self):
        candidate = plan()
        candidate["moves"][0]["support_quotes"] = []
        self.assertIn("attested_quote_required", codes(validate_plan(history_task(), candidate)))
        candidate["moves"][0]["claim_status"] = "inference"
        self.assertIn("support_passage_missing", codes(validate_plan(history_task(), candidate)))
        candidate["moves"][0]["source_ids"] = []
        self.assertIn("missing_provenance", codes(validate_plan(history_task(), candidate)))
        candidate["moves"][0]["claim_status"] = "unknown"
        self.assertNotIn("missing_provenance", codes(validate_plan(history_task(), candidate)))

    def test_quote_must_be_listed_as_a_cited_source(self):
        candidate = plan()
        candidate["moves"][0]["support_quotes"] = [{"source_id": "S2", "quote": "The neighboring district remained calm."}]
        self.assertIn("quote_source_not_cited", codes(validate_plan(history_task(), candidate)))

    def test_dependencies_cannot_be_unknown_or_cyclic(self):
        candidate = plan()
        candidate["moves"][0]["depends_on"] = ["M2", "missing"]
        found = codes(validate_plan(history_task(), candidate))
        self.assertIn("unknown_dependency", found)
        self.assertIn("dependency_cycle", found)

    def test_duplicate_ids_are_material(self):
        candidate = plan()
        candidate["moves"].append(copy.deepcopy(candidate["moves"][0]))
        issues = validate_plan(history_task(), candidate)
        self.assertTrue(any(issue["code"] == "duplicate_id" and issue["severity"] == "material" for issue in issues))

    def test_date_constraints_check_years_not_prose_order(self):
        candidate = plan()
        candidate["moves"][0]["year"] = 1850
        self.assertIn("date_order_violation", codes(validate_plan(history_task(), candidate)))
        candidate["moves"][0]["year"] = 1842
        issues = validate_plan(history_task(), candidate)
        self.assertTrue(any(issue["code"] == "date_constraint_unresolved" and issue["severity"] == "minor" for issue in issues))
        candidate["moves"][0]["year"] = None
        self.assertIn("date_constraint_unresolved", codes(validate_plan(history_task(), candidate)))

    def test_missing_required_date_move_is_reported(self):
        task = history_task()
        task["date_constraints"][0]["after"] = "not_in_plan"
        self.assertIn("date_constraint_unknown_move", codes(validate_plan(task, plan())))

    def test_exact_source_quote_does_not_certify_entailment(self):
        candidate = plan()
        candidate["moves"][0]["description"] = "This report proves the author's private intentions."
        # The exact quotation check is deliberately narrower than semantic review.
        self.assertEqual(validate_plan(history_task(), candidate), [])


class DraftChecksTest(unittest.TestCase):
    def test_valid_annotated_draft(self):
        self.assertEqual(validate_draft(history_task(), draft(), plan()), [])

    def test_empty_history_claim_list_is_not_accepted(self):
        candidate = draft()
        candidate["claims"] = []
        self.assertIn("history_claims_required", codes(validate_draft(history_task(), candidate)))

    def test_annotations_must_be_exact_substrings(self):
        candidate = draft()
        candidate["claims"][0]["text"] = "The tax rose in 1841."
        self.assertIn("claim_text_missing", codes(validate_draft(history_task(), candidate)))

    def test_unknown_and_unreferenced_citation_markers(self):
        candidate = draft()
        candidate["text"] += " [S999] [S2]"
        found = codes(validate_draft(history_task(), candidate))
        self.assertIn("unknown_citation_marker", found)
        self.assertIn("unreferenced_citation_marker", found)

    def test_fabricated_second_sentence_cannot_borrow_an_earlier_citation(self):
        candidate = draft()
        candidate["text"] += " The king secretly directed every arrest. [S1]"
        issues = validate_draft(history_task(), candidate)
        unbound = [issue for issue in issues if issue["code"] == "unreferenced_citation_marker"]
        self.assertEqual(len(unbound), 1)
        self.assertEqual(unbound[0]["severity"], "material")

    def test_marker_cannot_bind_backwards_from_the_following_supported_sentence(self):
        candidate = draft()
        candidate["text"] = "The king secretly directed every arrest. [S1] " + candidate["text"]
        self.assertIn("unreferenced_citation_marker", codes(validate_draft(history_task(), candidate)))

    def test_marker_inside_exact_claim_span_is_bound(self):
        candidate = draft()
        candidate["claims"][0]["text"] += " [S1]"
        self.assertNotIn("unreferenced_citation_marker", codes(validate_draft(history_task(), candidate)))

    def test_adjacent_marker_allows_punctuation_but_no_intervening_prose(self):
        candidate = draft()
        candidate["claims"][0]["text"] = "The tax rose in 1840"
        candidate["text"] = "The tax rose in 1840.\n— ([S1]) Unrest began in 1842. [S1]"
        self.assertNotIn("unreferenced_citation_marker", codes(validate_draft(history_task(), candidate)))
        candidate["text"] = "The tax rose in 1840 and immediately caused every later arrest. [S1] Unrest began in 1842. [S1]"
        self.assertIn("unreferenced_citation_marker", codes(validate_draft(history_task(), candidate)))

    def test_adjacent_citation_cluster_binds_each_source_to_the_claim(self):
        candidate = draft()
        candidate["claims"][0]["source_ids"] = ["S1", "S2"]
        candidate["text"] = "The tax rose in 1840. [S1] [S2] Unrest began in 1842. [S1]"
        self.assertNotIn("unreferenced_citation_marker", codes(validate_draft(history_task(), candidate)))

    def test_source_used_elsewhere_does_not_bind_wrong_local_marker(self):
        task = history_task()
        task["date_constraints"] = []
        candidate = {
            "text": "The tax rose in 1840. [S2] The neighboring district remained calm. [S2]",
            "claims": [
                {"id": "C1", "text": "The tax rose in 1840.", "source_ids": ["S1"],
                 "support_quotes": [{"source_id": "S1", "quote": "The tax rose in 1840."}], "claim_status": "attested"},
                {"id": "C2", "text": "The neighboring district remained calm.", "source_ids": ["S2"],
                 "support_quotes": [{"source_id": "S2", "quote": "The neighboring district remained calm."}], "claim_status": "attested"},
            ],
            "state_updates": {"world": [], "reader": []},
        }
        self.assertIn("unreferenced_citation_marker", codes(validate_draft(task, candidate)))

    def test_history_cannot_commit_world_updates(self):
        candidate = draft()
        candidate["state_updates"]["world"] = [{"id": "W1", "text": "Tax is proven causal.", "evidence": "The tax rose in 1840."}]
        self.assertIn("history_world_state_update", codes(validate_draft(history_task(), candidate)))

    def test_shared_plan_claim_id_cannot_change_declared_year(self):
        candidate = draft()
        candidate["claims"][0]["year"] = 1841
        self.assertIn("planned_year_changed", codes(validate_draft(history_task(), candidate, plan())))

    def test_direct_draft_checks_date_constraints_without_a_plan(self):
        candidate = draft()
        candidate["claims"][0]["year"] = 1850
        issues = validate_draft(history_task(), candidate)
        self.assertTrue(any(issue["code"] == "date_order_violation" and issue["severity"] == "material" for issue in issues))

    def test_final_claims_must_preserve_constrained_event_ids(self):
        candidate = draft()
        candidate["claims"][0]["id"] = "C1"
        for selected_plan in (None, plan()):
            with self.subTest(has_plan=selected_plan is not None):
                issues = validate_draft(history_task(), candidate, selected_plan)
                self.assertTrue(any(issue["code"] == "date_constraint_unknown_claim" and issue["severity"] == "material" for issue in issues))

    def test_same_or_missing_final_year_is_unresolved_not_proven_wrong(self):
        for year in (1842, None):
            with self.subTest(year=year):
                candidate = draft()
                candidate["claims"][0]["year"] = year
                issues = validate_draft(history_task(), candidate)
                self.assertTrue(any(issue["code"] == "date_constraint_unresolved" and issue["severity"] == "minor" for issue in issues))
                self.assertNotIn("date_order_violation", codes(issues))

    def test_dated_shared_id_reports_omitted_year_even_without_order_constraint(self):
        task = history_task()
        task["date_constraints"] = []
        candidate = draft()
        candidate["claims"][0]["year"] = None
        issues = validate_draft(task, candidate, plan())
        self.assertTrue(any(issue["code"] == "planned_year_missing" and issue["severity"] == "minor" for issue in issues))

    def test_final_checks_do_not_infer_a_year_from_claim_text(self):
        candidate = draft()
        candidate["claims"][0]["year"] = None
        # The prose contains 1840, but this checker does not extract facts from it.
        self.assertIn("date_constraint_unresolved", codes(validate_draft(history_task(), candidate)))

    def test_initial_commitment_cannot_be_overwritten(self):
        task = {"mode": "english", "sources": [], "commitments": [{"id": "W1", "text": "The door was locked."}], "initial_state": {"world": [], "reader": []}}
        candidate = {"text": "The door had always been open.", "claims": [], "state_updates": {"world": [{"id": "W1", "text": "The door was open.", "evidence": "The door had always been open."}], "reader": []}}
        self.assertIn("state_overwrite", codes(validate_draft(task, candidate)))


class StateAndDuplicateChecksTest(unittest.TestCase):
    def test_updates_require_draft_evidence_and_preserve_existing_ids(self):
        existing = {"world": [{"id": "W1", "text": "The key is lost."}], "reader": []}
        updates = {"world": [{"id": "W1", "text": "The key is found.", "evidence": "not present"}], "reader": []}
        found = codes(validate_state_update(existing, updates, "She searched the room."))
        self.assertEqual(found, {"state_overwrite", "state_evidence_missing"})

    def test_new_state_entry_can_be_added_with_exact_evidence(self):
        existing = {"world": [], "reader": []}
        updates = {"world": [{"id": "W1", "text": "The door is locked.", "evidence": "She locked the door."}], "reader": []}
        self.assertEqual(validate_state_update(existing, updates, "She locked the door."), [])

    def test_duplicate_diagnostics_are_minor_and_not_semantic_rewards(self):
        candidate = plan()
        duplicate = copy.deepcopy(candidate)
        duplicate["id"] = "P2"
        issues = plan_duplicate_diagnostics([candidate, duplicate])
        self.assertEqual(codes(issues), {"duplicate_plan_text"})
        self.assertTrue(all(issue["severity"] == "minor" for issue in issues))


class LengthChecksTest(unittest.TestCase):
    def check_length(self, count, **limits):
        task = {"mode": "english", "sources": [], **limits}
        candidate = {"text": " ".join(["word"] * count), "claims": [], "state_updates": {"world": [], "reader": []}}
        return validate_draft(task, candidate)

    def test_explicit_minimum_is_material(self):
        issues = self.check_length(4, target_words=700, min_words=500)
        self.assertTrue(any(issue["code"] == "below_min_words" and issue["severity"] == "material" for issue in issues))

    def test_explicit_maximum_is_material(self):
        issues = self.check_length(101, max_words=100)
        self.assertTrue(any(issue["code"] == "above_max_words" and issue["severity"] == "material" for issue in issues))

    def test_explicit_bounds_are_inclusive(self):
        self.assertEqual(self.check_length(100, min_words=100, max_words=100), [])

    def test_target_deviation_alone_is_advisory(self):
        issues = self.check_length(4, target_words=700)
        self.assertEqual(codes(issues), {"target_length_deviation"})
        self.assertEqual(issues[0]["severity"], "minor")

    def test_broad_target_band_is_inclusive(self):
        for count in (50, 150):
            with self.subTest(count=count):
                self.assertEqual(self.check_length(count, target_words=100), [])
        for count in (49, 151):
            with self.subTest(count=count):
                self.assertIn("target_length_deviation", codes(self.check_length(count, target_words=100)))

    def test_no_requested_length_does_not_invent_a_minimum(self):
        self.assertEqual(self.check_length(4), [])


if __name__ == "__main__":
    unittest.main()
