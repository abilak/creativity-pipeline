"""Reject contradictory or ambiguous external inputs before generation begins."""
import unittest

from pydantic import ValidationError

from creativity_pipeline.checks import validate_plan
from creativity_pipeline.models import Claim, Move, Plan, Review, Settings, Source, SupportQuote, Task


def task_data(mode="english"):
    value = {"task_id": "contract_test", "mode": mode, "prompt": "Develop this account."}
    if mode == "history":
        value["sources"] = [{
            "id": "S1", "title": "Fictional source for testing",
            "text": "The levy changed in 1811. Demonstrations occurred in 1812.",
        }]
    return value


class InputContractTests(unittest.TestCase):
    def test_source_and_support_quote_preserve_original_whitespace(self):
        text = "  A source line.\r\n\nAnother line.  "
        source = Source(id="S1", title="Test source", text=text)
        quote = SupportQuote(source_id="S1", quote=text)
        self.assertEqual(source.text, text)
        self.assertEqual(quote.quote, text)
        with self.assertRaises(ValidationError):
            Source(id="S1", title="Blank", text=" \r\n ")

    def test_target_length_does_not_implicitly_create_hard_bounds(self):
        task = Task.model_validate(task_data())
        self.assertEqual(task.target_words, 700)
        self.assertIsNone(task.min_words)
        self.assertIsNone(task.max_words)

    def test_valid_optional_length_bounds_are_preserved(self):
        for limits in ({"min_words": 500}, {"max_words": 900},
                       {"min_words": 500, "max_words": 900},
                       {"min_words": 700, "max_words": 700}):
            with self.subTest(limits=limits):
                task = Task.model_validate({**task_data(), **limits})
                for field, value in limits.items():
                    self.assertEqual(getattr(task, field), value)

    def test_hard_bounds_cannot_conflict_with_the_target(self):
        for limits in ({"min_words": 701}, {"max_words": 699},
                       {"min_words": 800, "max_words": 600}):
            with self.subTest(limits=limits):
                with self.assertRaises(ValidationError):
                    Task.model_validate({**task_data(), **limits})

    def test_hard_bounds_must_be_positive_and_within_supported_limits(self):
        for field, value in (("min_words", 0), ("max_words", 0),
                             ("min_words", -1), ("max_words", -1),
                             ("min_words", 4001), ("max_words", 8001)):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValidationError):
                    Task.model_validate({**task_data(), field: value})

    def test_evidence_audit_defaults_on_and_selection_policy_is_explicit(self):
        self.assertTrue(Settings().history_evidence_audit)
        self.assertEqual(Settings().selection_policy, "judge")
        self.assertEqual(Settings(selection_policy="sample").selection_policy, "sample")
        self.assertFalse(Settings(history_evidence_audit=False).history_evidence_audit)
        with self.assertRaises(ValidationError):
            Settings(selection_policy="arbitrary")

    def test_conflicting_immutable_state_and_commitment_are_rejected(self):
        data = task_data()
        data["commitments"] = [{"id": "key", "text": "Mara has one key."}]
        data["initial_state"] = {"world": [{"id": "key", "text": "Mara has two keys."}], "reader": []}
        with self.assertRaisesRegex(ValidationError, "conflicts with commitment"):
            Task.model_validate(data)

    def test_agreeing_commitment_and_world_state_can_share_an_id(self):
        data = task_data()
        entry = {"id": "key", "text": "Mara has one key."}
        data["commitments"] = [entry]
        data["initial_state"] = {"world": [entry], "reader": []}
        task = Task.model_validate(data)
        self.assertEqual(task.initial_state.world[0].text, task.commitments[0].text)

    def test_reader_knowledge_is_not_mistaken_for_immutable_world_state(self):
        data = task_data()
        data["commitments"] = [{"id": "key", "text": "Mara has one key."}]
        data["initial_state"] = {
            "world": [{"id": "key", "text": "Mara has one key."}],
            "reader": [{"id": "key", "text": "The reader does not yet know whether Mara has a key."}],
        }
        task = Task.model_validate(data)
        self.assertNotEqual(task.initial_state.world[0].text, task.initial_state.reader[0].text)

    def test_structured_year_never_coerces_booleans_strings_or_floats(self):
        for contract, content_field in ((Move, "description"), (Claim, "text")):
            for year in (True, False, "1811", 1811.0, 1811.5):
                with self.subTest(contract=contract.__name__, year=year, value_type=type(year).__name__):
                    with self.assertRaises(ValidationError):
                        contract.model_validate({"id": "M1", content_field: "An event.", "claim_status": "unknown", "year": year})

    def test_integer_and_unknown_years_are_preserved(self):
        for contract, content_field in ((Move, "description"), (Claim, "text")):
            for year in (-44, 1811, None):
                with self.subTest(contract=contract.__name__, year=year):
                    item = contract.model_validate({"id": "M1", content_field: "An event.", "claim_status": "unknown", "year": year})
                    self.assertEqual(item.year, year)
                    if year is not None:
                        self.assertIs(type(item.year), int)

    def test_date_constraint_cannot_require_an_event_before_itself(self):
        data = task_data()
        data["date_constraints"] = [{"before": "M1", "after": "M1"}]
        with self.assertRaisesRegex(ValidationError, "cycle or self-ordering"):
            Task.model_validate(data)

    def test_date_constraints_cannot_contain_an_indirect_cycle(self):
        data = task_data()
        data["date_constraints"] = [
            {"before": "M1", "after": "M2"},
            {"before": "M2", "after": "M3"},
            {"before": "M3", "after": "M1"},
        ]
        with self.assertRaisesRegex(ValidationError, "cycle or self-ordering"):
            Task.model_validate(data)

    def test_valid_nonchronological_presentation_does_not_change_event_order(self):
        data = task_data("history")
        data["date_constraints"] = [{"before": "M1", "after": "M2"}]
        task = Task.model_validate(data)
        plan = Plan.model_validate({
            "id": "P1", "approach": "Start with demonstrations, then examine their context.",
            "intent": "Separate narrated order from event chronology.",
            "moves": [
                {"id": "M1", "description": "The levy changed in 1811.", "year": 1811,
                 "claim_status": "attested", "source_ids": ["S1"],
                 "support_quotes": [{"source_id": "S1", "quote": "The levy changed in 1811."}]},
                {"id": "M2", "description": "Demonstrations occurred in 1812.", "year": 1812,
                 "claim_status": "attested", "source_ids": ["S1"], "depends_on": ["M1"],
                 "support_quotes": [{"source_id": "S1", "quote": "Demonstrations occurred in 1812."}]},
            ],
            "presentation": ["M2", "M1"],
        })
        self.assertEqual(plan.presentation, ["M2", "M1"])
        self.assertEqual(validate_plan(task.model_dump(), plan.model_dump()), [])

    def test_history_requires_sources_but_fiction_does_not(self):
        data = task_data("history")
        data["sources"] = []
        with self.assertRaisesRegex(ValidationError, "History requires a source packet"):
            Task.model_validate(data)
        self.assertEqual(Task.model_validate(task_data("english")).sources, [])

    def test_empty_source_cannot_satisfy_history_evidence_requirement(self):
        data = task_data("history")
        data["sources"][0]["text"] = "   \n  "
        with self.assertRaises(ValidationError):
            Task.model_validate(data)

    def test_duplicate_source_ids_are_rejected_before_provenance_is_ambiguous(self):
        data = task_data("history")
        data["sources"].append({"id": "S1", "title": "Different report", "text": "A conflicting account."})
        with self.assertRaisesRegex(ValidationError, "Duplicate source IDs"):
            Task.model_validate(data)

    def test_accept_review_cannot_include_a_material_defect(self):
        with self.assertRaisesRegex(ValidationError, "accept review cannot contain material issues"):
            Review.model_validate({
                "decision": "accept", "issues": [{
                    "code": "source_mismatch", "message": "The source does not contain this quotation.",
                    "severity": "material",
                }],
            })


if __name__ == "__main__":
    unittest.main()
