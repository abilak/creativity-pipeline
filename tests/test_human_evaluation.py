"""Contract tests use synthetic records explicitly confined to test fixtures."""

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from creativity_pipeline.evaluation import freeze_outputs
from creativity_pipeline.human_evaluation import (
    MODE_DIMENSIONS,
    RATING_FIELDS,
    export_blind_review,
    import_human_ratings,
)


def review_records():
    # Labels are declared test metadata, never evidence of a real model run.
    rows = [
        {"output_id": "PRIVATE-output-model", "task_id": "english-task", "mode": "english",
         "baseline": "PRIVATE-strategy", "condition_id": "PRIVATE-base-condition", "origin": "model",
         "text": '<script>alert("unsafe")</script>\r\nA sentence & more.', "status": "PRIVATE-accepted",
         "backend": {"model_id": "PRIVATE-model"}, "seed": 918271,
         "task": {"task_id": "PRIVATE-task-label", "prompt": "Write about <b>departure</b>.",
                  "commitments": [{"id": "C1", "text": "The train never stops."}]}},
        {"output_id": "PRIVATE-output-failed", "task_id": "english-task", "mode": "english",
         "baseline": "PRIVATE-strategy", "condition_id": "PRIVATE-base-condition", "origin": "model",
         "text": "", "status": "PRIVATE-failed", "task": {"prompt": "Write about departure."}},
        {"output_id": "PRIVATE-output-human", "task_id": "english-task", "mode": "english",
         "baseline": "PRIVATE-control", "condition_id": "PRIVATE-human-condition", "origin": "human",
         "text": "A quiet departure.", "status": "PRIVATE-reference", "task": {"prompt": "Write about departure."}},
        {"output_id": "PRIVATE-output-history", "task_id": "history-task", "mode": "history",
         "baseline": "PRIVATE-strategy", "condition_id": "PRIVATE-base-condition", "origin": "model",
         "text": "The letter supports a limited inference. [S1]", "status": "PRIVATE-needs-review",
         "task": {"prompt": "Explain what the letter establishes.", "sources": [
             {"id": "S1", "title": "A letter", "author": "Witness", "date": "1850", "text": "I saw the crowd disperse."}]}},
    ]
    return rows


class BlindReviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = freeze_outputs(review_records())
        self.result = export_blind_review(self.manifest, self.root / "review", seed=42)
        self.key_path = Path(self.result["private_key"])
        self.key = json.loads(self.key_path.read_text(encoding="utf-8"))
        self.rows = list(csv.DictReader(io.StringIO(Path(self.result["ratings"]).read_text(encoding="utf-8"))))
        self.items = {item["output_id"]: item for item in self.key["items"]}

    def filled_rows(self, output_id="PRIVATE-output-model", reviewer_id="reader-1", score="3", **changes):
        rows = [dict(row) for row in self.rows]
        item = self.items[output_id]
        row = next(row for row in rows if row["item_id"] == item["item_id"])
        row["reviewer_id"] = reviewer_id
        row["material_error"] = "no"
        for dimension in MODE_DIMENSIONS[item["mode"]]:
            row[dimension] = score
        row.update(changes)
        return rows

    def import_rows(self, rows):
        path = self.root / "completed.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RATING_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return import_human_ratings(path, self.key_path)

    def test_public_packet_hides_metadata_and_escapes_text_and_task_content(self):
        packet = Path(self.result["packet"]).read_bytes().decode("utf-8")
        self.assertNotIn("PRIVATE-", packet)
        self.assertNotIn("918271", packet)
        self.assertNotIn("<script>", packet)
        self.assertIn("&lt;script&gt;", packet)
        self.assertIn("&lt;b&gt;departure&lt;/b&gt;", packet)
        self.assertIn("I saw the crowd disperse.", packet)
        self.assertIn("The train never stops.", packet)
        reviewer_dir = Path(self.result["packet"]).parent
        self.assertEqual({path.name for path in reviewer_dir.iterdir()}, {"packet.html", "ratings.csv"})
        self.assertNotEqual(self.key_path.parent, reviewer_dir)
        self.assertEqual(self.key["manifest"], self.manifest)

    def test_randomization_is_reproducible_and_independent_of_manifest_order(self):
        second = export_blind_review(list(reversed(self.manifest)), self.root / "second", seed=42)
        second_key = json.loads(Path(second["private_key"]).read_text())
        self.assertEqual(second_key["items"], self.key["items"])
        self.assertEqual(Path(second["packet"]).read_bytes(), Path(self.result["packet"]).read_bytes())
        third = export_blind_review(self.manifest, self.root / "third", seed=43)
        third_key = json.loads(Path(third["private_key"]).read_text())
        self.assertNotEqual([item["output_id"] for item in self.key["items"]],
                            [item["output_id"] for item in third_key["items"]])

    def test_existing_packet_is_never_overwritten(self):
        original = self.key_path.read_bytes()
        with self.assertRaises(FileExistsError):
            export_blind_review(self.manifest, self.result["output_dir"], seed=99)
        self.assertEqual(self.key_path.read_bytes(), original)

    def test_invalid_or_fixture_manifests_leave_no_output_directory(self):
        for change in ({"text": "tampered"}, {"backend": {"fixture": True}}):
            with self.subTest(change=change):
                rows = [dict(row) for row in self.manifest]
                rows[0].update(change)
                target = self.root / "invalid"
                with self.assertRaises(ValueError):
                    export_blind_review(rows, target)
                self.assertFalse(target.exists())

    def test_blank_ratings_keep_all_designated_failures_without_zero_scores(self):
        report = import_human_ratings(self.result["ratings"], self.key_path)
        self.assertEqual((report["n_designated"], report["n_with_text"], report["n_missing_text"]), (4, 3, 1))
        self.assertEqual(report["n_unreviewed"], 4)
        self.assertEqual(report["n_ratings"], 0)
        model = next(row for row in report["conditions"] if row["mode"] == "english" and row["origin"] == "model")
        self.assertEqual(model["n_designated"], 2)
        self.assertEqual(model["designated_status_counts"]["PRIVATE-failed"], 1)
        self.assertIsNone(model["dimensions"]["overall_quality"]["mean"])
        self.assertIn("No text was available", Path(self.result["packet"]).read_text())

    def test_scores_and_error_observations_stay_separate_and_keep_unreviewed_counts(self):
        rows = self.filled_rows(material_error="yes", error_notes="The train stops despite the commitment.")
        report = self.import_rows(rows)
        self.assertEqual((report["n_ratings"], report["n_reviewed_outputs"], report["n_unreviewed"]), (1, 1, 3))
        condition = next(row for row in report["conditions"] if row["mode"] == "english" and row["origin"] == "model")
        self.assertEqual(condition["n_unreviewed"], 1)
        self.assertEqual(condition["n_missing_text"], 1)
        self.assertEqual(condition["dimensions"]["voice"]["mean"], 3)
        self.assertEqual(condition["material_error_counts"], {"yes": 1, "no": 0, "uncertain": 0})
        self.assertNotIn("composite", condition)
        self.assertNotIn("evidence_support", condition["dimensions"])

    def test_history_has_distinct_required_dimensions(self):
        report = self.import_rows(self.filled_rows("PRIVATE-output-history", score="4", material_error="uncertain"))
        condition = next(row for row in report["conditions"] if row["mode"] == "history")
        self.assertEqual(condition["dimensions"]["evidence_support"]["mean"], 4)
        self.assertIn("historical_reasoning", condition["dimensions"])
        self.assertNotIn("voice", condition["dimensions"])
        rows = self.filled_rows("PRIVATE-output-history", voice="4")
        with self.assertRaisesRegex(ValueError, "not applicable"):
            self.import_rows(rows)

    def test_scores_require_integer_zero_through_four_and_reviewer_id(self):
        for bad in ("-1", "5", "3.0", "NaN", "true", "", "03"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "integer from 0 to 4"):
                    self.import_rows(self.filled_rows(overall_quality=bad))
        with self.assertRaisesRegex(ValueError, "reviewer_id"):
            self.import_rows(self.filled_rows(reviewer_id=""))
        with self.assertRaisesRegex(ValueError, "error_notes"):
            self.import_rows(self.filled_rows(material_error="yes"))

    def test_duplicate_reviewer_item_is_rejected_but_independent_reviewers_are_retained(self):
        rows = self.filled_rows()
        filled = next(row for row in rows if row["reviewer_id"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.import_rows(rows + [dict(filled)])
        second = {**filled, "reviewer_id": "reader-2", "overall_quality": "1"}
        report = self.import_rows(rows + [second])
        self.assertEqual(report["n_ratings"], 2)
        self.assertEqual(report["n_reviewed_outputs"], 1)
        condition = next(row for row in report["conditions"] if row["mode"] == "english" and row["origin"] == "model")
        self.assertEqual(condition["n_reviewers"], 2)
        self.assertEqual(condition["dimensions"]["overall_quality"]["mean"], 2)

    def test_no_text_outputs_cannot_receive_quality_scores(self):
        with self.assertRaisesRegex(ValueError, "no-text"):
            self.import_rows(self.filled_rows("PRIVATE-output-failed"))

    def test_unknown_items_and_changed_text_hashes_are_rejected_even_when_unrated(self):
        rows = [dict(row) for row in self.rows]
        rows[0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.import_rows(rows)
        rows[0]["item_id"] = "invented-item"
        with self.assertRaisesRegex(ValueError, "unknown item"):
            self.import_rows(rows)

    def test_duplicate_score_column_cannot_silently_override_a_judgment(self):
        self.import_rows(self.filled_rows(score="1"))
        path = self.root / "completed.csv"
        lines = path.read_text().splitlines()
        path.write_text(lines[0] + ",overall_quality\n" + "\n".join(line + ",4" for line in lines[1:]) + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate column"):
            import_human_ratings(path, self.key_path)

    def test_edited_reviewer_packet_cannot_be_joined_back_to_original_ratings(self):
        path = Path(self.result["packet"])
        path.write_bytes(path.read_bytes().replace(b"A sentence", b"An edited sentence"))
        with self.assertRaisesRegex(ValueError, "packet hash mismatch"):
            self.import_rows(self.filled_rows())

    def test_private_mapping_and_provenance_tampering_are_rejected(self):
        self.key["items"][0]["condition_id"] = "changed-condition"
        self.key_path.write_text(json.dumps(self.key), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "condition_id conflicts"):
            self.import_rows(self.filled_rows())
        self.key["items"][0]["condition_id"] = self.manifest[0]["condition_id"]
        self.key["manifest"][0]["status"] = "changed-status"
        self.key_path.write_text(json.dumps(self.key), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            self.import_rows(self.filled_rows())


if __name__ == "__main__":
    unittest.main()
