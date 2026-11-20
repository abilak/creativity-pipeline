"""Regressions for state continuity and preserving interrupted experiments."""
from contextlib import redirect_stdout, redirect_stderr
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from creativity_pipeline import cli
from creativity_pipeline.demo import DemoBackend
from creativity_pipeline.io import atomic_json, load_json


class InterruptedBackend(DemoBackend):
    def describe(self):
        return {"backend": "test_fake", "model_id": "test-only-model", "testing_only": True}

    def generate(self, request):
        raise KeyboardInterrupt


class AuditCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def invoke(self, *args):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return cli.main([str(value) for value in args])

    def test_state_import_preserves_new_facts_and_reader_world_separation(self):
        task = self.root / "task.json"
        state = self.root / "state.json"
        atomic_json(task, {"task_id": "next", "mode": "english", "prompt": "Continue.",
                           "initial_state": {"world": [{"id": "new", "text": "The train arrived."}],
                                             "reader": [{"id": "old", "text": "The reader knows the clock is wrong."}]}})
        atomic_json(state, {"world": [{"id": "old", "text": "The clock is wrong."}], "reader": []})
        actual = cli._task(task, state)
        self.assertEqual([row.id for row in actual.initial_state.world], ["old", "new"])
        self.assertEqual(actual.initial_state.reader[0].text, "The reader knows the clock is wrong.")

    def test_state_import_rejects_conflicting_definitions_and_duplicate_imports(self):
        task, state = self.root / "task.json", self.root / "state.json"
        atomic_json(task, {"task_id": "next", "mode": "english", "prompt": "Continue.",
                           "initial_state": {"world": [{"id": "old", "text": "The clock works."}]}})
        atomic_json(state, {"world": [{"id": "old", "text": "The clock is broken."}]})
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            cli._task(task, state)
        atomic_json(state, {"world": [{"id": "old", "text": "The clock works."}] * 2})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            cli._task(task, state)

    def test_interruption_keeps_every_designated_run_and_unknown_accounting(self):
        task, config, output = self.root / "task.json", self.root / "local.yaml", self.root / "runs"
        atomic_json(task, {"task_id": "case", "mode": "english", "prompt": "A short scene."})
        config.write_text(yaml.safe_dump({"backend": "transformers", "schema_retries": 0}), encoding="utf-8")
        with patch.object(cli, "_backend", return_value=InterruptedBackend()):
            status = self.invoke("benchmark", "--tasks", task, "--config", config,
                                 "--samples", "2", "--output", output)
        self.assertEqual(status, 130)
        ledger = load_json(output / "benchmark.json")
        self.assertEqual(len(ledger["designated"]), 6)
        self.assertEqual(ledger["designated"][0]["status"], "interrupted")
        self.assertEqual([row["status"] for row in ledger["designated"][1:]], ["pending"] * 5)
        summary = load_json(output / ledger["designated"][0]["run"] / "summary.json")
        self.assertFalse(summary["tokens"]["complete"])
        self.assertEqual(summary["tokens"]["unaccounted_calls"], 1)
        records = cli._run_records([output])
        self.assertEqual(len(records), 6)
        self.assertEqual(len({row["output_id"] for row in records}), 6)
        self.assertEqual(sum(row["status"] == "pending" for row in records), 5)
        self.assertTrue(all(row["text"] == "" for row in records))

    def test_legacy_run_without_stable_identity_cannot_be_counted_by_path(self):
        run = self.root / "legacy"
        run.mkdir()
        atomic_json(run / "summary.json", {"task_id": "legacy", "mode": "english", "strategy": "direct",
                                          "backend": {}, "status": "failed", "tokens": {}, "output_sha256": None})
        atomic_json(run / "config.json", {"seed": 1})
        atomic_json(run / "task.json", {"task_id": "legacy", "mode": "english", "prompt": "Write."})
        with self.assertRaisesRegex(ValueError, "stable run_id"):
            cli._run_records([run])

    def test_same_strategy_different_models_receive_different_conditions(self):
        roots = []
        for index in range(2):
            run = self.root / str(index)
            run.mkdir()
            roots.append(run)
            atomic_json(run / "summary.json", {"run_id": str(index), "task_id": "same", "mode": "english",
                                              "strategy": "multi_plan", "status": "failed", "tokens": {},
                                              "backend": {"model_id": f"Model-{index}"}, "output_sha256": None})
            atomic_json(run / "config.json", {"seed": 42, "model_id": f"Model-{index}", "strategy": "multi_plan"})
            atomic_json(run / "task.json", {"task_id": "same", "mode": "english", "prompt": "Same brief."})
        rows = cli._run_records(roots)
        self.assertNotEqual(rows[0]["condition_id"], rows[1]["condition_id"])
        self.assertEqual(rows[0]["task_sha256"], rows[1]["task_sha256"])

    def test_blind_review_cli_preserves_unrated_denominators_then_imports_a_rating(self):
        records = self.root / "test-records.json"
        manifest = self.root / "frozen.json"
        packet = self.root / "review-pack"
        atomic_json(records, [{"output_id": "test-only", "task_id": "brief", "mode": "english",
                              "baseline": "test-only", "condition_id": "test-condition",
                              "origin": "model", "text": "A temporary contract-test text.",
                              "testing_only": True, "task": {"prompt": "A contract-test brief."}}])
        self.assertEqual(self.invoke("freeze-outputs", "--records", records, "--output", manifest), 0)
        self.assertEqual(self.invoke("export-review", "--manifest", manifest, "--output", packet), 0)
        ratings, key = packet / "reviewer" / "ratings.csv", packet / "private_key.json"
        blank_report = self.root / "blank.json"
        self.assertEqual(self.invoke("import-review", ratings, "--key", key, "--output", blank_report), 0)
        self.assertEqual(load_json(blank_report)["n_unreviewed"], 1)
        with ratings.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        rows[0].update(reviewer_id="unit-test-rater", overall_quality="2", coherence="2",
                       originality="1", voice="2", material_error="no")
        with ratings.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        report_path = self.root / "rated.json"
        self.assertEqual(self.invoke("import-review", ratings, "--key", key, "--output", report_path), 0)
        self.assertEqual(load_json(report_path)["n_ratings"], 1)
        self.assertEqual(load_json(report_path)["n_unreviewed"], 0)


if __name__ == "__main__":
    unittest.main()
