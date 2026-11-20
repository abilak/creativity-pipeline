"""CLI glue tests. All generated run records here are temporary test data."""
from contextlib import redirect_stderr, redirect_stdout
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from creativity_pipeline import cli
from creativity_pipeline.demo import DemoBackend
from creativity_pipeline.io import atomic_json, load_json, text_hash


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class RejectDraftBackend(DemoBackend):
    def generate(self, request):
        result = super().generate(request)
        if request.role == "review_draft":
            value = json.loads(result.text)
            value["decision"] = "reject"
            return replace(result, text=json.dumps(value))
        return result


class FailDirectBackend(DemoBackend):
    def generate(self, request):
        result = super().generate(request)
        if request.role == "draft" and request.metadata.get("plan") is None:
            return replace(result, text="deliberately invalid test response", output_tokens=20)
        return result


class CliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def invoke(self, *arguments):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            status = cli.main([str(value) for value in arguments])
        return status, output.getvalue(), error.getvalue()

    def config(self, **overrides):
        values = {
            "backend": "demo", "strategy": "multi_plan", "candidates": 2,
            "schema_retries": 0, "max_repairs": 0,
            "max_context_tokens": 8192, "max_run_tokens": 60000,
            "max_output_tokens": 16000, "max_calls": 24,
            "plan_tokens": 1024, "draft_tokens": 3072,
            "review_tokens": 768, "selection_tokens": 256, "seed": 7,
        }
        values.update(overrides)
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(values))
        return path, values

    def make_run(self, name, *, text="Temporary test output.", status="accepted_by_checks", fixture=False, backend_fixture=False):
        """Simulate the disk contract only; these records never leave the test directory."""
        directory = self.root / "runs" / name
        directory.mkdir(parents=True)
        summary = {
            "run_id": "test-" + name,
            "task_id": name, "mode": "english", "strategy": "direct", "status": status,
            "is_fixture": fixture,
            "backend": {"backend": "test_record_only", "fixture": backend_fixture, "testing_only": True},
            "tokens": {"calls": 2, "input_tokens": 20, "output_tokens": 10,
                       "total_tokens": 30, "accounting": ["test_only_estimate"]},
            "output_sha256": text_hash(text) if text is not None else None,
            "testing_only": True,
        }
        atomic_json(directory / "summary.json", summary)
        atomic_json(directory / "config.json", {"seed": 7})
        atomic_json(directory / "task.json", {"task_id": name, "mode": "english", "prompt": "Temporary test brief."})
        if text is not None:
            (directory / "final.txt").write_text(text, encoding="utf-8")
        return directory

    def write_training_events(self, directory, rows=None):
        plan = {
            "id": "P1", "approach": "A particular consequence", "intent": "Leave the ending open",
            "moves": [{"id": "M1", "description": "Mara notices a hinge.", "claim_status": "fiction"}],
            "presentation": ["M1"], "open_options": [],
        }
        draft = {"text": "Mara left the hinge alone.", "claims": [], "state_updates": {"world": [], "reader": []}}
        plan_text = "\n" + json.dumps(plan, ensure_ascii=False, indent=3) + "\n"
        draft_text = json.dumps(draft, ensure_ascii=False, indent=1)
        if rows is None:
            rows = [
                ("plan", False, plan_text, "stop", {}),
                ("review_plan", False, '{"decision":"accept","issues":[],"strengths":["review-only marker"]}', "stop", {}),
                ("draft", True, draft_text, "stop", {"raw_output": "<think>private test trace</think>" + draft_text}),
                ("repair", False, draft_text, "stop", {}),
                ("draft", False, draft_text, "length", {}),
                ("draft", False, draft_text, "stop", {"complete": False}),
                ("draft", False, "bad JSON", "stop", {}),
            ]
        events, saved = [], {}
        for call_id, (role, thinking, completion, finish, metadata) in enumerate(rows, start=1):
            request = {
                "role": role, "thinking": thinking,
                "messages": [
                    {"role": "system", "content": "Keep exact spacing.\n  This is test data."},
                    {"role": "user", "content": f"Mara’s task #{call_id}:\nDo not change this prompt."},
                ],
            }
            response = {"text": completion, "finish_reason": finish, "metadata": metadata}
            events.extend([
                {"kind": "request", "call_id": call_id, "request": request},
                {"kind": "response", "call_id": call_id, "response": response},
            ])
            saved[call_id] = {"request": request, "response": response}
        (directory / "events.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events))
        return saved

    def test_demo_cli_runs_both_modes_and_marks_outputs_as_fixtures(self):
        for mode in ("english", "history"):
            with self.subTest(mode=mode):
                output = self.root / f"demo_{mode}"
                status, stdout, stderr = self.invoke("demo", "--task", mode, "--output", output)
                self.assertEqual(status, 0, stderr)
                printed = json.loads(stdout)
                self.assertEqual(printed, load_json(output / "summary.json"))
                self.assertEqual(printed["status"], "accepted_by_checks")
                self.assertTrue(printed["is_fixture"])
                self.assertTrue(printed["backend"]["fixture"])
                self.assertTrue((output / "final.txt").exists())

    def test_generate_returns_error_status_when_checks_reject_draft(self):
        config, _ = self.config(strategy="direct")
        output = self.root / "rejected"
        with patch("creativity_pipeline.cli._backend", return_value=RejectDraftBackend()):
            status, stdout, _ = self.invoke("generate", EXAMPLES / "english.json", "--config", config, "--output", output)
        self.assertEqual(status, 2)
        self.assertEqual(json.loads(stdout)["status"], "needs_review")
        self.assertTrue((output / "summary.json").exists())

    def test_critique_preserves_supplied_crlf_and_does_not_create_generation_records(self):
        config, _ = self.config(strategy="direct")
        prose = "  Mara set down the key.\r\n\r\nThen she opened the window.\r\n"
        supplied = self.root / "draft.txt"
        supplied.write_bytes(prose.encode("utf-8"))
        output = self.root / "critique"
        status, stdout, error = self.invoke("critique", EXAMPLES / "english.json", "--draft", supplied,
                                           "--config", config, "--output", output)
        self.assertEqual(status, 0, error + stdout)
        assessed = json.loads(stdout)
        self.assertEqual(assessed["input_sha256"], text_hash(prose))
        self.assertEqual(load_json(output / "input_draft.json")["text"], prose)
        self.assertTrue(assessed["is_fixture"])
        self.assertFalse((output / "summary.json").exists())
        self.assertFalse((output / "final.txt").exists())

    def test_run_export_preserves_legitimate_crlf_text_hash(self):
        prose = "First test line.\r\n\r\nSecond test line.\r\n"
        run = self.make_run("crlf", text=prose)
        rows = cli._run_records([str(run)])
        self.assertEqual(rows[0]["text"], prose)
        self.assertEqual(text_hash(rows[0]["text"]), load_json(run / "summary.json")["output_sha256"])

    def test_benchmark_records_every_strategy_with_equal_caps_and_measured_usage(self):
        config, configured = self.config()
        output = self.root / "benchmark"
        status, stdout, stderr = self.invoke(
            "benchmark", "--tasks", EXAMPLES / "english.json", "--config", config,
            "--samples", "2", "--output", output,
        )
        self.assertEqual(status, 0, stderr)
        self.assertEqual(json.loads(stdout)["runs"], 6)
        overview = load_json(output / "benchmark.json")
        self.assertIn("not equal realized compute", overview["note"])
        self.assertEqual(len(overview["results"]), 6)
        caps = (
            "max_context_tokens", "max_run_tokens", "max_output_tokens", "max_calls",
            "plan_tokens", "draft_tokens", "review_tokens", "selection_tokens",
            "max_repairs", "schema_retries",
        )
        strategies = set()
        calls_by_strategy = {}
        for row in overview["results"]:
            run_dir = output / row["run"]
            actual_config = load_json(run_dir / "config.json")
            strategies.add(actual_config["strategy"])
            for key in caps:
                self.assertEqual(actual_config[key], configured[key], key)
            expected_seed = 1007 if row["run"].endswith("s001") else 7
            self.assertEqual(actual_config["seed"], expected_seed)
            events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
            responses = [event["response"] for event in events if event["kind"] == "response"]
            self.assertEqual(row["tokens"]["input_tokens"], sum(r["input_tokens"] for r in responses))
            self.assertEqual(row["tokens"]["output_tokens"], sum(r["output_tokens"] for r in responses))
            self.assertEqual(row["tokens"]["accounting"], ["fixture_estimate"])
            self.assertGreaterEqual(row["elapsed_seconds"], 0)
            calls_by_strategy[row["strategy"]] = row["tokens"]["calls"]
        self.assertEqual(strategies, {"direct", "single_plan", "multi_plan"})
        self.assertLess(calls_by_strategy["direct"], calls_by_strategy["single_plan"])
        self.assertLess(calls_by_strategy["single_plan"], calls_by_strategy["multi_plan"])

    def test_benchmark_retains_failed_strategies_and_returns_nonzero(self):
        config, _ = self.config()
        output = self.root / "benchmark_failures"
        with patch("creativity_pipeline.cli._backend", return_value=FailDirectBackend()):
            status, stdout, _ = self.invoke(
                "benchmark", "--tasks", EXAMPLES / "english.json", "--config", config, "--output", output,
            )
        self.assertEqual(status, 2)
        self.assertEqual(json.loads(stdout)["runs"], 3)
        rows = load_json(output / "benchmark.json")["results"]
        by_strategy = {row["strategy"]: row for row in rows}
        self.assertEqual(by_strategy["direct"]["status"], "failed")
        self.assertGreater(by_strategy["direct"]["tokens"]["output_tokens"], 0)
        self.assertEqual(by_strategy["single_plan"]["status"], "accepted_by_checks")
        self.assertEqual(by_strategy["multi_plan"]["status"], "accepted_by_checks")

    def test_repeated_benchmark_strategy_is_rejected_before_creating_output(self):
        config, _ = self.config()
        output = self.root / "duplicate_strategy"
        with patch("creativity_pipeline.cli._backend", return_value=DemoBackend()) as factory:
            status, _, stderr = self.invoke(
                "benchmark", "--tasks", EXAMPLES / "english.json", "--config", config,
                "--strategies", "direct", "direct", "--output", output,
            )
        self.assertEqual(status, 2)
        self.assertTrue("duplicate" in stderr.lower() or "unique" in stderr.lower())
        self.assertFalse(output.exists())
        factory.assert_not_called()

    def test_detector_export_rejects_either_fixture_marker_on_runs(self):
        for index, (fixture, backend_fixture) in enumerate(((True, False), (False, True))):
            with self.subTest(fixture=fixture, backend_fixture=backend_fixture):
                run = self.make_run(f"fixture{index}", fixture=fixture, backend_fixture=backend_fixture)
                manifest, sheet = self.root / f"manifest{index}.json", self.root / f"sheet{index}.csv"
                status, _, stderr = self.invoke("export-detectors", run, "--manifest", manifest, "--csv", sheet)
                self.assertEqual(status, 2)
                self.assertIn("fixture", stderr.lower())
                self.assertFalse(manifest.exists())
                self.assertFalse(sheet.exists())

    def test_detector_export_refuses_changed_final_text(self):
        run = self.make_run("modified", text="Original test text.")
        (run / "final.txt").write_text("Original test text.\n")
        manifest, sheet = self.root / "manifest.json", self.root / "sheet.csv"
        status, _, stderr = self.invoke("export-detectors", run, "--manifest", manifest, "--csv", sheet)
        self.assertEqual(status, 2)
        self.assertIn("changed", stderr.lower())
        self.assertFalse(manifest.exists())
        self.assertFalse(sheet.exists())

    def test_detector_export_keeps_failed_runs_and_explicit_human_controls(self):
        self.make_run("success", text="Designated temporary test prose.")
        self.make_run("failure", text=None, status="budget_exhausted")
        controls = self.root / "controls.json"
        atomic_json(controls, [{
            "output_id": "human_control_test_only", "task_id": "control_task",
            "mode": "english", "baseline": "human_control", "origin": "human",
            "text": "Temporary control record used only by this unit test.", "testing_only": True,
        }])
        manifest, sheet = self.root / "manifest.json", self.root / "sheet.csv"
        status, stdout, stderr = self.invoke(
            "export-detectors", self.root / "runs", "--records", controls,
            "--manifest", manifest, "--csv", sheet, "--detector", "Pangram", "--version", "4.1",
        )
        self.assertEqual(status, 0, stderr)
        self.assertEqual(json.loads(stdout)["designated_outputs"], 3)
        rows = load_json(manifest)
        failed = next(row for row in rows if row["task_id"] == "failure")
        self.assertEqual(failed["status"], "budget_exhausted")
        self.assertEqual(failed["text"], "")
        self.assertEqual(failed["sha256"], text_hash(""))
        self.assertEqual(failed["tokens"]["total_tokens"], 30)
        self.assertEqual(sum(row["origin"] == "human" for row in rows), 1)
        with sheet.open(newline="") as stream:
            scans = list(csv.DictReader(stream))
        self.assertEqual(len(scans), 3)
        self.assertTrue(all(row["version"] == "4.1" and row["label"] == "" for row in scans))

    def test_additional_detector_records_reject_fixtures_and_nonobjects_cleanly(self):
        base = {
            "output_id": "test", "task_id": "test", "mode": "english",
            "baseline": "test", "origin": "model", "text": "Test record.",
        }
        cases = [
            {**base, "is_fixture": True}, {**base, "backend": {"fixture": True}}, 123,
        ]
        for index, record in enumerate(cases):
            with self.subTest(record=record):
                records = self.root / f"records{index}.json"
                atomic_json(records, [record])
                manifest, sheet = self.root / f"extra{index}.json", self.root / f"extra{index}.csv"
                status, _, stderr = self.invoke("export-detectors", "--records", records, "--manifest", manifest, "--csv", sheet)
                self.assertEqual(status, 2)
                self.assertTrue(stderr)
                self.assertFalse(manifest.exists())
                self.assertFalse(sheet.exists())

    def test_detector_export_never_overwrites_an_existing_sheet(self):
        run = self.make_run("sheet_preservation")
        manifest, sheet = self.root / "manifest.json", self.root / "sheet.csv"
        sheet.write_text("existing manual observations\n")
        status, _, _ = self.invoke("export-detectors", run, "--manifest", manifest, "--csv", sheet)
        self.assertEqual(status, 2)
        self.assertEqual(sheet.read_text(), "existing manual observations\n")
        self.assertFalse(manifest.exists())

    def test_training_export_preserves_exact_prompt_and_json_and_stays_unaccepted(self):
        run = self.make_run("training_proposals")
        saved = self.write_training_events(run)
        output = self.root / "proposals.jsonl"
        status, stdout, stderr = self.invoke(
            "export-training", run, "--group-id", "source_work_A", "--approach-id", "pending_review",
            "--split", "train", "--roles", "plan", "draft", "repair", "--output", output,
        )
        self.assertEqual(status, 0, stderr)
        report = json.loads(stdout)
        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["accepted_rows"], 0)
        self.assertEqual(report["skipped_thinking_requests"], 1)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        for row, call_id in zip(rows, (1, 4)):
            self.assertEqual(row["messages"], saved[call_id]["request"]["messages"])
            self.assertEqual(row["completion"], saved[call_id]["response"]["text"])
            self.assertIs(row["accepted"], False)
            self.assertEqual(row["split"], "train")
            self.assertEqual(row["group_id"], "source_work_A")
            self.assertIn("UNCURATED", row["provenance"])
        exported = output.read_text()
        self.assertNotIn("review-only marker", exported)
        self.assertNotIn("private test trace", exported)

    def test_training_export_refuses_fixtures(self):
        for index, (fixture, backend_fixture) in enumerate(((True, False), (False, True))):
            with self.subTest(fixture=fixture, backend_fixture=backend_fixture):
                run = self.make_run(f"training_fixture{index}", fixture=fixture, backend_fixture=backend_fixture)
                self.write_training_events(run)
                output = self.root / f"proposals{index}.jsonl"
                status, _, stderr = self.invoke("export-training", run, "--group-id", "g", "--split", "train", "--output", output)
                self.assertEqual(status, 2)
                self.assertIn("fixture", stderr.lower())
                self.assertFalse(output.exists())

    def test_training_export_never_overwrites_previous_curation(self):
        run = self.make_run("training_preservation")
        self.write_training_events(run)
        output = self.root / "curated.jsonl"
        output.write_text('{"preserve":"human curation"}\n')
        status, _, _ = self.invoke("export-training", run, "--group-id", "g", "--split", "train", "--output", output)
        self.assertEqual(status, 2)
        self.assertEqual(output.read_text(), '{"preserve":"human curation"}\n')

    def test_training_export_no_eligible_proposals_has_error_exit_and_no_file(self):
        run = self.make_run("no_eligible_proposals")
        self.write_training_events(run, rows=[("review_draft", False, '{"decision":"accept","issues":[]}', "stop", {})])
        output = self.root / "none.jsonl"
        status, _, stderr = self.invoke("export-training", run, "--group-id", "g", "--split", "train", "--output", output)
        self.assertEqual(status, 2)
        self.assertIn("No complete", stderr)
        self.assertFalse(output.exists())

    def test_history_state_input_is_rejected_before_backend_creation(self):
        state = self.root / "state.json"
        atomic_json(state, {"world": [], "reader": []})
        output = self.root / "history_state_rejected"
        with patch("creativity_pipeline.cli._backend") as factory:
            status, _, stderr = self.invoke(
                "generate", EXAMPLES / "history.json", "--state-in", state, "--output", output,
            )
        self.assertEqual(status, 2)
        self.assertIn("English fiction only", stderr)
        factory.assert_not_called()
        self.assertFalse(output.exists())

    def test_missing_input_is_reported_as_cli_error_without_traceback(self):
        status, stdout, stderr = self.invoke("generate", self.root / "absent.json", "--output", self.root / "unused")
        self.assertEqual(status, 2)
        self.assertEqual(stdout, "")
        self.assertIn("FileNotFoundError", stderr)
        self.assertNotIn("Traceback", stderr)


if __name__ == "__main__":
    unittest.main()
