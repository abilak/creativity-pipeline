"""Saved-run diagnostics must be useful without a model or private text disclosure."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect_run.py"
spec = importlib.util.spec_from_file_location("run_diagnostics", SCRIPT)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class RunDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()

    def write(self, name, value, directory=None):
        path = (directory or self.run) / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def fixture(self, calls=5):
        secret = "PRIVATE_PROMPT_SOURCE_DRAFT_OR_RESPONSE_" * 40
        error = "1 validation error for CriticReport\nrubric.0.judgment\n  Input should be 'strong' [type=literal_error, input_value='" + secret + "', input_type=str]\n    For further information visit https://errors.pydantic.dev/2/v/literal_error"
        self.write("summary.json", {
            "pipeline_version": "0.3.0", "status": "failed", "error": "semantic_critic did not produce a valid complete CriticReport: " + error,
            "mode": "history", "strategy": "single_plan", "accepted_plan_count": 1,
            "draft_revision": 0, "repairs": 0, "word_count": 581, "elapsed_seconds": 3795.34,
            "tokens": {"calls": calls, "input_tokens": 8827, "output_tokens": 3472, "total_tokens": 12299, "unaccounted_calls": 0, "complete": True},
            "backend": {"backend": "sequential_critic_router", "model_id": "Qwen/Qwen3-8B", "device": "cuda", "load_in_4bit": True, "compute_dtype": "bfloat16", "attention_implementation": "sdpa", "critic_mode": "shared_backbone_adapter_disabled",
                        "cuda_device": {"index": 0, "name": "ACTUAL_RUN_GPU", "total_memory_bytes": 20 * 2 ** 30, "compute_capability": "8.9", "torch_cuda_version": "12.8"},
                        "packages": {"torch": "2.8.0", "transformers": "4.57.1", "bitsandbytes": "0.48.2", "peft": "0.17.1", "accelerate": "1.10.0", "safetensors": "0.5.3"}},
            "final_issues": [{"code": "unknown_quote_source", "message": secret}] * 2,
            "final_critic": {"private": secret},
        })
        self.write("task.json", {"mode": "history", "prompt": secret, "sources": [{"id": "S1", "text": secret}]})
        self.write("config.json", {"semantic_critic": True, "critic_model_id": None, "max_context_tokens": 8192, "critic_max_context_tokens": None, "draft_tokens": 3072, "critic_tokens": 3072, "review_tokens": 768, "max_repairs": 2, "schema_retries": 1, "private": secret})
        events = []
        for index in range(1, calls + 1):
            role = ("plan", "review_plan", "draft", "semantic_critic", "semantic_critic")[(index - 1) % 5]
            events.extend([
                {"kind": "request", "call_id": index, "estimated_input_tokens": 1000,
                 "request": {"role": role, "max_new_tokens": 3072, "thinking": False, "messages": [{"content": secret}], "metadata": {"task": secret}}},
                {"kind": "response", "call_id": index, "response": {
                    "input_tokens": 1000, "output_tokens": 100, "finish_reason": "stop", "elapsed_seconds": 75.0, "text": secret,
                    "metadata": {"raw_output": secret, "complete": True, "incomplete_reason": None, "routing": {"backend_slot": "writer", "switched_model": False, "generation_seconds": 40.0, "preparation_seconds": 35.0, "unload_seconds": 0.0, "total_seconds": 75.0}}}},
            ])
            if index >= 4:
                events.append({"kind": "schema_error", "call_id": index, "error": error})
        (self.run / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        return secret

    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = diagnostics.main([str(arg) for arg in args])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_real_event_keys_privacy_and_timing_are_preserved(self):
        secret = self.fixture()
        status, stdout, stderr = self.invoke(self.run)
        self.assertEqual(status, 0, stderr)
        report = json.loads(stdout)
        self.assertNotIn(secret[:30], stdout)
        self.assertEqual(report["supplied_source_ids"], ["S1"])
        self.assertEqual(report["issue_counts"], {"unknown_quote_source": 2})
        self.assertEqual(report["summary"]["mode"], "history")
        self.assertEqual(report["backend"]["cuda_device"]["name"], "ACTUAL_RUN_GPU")
        self.assertEqual([row["role"] for row in report["calls"]], ["plan", "review_plan", "draft", "semantic_critic", "semantic_critic"])
        self.assertIn("rubric.0.judgment: literal_error", report["calls"][-1]["schema_error"])
        self.assertEqual(report["calls"][-1]["routing"]["generation_seconds"], 40.0)
        self.assertIn("not decode alone", report["timing_note"])
        self.assertLess(len(stdout), 6000)

    def test_works_without_site_packages_and_does_not_modify_run(self):
        self.fixture()
        before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.run.iterdir()}
        process = subprocess.run([sys.executable, "-I", "-S", str(SCRIPT), str(self.run)], capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["summary"]["status"], "failed")
        after = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.run.iterdir()}
        self.assertEqual(before, after)

    def test_last_calls_are_bounded_and_interrupted_events_are_reported(self):
        self.fixture(calls=19)
        with (self.run / "events.jsonl").open("ab") as stream:
            stream.write(b'{"kind": "response",')
        report = diagnostics.inspect_run(self.run)
        self.assertEqual([row["call_id"] for row in report["calls"]], list(range(8, 20)))
        self.assertEqual(report["call_rows_omitted"], 7)
        self.assertIn("Skipped 1 malformed event lines", report["warnings"])

    def test_corrupt_and_missing_files_still_produce_report(self):
        (self.run / "summary.json").write_text("not JSON", encoding="utf-8")
        self.write("task.json", ["wrong shape"])
        (self.run / "events.jsonl").write_bytes(b'\xff\n[]\n{"call_id": 1, "kind": "request", "request": []}\n')
        report = diagnostics.inspect_run(self.run)
        self.assertEqual(report["summary"], {})
        self.assertEqual(len(report["warnings"]), 4)
        self.assertEqual(report["calls"], [{"call_id": 1, "prompt_tokens": None}])

    def test_latest_explicitly_reports_chosen_assessment(self):
        self.fixture()
        assessment = self.root / "assessment"
        assessment.mkdir()
        path = self.write("assessment.json", {"status": "valid_report", "mode": "english"}, assessment)
        os.utime(self.run / "summary.json", ns=(1_000_000_000, 1_000_000_000))
        os.utime(path, ns=(2_000_000_000, 2_000_000_000))
        status, stdout, stderr = self.invoke("--latest", self.root)
        self.assertEqual(status, 0, stderr)
        report = json.loads(stdout)
        self.assertEqual(report["selected_run"], str(assessment.resolve()))
        self.assertEqual(report["summary_file"], "assessment.json")
        self.assertEqual(report["summary"]["mode"], "english")

    def test_errors_invalid_arguments_and_no_runs_are_clear(self):
        status, stdout, stderr = self.invoke("--latest", self.root)
        self.assertEqual(status, 2)
        self.assertIn("No summary.json", stderr)
        self.assertEqual(stdout, "")
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            diagnostics.main([str(self.run), "--latest", str(self.root)])
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            diagnostics.main([str(self.run), "--max-calls", "0"])

    def test_unknown_files_and_external_symlinks_are_not_read(self):
        self.fixture()
        outside = self.root / "external.json"
        outside.write_text('{"mode":"SECRET_EXTERNAL_CONTENT"}', encoding="utf-8")
        (self.run / "task.json").unlink()
        (self.run / "task.json").symlink_to(outside)
        (self.run / "final.txt").write_text("PRIVATE_DRAFT", encoding="utf-8")
        report = diagnostics.inspect_run(self.run)
        encoded = json.dumps(report)
        self.assertNotIn("SECRET_EXTERNAL_CONTENT", encoded)
        self.assertNotIn("PRIVATE_DRAFT", encoded)
        self.assertIn("task.json points outside this run; skipped", report["warnings"])

    def test_bad_metadata_and_nonfinite_values_cannot_break_json_output(self):
        self.write("summary.json", {"elapsed_seconds": float("inf"), "word_count": 10 ** 400,
                                   "backend": {"cuda_device": ["wrong"], "packages": {"torch": {"private": "x"}}},
                                   "tokens": [], "final_issues": [None, {"code": ["wrong"]}]})
        self.write("task.json", {"sources": "wrong"})
        self.write("config.json", {})
        status, stdout, stderr = self.invoke(self.run)
        self.assertEqual(status, 0, stderr)
        report = json.loads(stdout)
        self.assertIsNone(report["summary"]["elapsed_seconds"])
        self.assertIsNone(report["summary"]["word_count"])
        self.assertEqual(report["source_count"], 0)


if __name__ == "__main__":
    unittest.main()
