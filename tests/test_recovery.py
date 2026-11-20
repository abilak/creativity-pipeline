"""Saved-draft recovery must preserve evidence, provenance, and fresh gates."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from creativity_pipeline import cli
from creativity_pipeline.io import atomic_json, load_json, text_hash
from creativity_pipeline.pipeline import Pipeline
from creativity_pipeline.recovery import load_saved_run
from test_pipeline import (Reply, ScriptedBackend, draft, english_task, history_task,
                           historical_draft, review, settings, valid_plan)
from test_pipeline_critic import report


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def parent(self, *, task=None, item=None, config=None, fixture=False, plan=None):
        task, item = task or english_task(), item or draft()
        config = config or settings(semantic_critic=True, critic_tokens=1024)
        parent = self.root / "parent"
        parent.mkdir()
        atomic_json(parent / "task.json", task.model_dump())
        atomic_json(parent / "config.json", config.model_dump())
        atomic_json(parent / "final.json", item)
        (parent / "final.txt").write_bytes(item["text"].encode("utf-8"))
        atomic_json(parent / "summary.json", {
            "pipeline_version": "0.3.0", "run_id": "earlier-run", "task_id": task.task_id,
            "mode": task.mode, "strategy": "single_plan", "status": "failed",
            "output_sha256": text_hash(item["text"]), "is_fixture": fixture,
            "backend": {"backend": "test_only", "fixture": fixture},
            "tokens": {"calls": 5, "input_tokens": 8827, "output_tokens": 3472},
            "elapsed_seconds": 3795.34, "final_critic": {"decision": "accept"},
            "critic_reviewed_output_sha256": text_hash(item["text"]),
        })
        if plan:
            atomic_json(parent / "selected_plan.json", plan)
        return parent, task, item, config

    def test_invalid_annotations_are_repaired_without_planning_or_redrafting(self):
        task, bad, good = history_task(), historical_draft(), historical_draft()
        bad["claims"][0]["text"] = "Paraphrased text absent from the draft."
        bad["claims"][0]["support_quotes"][0]["source_id"] = "S7"
        parent, task, _, config = self.parent(task=task, item=bad, plan=valid_plan(history=True))
        manifest = {"selected_source_ids": ["S1"], "n_omitted_passages": 7}
        atomic_json(parent / "source_selection.json", manifest)
        before = {p.name: p.read_bytes() for p in parent.iterdir()}
        saved = load_saved_run(parent)
        backend = ScriptedBackend([
            Reply("repair", good), Reply("semantic_critic", report(task, good["text"])),
            Reply("review_draft", review()),
        ])
        output = self.root / "continued"
        result = Pipeline(backend, config, output).run(task, saved_run=saved)
        self.assertEqual(result["status"], "accepted_by_checks", result["error"])
        self.assertEqual([r.role for r in backend.requests], ["repair", "semantic_critic", "review_draft"])
        self.assertEqual(result["strategy"], "repair_run")
        self.assertEqual(result["accepted_plan_count"], 0)
        self.assertTrue(result["reused_selected_plan"])
        self.assertEqual(result["tokens"]["calls"], 3)
        self.assertEqual(result["continuation"]["parent_tokens"]["calls"], 5)
        self.assertEqual(load_json(output / "source_selection.json"), manifest)
        self.assertEqual(load_json(output / "input_draft.json")["claims"][0]["support_quotes"][0]["source_id"], "S7")
        self.assertEqual(before, {p.name: p.read_bytes() for p in parent.iterdir()})
        self.assertEqual(backend.requests[0].metadata["task"]["sources"], task.model_dump()["sources"])

    def test_old_critic_approval_is_discarded_and_failed_new_critic_does_not_commit_state(self):
        parent, task, item, config = self.parent()
        # A final state file is not a new initial state for this same draft.
        atomic_json(parent / "state.json", {"world": [{"id": "UNTRUSTED", "text": "Prior draft state.", "evidence": ""}], "reader": []})
        backend = ScriptedBackend([Reply("semantic_critic", "bad JSON")])
        output = self.root / "continued"
        result = Pipeline(backend, config, output).run(task, saved_run=load_saved_run(parent))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["output_sha256"], text_hash(item["text"]))
        self.assertIsNone(result["final_critic"])
        self.assertIsNone(result["critic_reviewed_output_sha256"])
        self.assertEqual(load_json(output / "state.json"), task.initial_state.model_dump())
        self.assertEqual([r.role for r in backend.requests], ["semantic_critic"])

    def test_changed_final_json_is_rejected_before_backend_construction(self):
        parent, _, item, _ = self.parent()
        atomic_json(parent / "final.json", {**item, "text": "An edited draft."})
        with patch("creativity_pipeline.cli._backend") as make_backend:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = cli.main(["repair-run", str(parent), "--output", str(self.root / "new")])
        self.assertEqual(status, 2)
        make_backend.assert_not_called()
        self.assertFalse((self.root / "new").exists())

    def test_changed_text_export_is_rejected(self):
        parent, _, _, _ = self.parent()
        (parent / "final.txt").write_text("Different exported prose.")
        with self.assertRaisesRegex(ValueError, "final.txt and final.json disagree"):
            load_saved_run(parent)

    def test_task_identity_mismatch_is_rejected(self):
        parent, _, _, _ = self.parent()
        task = load_json(parent / "task.json")
        task["task_id"] = "other-task"
        atomic_json(parent / "task.json", task)
        with self.assertRaisesRegex(ValueError, "identity disagrees"):
            load_saved_run(parent)

    def test_invalid_saved_plan_cannot_be_reused(self):
        parent, _, _, _ = self.parent(task=history_task(), item=historical_draft(), plan=valid_plan(history=True))
        plan = load_json(parent / "selected_plan.json")
        plan["moves"][0]["source_ids"] = ["S8"]
        atomic_json(parent / "selected_plan.json", plan)
        with self.assertRaisesRegex(ValueError, "plan fails checks"):
            load_saved_run(parent)

    def test_caller_cannot_change_sources_after_loading_a_saved_run(self):
        parent, task, _, config = self.parent(task=history_task(), item=historical_draft())
        saved = load_saved_run(parent)
        changed = task.model_copy(update={"prompt": "A different task"})
        with self.assertRaisesRegex(ValueError, "exact saved task"):
            Pipeline(ScriptedBackend([]), config, self.root / "new").run(changed, saved_run=saved)
        self.assertFalse((self.root / "new").exists())

    def test_in_place_source_or_annotation_changes_invalidate_the_loaded_snapshot(self):
        parent, _, _, config = self.parent(task=history_task(), item=historical_draft())
        for name in ("source", "annotation", "provenance"):
            with self.subTest(name=name):
                saved = load_saved_run(parent)
                if name == "source":
                    saved.task.sources[0].text = "Changed evidence."
                elif name == "annotation":
                    saved.draft.claims[0].source_ids.append("S99")
                else:
                    saved.provenance["parent_is_fixture"] = True
                output = self.root / name
                with self.assertRaisesRegex(ValueError, "changed after loading"):
                    Pipeline(ScriptedBackend([]), config, output).run(saved.task, saved_run=saved)
                self.assertFalse(output.exists())

    def test_conflicting_manifest_is_rejected_before_creating_an_output(self):
        parent, task, _, config = self.parent(task=history_task(), item=historical_draft())
        atomic_json(parent / "source_selection.json", {"selected_source_ids": ["S1"]})
        saved = load_saved_run(parent)
        output = self.root / "new"
        with self.assertRaisesRegex(ValueError, "exact saved source-selection"):
            Pipeline(ScriptedBackend([]), config, output,
                     source_manifest={"selected_source_ids": ["S9"]}).run(task, saved_run=saved)
        self.assertFalse(output.exists())

    def test_cli_inherits_saved_caps_and_refuses_to_overwrite_parent(self):
        config = settings(semantic_critic=False, max_repairs=0, max_calls=1)
        parent, _, _, _ = self.parent(config=config)
        backend = ScriptedBackend([Reply("review_draft", review())])
        with patch("creativity_pipeline.cli._backend", return_value=backend) as factory:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = cli.main(["repair-run", str(parent), "--output", str(self.root / "new")])
        self.assertEqual(status, 0)
        self.assertEqual(factory.call_args.args[0].max_calls, 1)
        self.assertEqual([r.role for r in backend.requests], ["review_draft"])
        before = {p.name: p.read_bytes() for p in parent.iterdir()}
        with patch("creativity_pipeline.cli._backend", return_value=ScriptedBackend([])):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = cli.main(["repair-run", str(parent), "--output", str(parent)])
        self.assertEqual(status, 2)
        self.assertEqual(before, {p.name: p.read_bytes() for p in parent.iterdir()})

    def test_fixture_origin_survives_a_nonfixture_review_backend(self):
        parent, task, _, _ = self.parent(fixture=True)
        backend = ScriptedBackend([Reply("review_draft", review())])
        config = settings(backend="transformers", semantic_critic=False)
        output = self.root / "continued"
        result = Pipeline(backend, config, output).run(task, saved_run=load_saved_run(parent))
        self.assertEqual(result["status"], "accepted_by_checks")
        self.assertTrue(result["is_fixture"])
        with self.assertRaisesRegex(ValueError, "demo fixtures"):
            cli._run_records([output])

    def test_export_keeps_continuation_separate_from_fresh_generation(self):
        parent, task, _, _ = self.parent()
        backend = ScriptedBackend([Reply("review_draft", review())])
        config = settings(backend="transformers", semantic_critic=False)
        output = self.root / "continued"
        Pipeline(backend, config, output).run(task, saved_run=load_saved_run(parent))
        # ScriptedBackend is expressly a test fixture; simulate an ordinary
        # saved backend record only for checking export provenance.
        summary = load_json(output / "summary.json")
        summary["backend"]["fixture"] = False
        atomic_json(output / "summary.json", summary)
        row = cli._run_records([output])[0]
        self.assertEqual(row["baseline"], "repair_run")
        self.assertEqual(row["continuation"]["parent_run_id"], "earlier-run")
        self.assertEqual(row["configuration"]["continuation_parent_condition_id"],
                         row["continuation"]["parent_condition"]["condition_id"])


if __name__ == "__main__":
    unittest.main()
