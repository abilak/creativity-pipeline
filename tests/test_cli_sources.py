"""Source-library CLI integration uses real SQLite and deterministic fixtures."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from creativity_pipeline import cli
from creativity_pipeline.io import atomic_json, load_json, text_hash
from creativity_pipeline.models import Settings


class SourceCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.records = [
            {"id": "speech", "title": "Invented council speech", "text": "The council opposed women holding office.",
             "author": "A fictional councillor", "date": "1840", "locator": "Invented paragraph 1", "independence_group": "speech"},
            {"id": "ledger", "title": "Invented ledger", "text": "Women kept the council accounts accurately.",
             "author": "A fictional clerk", "date": "1841", "independence_group": "ledger"},
            {"id": "weather", "title": "Invented weather note", "text": "Rain fell on Monday."},
        ]
        self.input = self.root / "sources.jsonl"
        self.input.write_text("".join(json.dumps(row) + "\n" for row in self.records), encoding="utf-8")
        self.library = self.root / "library.sqlite"

    def invoke(self, *arguments):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            status = cli.main([str(value) for value in arguments])
        return status, output.getvalue(), error.getvalue()

    def build(self):
        status, output, error = self.invoke("build-sources", self.input, "--output", self.library)
        self.assertEqual(status, 0, error)
        self.assertEqual(json.loads(output)["n_documents"], 3)

    def test_build_retrieve_quote_and_no_overwrite(self):
        self.build()
        output = self.root / "packet.json"
        status, _, error = self.invoke("retrieve-sources", self.library, "--query", "council women",
                                      "--counter-query", "accounts accurately", "--output", output)
        self.assertEqual(status, 0, error)
        packet = load_json(output)
        self.assertEqual(packet["manifest"]["n_selected_documents"], 2)
        self.assertEqual(packet["manifest"]["n_omitted_documents"], 1)
        source = packet["sources"][0]
        status, stdout, error = self.invoke("verify-source-quote", self.library, "--source-id", source["id"],
                                           "--quote", source["text"])
        self.assertEqual(status, 0, error)
        self.assertTrue(json.loads(stdout)["valid"])
        self.assertFalse(json.loads(stdout)["establishes_truth_or_entailment"])
        status, _, _ = self.invoke("build-sources", self.input, "--output", self.library)
        self.assertEqual(status, 2)
        status, _, _ = self.invoke("retrieve-sources", self.library, "--query", "council", "--output", output)
        self.assertEqual(status, 2)

    def test_generation_merges_inline_sources_and_records_scope_references(self):
        self.build()
        inline = {"id": "INLINE", "title": "User supplied metadata", "text": "An additional council notice survives.",
                  "author": "A fictional archivist", "date": "1842", "url": "", "locator": "Practice packet p. 2"}
        task = self.root / "task.json"
        atomic_json(task, {"task_id": "library_test", "mode": "history", "prompt": "Assess women and the council.",
                           "sources": [inline], "source_scope": "All material is invented for testing."})
        config = self.root / "config.yaml"
        config.write_text("backend: demo\nstrategy: direct\nsemantic_critic: false\nschema_retries: 0\n", encoding="utf-8")
        output = self.root / "generated"
        status, stdout, error = self.invoke("generate", task, "--config", config, "--output", output,
                                           "--source-library", self.library, "--source-query", "council women")
        self.assertEqual(status, 0, error + stdout)
        saved = load_json(output / "task.json")
        self.assertEqual(len(saved["sources"]), 3)
        self.assertEqual(saved["sources"][0]["id"], "INLINE")
        self.assertIn("invented", saved["source_scope"])
        self.assertIn("content outside it was not assessed", saved["source_scope"])
        self.assertIn("may also contain material omitted by retrieval", saved["source_scope"])
        self.assertTrue((output / "source_selection.json").exists())
        references = load_json(output / "references.json")
        self.assertEqual([row["id"] for row in references["references"]], ["INLINE"])
        self.assertEqual(references["references"][0]["locator"], inline["locator"])
        self.assertEqual(references["references"][0]["passage_sha256"], text_hash(inline["text"]))
        self.assertFalse(references["external_authenticity_verified"])
        self.assertEqual(len(references["supplied_but_not_cited"]), 2)

    def test_library_can_supply_missing_inline_packet_and_keeps_original_text(self):
        self.build()
        task_path = self.root / "task.json"
        atomic_json(task_path, {"task_id": "retrieval", "mode": "history", "prompt": "Assess council records."})
        args = cli.parser().parse_args(["generate", str(task_path), "--source-library", str(self.library),
                                        "--output", str(self.root / "out")])
        task, manifest = cli._prepared_task(args)
        self.assertGreater(len(task.sources), 0)
        self.assertEqual(len(task.sources), manifest["n_selected_passages"])
        original = "  Verbatim text.\r\n"
        manual_packet = {"sources": [{"id": "S1", "title": "Verbatim", "text": original}], "manifest": manifest}
        self.assertEqual(cli._task(task_path, source_packet=manual_packet).sources[0].text, original)

    def test_conflicting_ids_and_english_retrieval_are_rejected(self):
        self.build()
        task_path = self.root / "task.json"
        atomic_json(task_path, {"task_id": "english", "mode": "english", "prompt": "A fictional scene."})
        status, _, error = self.invoke("generate", task_path, "--source-library", self.library,
                                      "--output", self.root / "out")
        self.assertEqual(status, 2)
        self.assertIn("history", error)
        atomic_json(task_path, {"task_id": "history", "mode": "history", "prompt": "Assess records.",
                               "sources": [{"id": "S1", "title": "A", "text": "One."}]})
        packet = {"sources": [{"id": "S1", "title": "B", "text": "Another."}], "manifest": {}}
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            cli._task(task_path, source_packet=packet)

    def test_backend_factory_routes_without_loading_optional_dependencies(self):
        shared = cli._backend(Settings())
        self.assertIs(shared.writer, shared.critic)
        self.assertEqual(shared.describe()["critic_mode"], "shared_backbone_adapter_disabled")
        separate = cli._backend(Settings(critic_model_id="test/critic", critic_revision="commit-test",
                                         critic_max_context_tokens=4096))
        self.assertIsNot(separate.writer, separate.critic)
        self.assertEqual(separate.critic.model_id, "test/critic")
        self.assertEqual(separate.critic.revision, "commit-test")
        self.assertIsNone(separate.critic.adapter_path)
        self.assertFalse(separate.describe()["loaded"])


if __name__ == "__main__":
    unittest.main()
