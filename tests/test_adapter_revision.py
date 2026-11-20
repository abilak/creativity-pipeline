"""Adapter provenance must agree before budget accounting or model execution."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from creativity_pipeline.backends import BackendError, _adapter_payload_sha256
from test_backends import COMMIT, cpu_backend, make_runtime, request, write_adapter_metadata


class AdapterRevisionTests(unittest.TestCase):
    def setUp(self):
        self.runtime = make_runtime()
        self.runtime.patcher.start()
        self.addCleanup(self.runtime.patcher.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.records = write_adapter_metadata(self.directory, self.runtime.tokenizer)

    def update_metadata(self, name, **changes):
        self.records[name].update(changes)
        (self.directory / name).write_text(json.dumps(self.records[name]), encoding="utf-8")

    def test_default_main_adopts_adapter_pin_before_counting_without_model_weights(self):
        backend = cpu_backend(adapter_path=str(self.directory))
        self.assertGreater(backend.count_input_tokens(request()), 0)
        record = self.runtime.record
        self.assertEqual(record.config_loads[0][1]["revision"], COMMIT)
        self.assertEqual(record.tokenizer_loads[0][1]["revision"], COMMIT)
        self.assertEqual(record.model_loads, [])
        self.assertNotIn("torch", record.imports)
        self.assertEqual(backend.describe()["adapter_base_revision"], COMMIT)
        self.assertTrue(backend.describe()["adapter_template_verified"])

    def test_explicit_different_sha_fails_before_config_or_tokenizer_load(self):
        backend = cpu_backend(adapter_path=str(self.directory), revision="b" * 40)
        with self.assertRaisesRegex(BackendError, "Explicit model revision conflicts"):
            backend.count_input_tokens(request())
        self.assertEqual(self.runtime.record.imports, [])
        self.assertEqual(self.runtime.record.config_loads, [])

    def test_explicit_tag_is_resolved_once_and_must_match_adapter_pin(self):
        backend = cpu_backend(adapter_path=str(self.directory), revision="release-tag")
        backend.count_input_tokens(request())
        backend.generate(request())
        backend.count_input_tokens(request())
        record = self.runtime.record
        self.assertEqual(len(record.config_loads), 1)
        self.assertEqual(record.config_loads[0][1]["revision"], "release-tag")
        self.assertEqual(record.tokenizer_loads[0][1]["revision"], COMMIT)
        self.assertEqual(record.model_loads[0][1]["revision"], COMMIT)
        self.assertIs(record.model_loads[0][1]["config"], self.runtime.model.config)

    def test_explicit_tag_resolving_elsewhere_fails_before_tokenizer_load(self):
        self.runtime.model.config._commit_hash = "b" * 40
        backend = cpu_backend(adapter_path=str(self.directory), revision="wrong-tag")
        with self.assertRaisesRegex(BackendError, "resolves to a different commit"):
            backend.count_input_tokens(request())
        self.assertEqual(self.runtime.record.tokenizer_loads, [])
        self.assertEqual(self.runtime.record.model_loads, [])

    def test_missing_or_unfinished_training_metadata_is_rejected(self):
        (self.directory / "training_report.json").unlink()
        with self.assertRaisesRegex(BackendError, "readable training_report.json"):
            cpu_backend(adapter_path=str(self.directory)).count_input_tokens(request())
        self.update_metadata("training_report.json", status="training")
        with self.assertRaisesRegex(BackendError, "completed run"):
            cpu_backend(adapter_path=str(self.directory)).count_input_tokens(request())
        self.assertEqual(self.runtime.record.config_loads, [])

    def test_model_and_revision_metadata_must_agree_before_token_count(self):
        self.update_metadata("training_config.json", model_id={"wrong": "type"})
        with self.assertRaisesRegex(BackendError, "model metadata disagrees"):
            cpu_backend(adapter_path=str(self.directory)).count_input_tokens(request())
        self.update_metadata("training_config.json", model_id="a-local-tiny-model",
                             resolved_model_revision="b" * 40)
        with self.assertRaisesRegex(BackendError, "one immutable base-model commit"):
            cpu_backend(adapter_path=str(self.directory)).count_input_tokens(request())
        self.assertEqual(self.runtime.record.config_loads, [])

    def test_changed_template_is_rejected_before_tokenization_and_weights(self):
        self.update_metadata("training_report.json", chat_template_sha256="0" * 64)
        with self.assertRaisesRegex(BackendError, "chat-template hash differs"):
            cpu_backend(adapter_path=str(self.directory)).count_input_tokens(request())
        self.assertEqual(self.runtime.tokenizer.tokenize_calls, [])
        self.assertEqual(self.runtime.record.model_loads, [])

    def test_model_pin_does_not_depend_on_tokenizer_init_kwargs(self):
        self.runtime.tokenizer.init_kwargs = {}
        backend = cpu_backend()
        backend.count_input_tokens(request())
        backend.generate(request())
        self.assertEqual(self.runtime.record.model_loads[0][1]["revision"], COMMIT)
        self.assertEqual(len(self.runtime.record.config_loads), 1)

    def test_conflicting_tokenizer_commit_is_rejected_before_token_count(self):
        self.runtime.tokenizer.init_kwargs = {"_commit_hash": "b" * 40}
        with self.assertRaisesRegex(BackendError, "Tokenizer commit does not match"):
            cpu_backend().count_input_tokens(request())
        self.assertEqual(self.runtime.tokenizer.tokenize_calls, [])
        self.assertEqual(self.runtime.record.model_loads, [])

    def test_loaded_model_must_match_previously_resolved_config(self):
        backend = cpu_backend()
        backend.count_input_tokens(request())
        self.runtime.model.config = SimpleNamespace(_commit_hash="b" * 40, max_position_embeddings=4096)
        with self.assertRaisesRegex(BackendError, "Loaded model commit differs"):
            backend.generate(request())
        self.assertEqual(self.runtime.model.calls, [])

    def test_partial_adapter_is_rejected_before_it_can_generate(self):
        self.runtime.model.adapter_load_result.missing_keys = ["layer.lora_B.default.weight"]
        backend = cpu_backend(adapter_path=str(self.directory))
        with self.assertRaisesRegex(BackendError, "tensor inventory.*missing_keys"):
            backend.generate(request())
        self.assertFalse(backend.describe()["adapter_loaded"])
        self.assertIsNone(backend.describe()["adapter_sha256"])
        self.assertEqual(self.runtime.model.calls, [])

    def test_adapter_identity_follows_payload_not_path_or_config_whitespace(self):
        first = _adapter_payload_sha256(str(self.directory))
        with tempfile.TemporaryDirectory() as other:
            write_adapter_metadata(other, self.runtime.tokenizer)
            config = self.records["adapter_config.json"]
            (Path(other) / "adapter_config.json").write_text(
                json.dumps(dict(reversed(list(config.items()))), indent=4), encoding="utf-8",
            )
            self.assertEqual(first, _adapter_payload_sha256(other))
            (Path(other) / "adapter_model.safetensors").write_bytes(b"a-different-fake-adapter")
            self.assertNotEqual(first, _adapter_payload_sha256(other))
        backend = cpu_backend(adapter_path=str(self.directory))
        result = backend.generate(request())
        self.assertEqual(result.metadata["adapter_sha256"], first)
        self.assertEqual(backend.describe()["adapter_sha256"], first)

    def test_unsupported_adapter_weights_fail_before_base_weight_load(self):
        (self.directory / "adapter_model.safetensors").rename(self.directory / "adapter_model.bin")
        with self.assertRaisesRegex(BackendError, "one adapter_model.safetensors"):
            cpu_backend(adapter_path=str(self.directory)).generate(request())
        self.assertEqual(self.runtime.record.model_loads, [])


if __name__ == "__main__":
    unittest.main()
