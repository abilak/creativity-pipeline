import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from creativity_pipeline.critic_backends import SequentialCriticBackend
from creativity_pipeline.interfaces import GenerationRequest, GenerationResult
from creativity_pipeline.evaluation import (
    aggregate_detector_results,
    condition_identity,
    condition_provenance,
    freeze_outputs,
    import_detector_csv,
    output_hash,
    write_detector_csv_template,
)


def records():
    rows = [
        {"output_id": "m1", "task_id": "e1", "mode": "english", "baseline": "plans", "origin": "model", "text": "A story.\n", "status": "accepted_by_checks"},
        {"output_id": "m2", "task_id": "e1", "mode": "english", "baseline": "plans", "origin": "model", "text": "", "status": "failed"},
        {"output_id": "h1", "task_id": "e1", "mode": "english", "baseline": "human_control", "origin": "human", "text": "A human story.", "status": "reference"},
        {"output_id": "t1", "task_id": "t1", "mode": "history", "baseline": "plans", "origin": "model", "text": "A historical interpretation.", "status": "needs_review"},
    ]
    for row in rows:
        row["condition_id"] = "human-controls" if row["origin"] == "human" else "model-base"
    return rows


def csv_file(manifest, entries, hash_field="sha256"):
    index = {row["output_id"]: row for row in manifest}
    fields = ["output_id", hash_field, "detector", "version", "query_date", "label", "classified_text_share", "confidence"]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for entry in entries:
        output_id = entry["output_id"]
        row = {"output_id": output_id, hash_field: index[output_id]["sha256"], "detector": "test-detector", "version": "4.1", "query_date": "2026-11-19", "label": "human"}
        row.update(entry)
        writer.writerow(row)
    stream.seek(0)
    return stream


class FrozenOutputsTest(unittest.TestCase):
    def test_fixture_markers_rejected_through_shared_export_and_import_apis(self):
        for marker in ({"is_fixture": True}, {"backend": {"fixture": True}}, {"metadata": {"backend": {"is_fixture": True}}}):
            with self.subTest(marker=marker):
                row = {**records()[0], **marker}
                with self.assertRaisesRegex(ValueError, "fixture"):
                    freeze_outputs([row])
                row["sha256"] = output_hash(row["text"])
                with self.assertRaisesRegex(ValueError, "fixture"):
                    aggregate_detector_results([], [row])

    def test_missing_condition_is_unresolved_and_does_not_pool_unknown_systems(self):
        rows = records()[:2]
        for row in rows:
            del row["condition_id"]
        manifest = freeze_outputs(rows)
        self.assertTrue(all(row["provenance_unresolved"] for row in manifest))
        self.assertNotEqual(manifest[0]["condition_id"], manifest[1]["condition_id"])
        self.assertEqual(freeze_outputs(manifest), manifest)
        report = aggregate_detector_results([], manifest)
        self.assertEqual(len(report), 2)
        self.assertTrue(all(row["provenance_unresolved"] for row in report))

    def test_hash_binds_exact_whitespace(self):
        manifest = freeze_outputs(records())
        self.assertEqual(manifest[0]["sha256"], output_hash("A story.\n"))
        self.assertNotEqual(manifest[0]["sha256"], output_hash("A story."))

    def test_explicit_origin_and_unique_id_required(self):
        rows = records()
        del rows[0]["origin"]
        with self.assertRaisesRegex(ValueError, "origin"):
            freeze_outputs(rows)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            freeze_outputs([records()[0], records()[0]])

    def test_existing_frozen_manifest_is_not_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            original = freeze_outputs(records(), path)
            self.assertEqual(freeze_outputs(records(), path), original)
            altered = records()
            altered[0]["text"] = "Different story."
            with self.assertRaisesRegex(ValueError, "different content"):
                freeze_outputs(altered, path)
            self.assertEqual(json.loads(path.read_text()), original)

    def test_frozen_manifest_tampering_is_detected_on_import(self):
        manifest = freeze_outputs(records())
        stream = csv_file(manifest, [{"output_id": "m1"}])
        manifest[0]["text"] += "tampered"
        with self.assertRaisesRegex(ValueError, "sha256"):
            import_detector_csv(stream, manifest)

    def test_template_contains_hashes_and_does_not_query_anything(self):
        manifest = freeze_outputs(records())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manual.csv"
            write_detector_csv_template(manifest, path)
            rows = list(csv.DictReader(io.StringIO(path.read_text())))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["sha256"], manifest[0]["sha256"])
            self.assertEqual(rows[0]["label"], "")


class DetectorImportTest(unittest.TestCase):
    def setUp(self):
        self.manifest = freeze_outputs(records())

    def test_valid_import_keeps_share_and_confidence_separate(self):
        stream = csv_file(self.manifest, [{"output_id": "m1", "label": "AI", "classified_text_share": "0.8", "confidence": "0.99"}])
        result = import_detector_csv(stream, self.manifest)[0]
        self.assertEqual(result["label"], "ai")
        self.assertEqual(result["classified_text_share"], 0.8)
        self.assertEqual(result["confidence"], 0.99)
        self.assertEqual(result["origin"], "model")

    def test_hash_mismatch_rejected(self):
        stream = csv_file(self.manifest, [{"output_id": "m1", "sha256": "0" * 64}])
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            import_detector_csv(stream, self.manifest)

    def test_version_and_date_are_required(self):
        for field in ("version", "query_date"):
            with self.subTest(field=field):
                stream = csv_file(self.manifest, [{"output_id": "m1", field: ""}])
                with self.assertRaisesRegex(ValueError, field):
                    import_detector_csv(stream, self.manifest)

    def test_invalid_dates_and_noncanonical_labels_rejected(self):
        for changes in ({"query_date": "2026-02-30"}, {"label": "probably generated"}):
            with self.subTest(changes=changes):
                stream = csv_file(self.manifest, [{"output_id": "m1", **changes}])
                with self.assertRaises(ValueError):
                    import_detector_csv(stream, self.manifest)

    def test_percentages_and_nan_not_silently_interpreted_as_fractions(self):
        for value in ("80", "nan", "inf", "-0.1"):
            with self.subTest(value=value):
                stream = csv_file(self.manifest, [{"output_id": "m1", "confidence": value}])
                with self.assertRaisesRegex(ValueError, "fraction"):
                    import_detector_csv(stream, self.manifest)

    def test_repeated_scan_cannot_double_count_a_new_date(self):
        stream = csv_file(self.manifest, [{"output_id": "m1"}, {"output_id": "m1", "query_date": "2026-11-20"}])
        with self.assertRaisesRegex(ValueError, "repeated scan"):
            import_detector_csv(stream, self.manifest)

    def test_output_sha256_header_alias(self):
        result = import_detector_csv(csv_file(self.manifest, [{"output_id": "m1"}], hash_field="output_sha256"), self.manifest)
        self.assertEqual(result[0]["sha256"], self.manifest[0]["sha256"])

    def test_duplicate_label_column_is_rejected_instead_of_silently_overridden(self):
        original = csv_file(self.manifest, [{"output_id": "m1", "label": "ai"}]).getvalue().splitlines()
        stream = io.StringIO(original[0] + ",label\n" + original[1] + ",human\n")
        with self.assertRaisesRegex(ValueError, "duplicate column"):
            import_detector_csv(stream, self.manifest)


class DetectorAggregationTest(unittest.TestCase):
    def test_empty_scans_keep_all_cohorts_and_failed_outputs(self):
        manifest = freeze_outputs(records())
        report = aggregate_detector_results([], manifest)
        self.assertEqual(len(report), 3)
        self.assertEqual(sum(row["n_designated"] for row in report), 4)
        self.assertEqual(sum(row["n_missing"] for row in report), 4)
        for row in report:
            self.assertIsNone(row["detector"])
            self.assertIsNone(row["version"])
            self.assertEqual(row["n_scanned"], 0)
            self.assertIsNone(row["label_rates_among_scanned"]["ai"])

    def test_distinct_model_conditions_are_not_pooled_under_one_strategy(self):
        rows = records()[:2]
        rows[1]["condition_id"] = "model-adapter"
        manifest = freeze_outputs(rows)
        scans = import_detector_csv(csv_file(manifest, [{"output_id": "m1"}]), manifest)
        report = aggregate_detector_results(scans, manifest)
        self.assertEqual({row["condition_id"] for row in report}, {"model-base", "model-adapter"})
        self.assertEqual([row["n_designated"] for row in report], [1, 1])
        self.assertEqual(sum(row["n_missing"] for row in report), 1)
        scans[0]["condition_id"] = "model-adapter"
        with self.assertRaisesRegex(ValueError, "condition_id conflicts"):
            aggregate_detector_results(scans, manifest)

    def test_missing_failed_outputs_remain_in_denominator(self):
        manifest = freeze_outputs(records())
        scans = import_detector_csv(csv_file(manifest, [{"output_id": "m1", "label": "mixed"}, {"output_id": "h1", "label": "ai"}]), manifest)
        result = aggregate_detector_results(scans, manifest)
        model = next(row for row in result if row["mode"] == "english" and row["origin"] == "model")
        self.assertEqual((model["n_designated"], model["n_scanned"], model["n_missing"]), (2, 1, 1))
        self.assertEqual(model["label_rates_among_scanned"]["mixed"], 1.0)
        self.assertEqual(model["label_rates_among_designated"]["mixed"], 0.5)
        self.assertEqual(model["designated_status_counts"]["failed"], 1)
        self.assertEqual(model["n_distinct_prompts"], 1)
        human = next(row for row in result if row["origin"] == "human")
        self.assertEqual(human["known_human_ai_label_count"], 1)
        history = next(row for row in result if row["mode"] == "history")
        self.assertEqual(history["n_missing"], 1)
        self.assertIsNone(history["label_rates_among_scanned"]["ai"])

    def test_versions_remain_separate(self):
        manifest = freeze_outputs(records())
        scans = import_detector_csv(csv_file(manifest, [{"output_id": "m1", "version": "4.0", "label": "ai"}, {"output_id": "m1", "version": "4.1", "label": "human"}]), manifest)
        result = aggregate_detector_results(scans, manifest)
        model = [row for row in result if row["mode"] == "english" and row["origin"] == "model"]
        self.assertEqual({row["version"] for row in model}, {"4.0", "4.1"})

    def test_human_origin_is_never_inferred_from_label_or_baseline(self):
        rows = records()
        rows[0]["baseline"] = "human_control"
        manifest = freeze_outputs(rows)
        scans = import_detector_csv(csv_file(manifest, [{"output_id": "m1", "label": "human"}]), manifest)
        result = aggregate_detector_results(scans, manifest)
        generated = next(row for row in result if row["baseline"] == "human_control" and row["origin"] == "model")
        self.assertNotIn("known_human_ai_label_count", generated)

    def test_aggregation_revalidates_duplicates_and_hashes(self):
        manifest = freeze_outputs(records())
        scans = import_detector_csv(csv_file(manifest, [{"output_id": "m1"}]), manifest)
        with self.assertRaisesRegex(ValueError, "double-count"):
            aggregate_detector_results(scans + scans, manifest)
        scans[0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            aggregate_detector_results(scans, manifest)


class ConditionIdentityTest(unittest.TestCase):
    def setUp(self):
        self.config = {"model_id": "example/model", "revision": "main", "strategy": "multi_plan", "seed": 42,
                       "draft_temperature": 0.7, "thinking_plan": False, "max_output_tokens": 4000,
                       "local_files_only": True, "adapter_path": None}
        self.backend = {"backend": "transformers", "model_id": "example/model", "resolved_commit": "a" * 40,
                        "requested_revision": "main", "template_sha256": "b" * 64, "adapter_path": None,
                        "load_in_4bit": True, "compute_dtype": "bfloat16"}

    def _critic_router(self, shared=False):
        class SnapshotBackend:
            def __init__(self, metadata):
                self.metadata = metadata

            def describe(self):
                return copy.deepcopy(self.metadata)

            def count_input_tokens(self, request):
                return 3

            def generate(self, request):
                self.metadata["loaded"] = True
                return GenerationResult(text="test", input_tokens=3, output_tokens=1)

            def unload(self):
                self.metadata["loaded"] = False
                return {"unloaded": True, "elapsed_seconds": 0.0}

        writer = SnapshotBackend({
            **self.backend, "loaded": False, "quantization": "nf4_double_quant",
            "attention_implementation": "sdpa", "effective_max_context_tokens": 8192,
            "adapter_path": "/writer/adapter", "adapter_sha256": "e" * 64,
            "packages": {"transformers": "4.57.1", "peft": "0.17.1"},
        })
        critic = None if shared else SnapshotBackend({
            **self.backend, "loaded": False, "model_id": "example/critic",
            "resolved_commit": "c" * 40, "template_sha256": "d" * 64,
            "quantization": "nf4_double_quant", "attention_implementation": "sdpa",
            "effective_max_context_tokens": 4096,
            "packages": {"transformers": "4.57.1", "peft": "0.17.1"},
        })
        config = {**self.config, "critic_model_id": None if shared else "example/critic",
                  "critic_revision": "main", "critic_max_context_tokens": 4096,
                  "critic_load_in_4bit": True, "semantic_critic": True, "critic_tokens": 1536}
        return config, SequentialCriticBackend(writer, critic)

    def test_per_sample_and_storage_changes_do_not_change_condition(self):
        changed = {**self.config, "seed": 917, "local_files_only": False,
                   "allow_download": True, "use_cache": False, "cache_dir": "/different/cache", "output_dir": "/different/run"}
        backend = {**self.backend, "last_generation_cuda_memory": {"allocated": 42}, "loaded": True,
                   "local_files_only": False, "cuda_device": "different GPU"}
        self.assertEqual(condition_identity(self.config, self.backend), condition_identity(changed, backend))
        self.assertFalse(condition_provenance(changed, backend)["provenance_unresolved"])

    def test_generation_model_template_and_precision_changes_separate_conditions(self):
        original = condition_identity(self.config, self.backend)
        for change in ({"draft_temperature": 0.9}, {"thinking_plan": True}, {"max_output_tokens": 5000}):
            with self.subTest(change=change):
                self.assertNotEqual(original, condition_identity({**self.config, **change}, self.backend))
        for change in ({"model_id": "example/different"}, {"resolved_commit": "c" * 40},
                       {"template_sha256": "c" * 64}, {"compute_dtype": "float32"}):
            with self.subTest(change=change):
                self.assertNotEqual(original, condition_identity(self.config, {**self.backend, **change}))

    def test_verified_adapter_content_hash_survives_relocation(self):
        one = {**self.backend, "adapter_path": "/first/adapter", "adapter_sha256": "d" * 64}
        two = {**self.backend, "adapter_path": "/second/adapter", "adapter_sha256": "d" * 64}
        self.assertEqual(condition_identity({**self.config, "adapter_path": "/first/adapter"}, one),
                         condition_identity({**self.config, "adapter_path": "/second/adapter"}, two))
        self.assertNotEqual(condition_identity(self.config, one), condition_identity(self.config, self.backend))
        self.assertFalse(condition_provenance(self.config, one)["provenance_unresolved"])

    def test_unhashed_adapter_path_and_mutable_revision_are_explicitly_unresolved(self):
        one = {**self.backend, "adapter_path": "/first/adapter", "resolved_commit": None}
        two = {**one, "adapter_path": "/second/adapter"}
        provenance = condition_provenance(self.config, one)
        self.assertTrue(provenance["provenance_unresolved"])
        self.assertTrue(any("adapter" in reason for reason in provenance["provenance_warnings"]))
        self.assertTrue(any("immutable" in reason for reason in provenance["provenance_warnings"]))
        self.assertNotEqual(condition_identity(self.config, one), condition_identity(self.config, two))

    def test_config_fingerprint_is_validated_when_manifest_is_frozen(self):
        row = {**records()[0], "configuration": self.config, "backend": self.backend,
               "condition_id": condition_identity(self.config, self.backend)}
        frozen = freeze_outputs([row])
        self.assertFalse(frozen[0]["provenance_unresolved"])
        self.assertEqual(freeze_outputs(frozen), frozen)
        row["configuration"] = {**self.config, "draft_temperature": 0.95}
        with self.assertRaisesRegex(ValueError, "condition_id conflicts"):
            freeze_outputs([row])

    def test_critic_checkpoint_template_precision_and_routing_changes_separate_conditions(self):
        config, router = self._critic_router()
        description = router.describe()
        original = condition_identity(config, description)
        for change in ({"resolved_commit": "f" * 40}, {"template_sha256": "f" * 64},
                       {"model_id": "example/other-critic"}, {"quantization": "different-method"},
                       {"compute_dtype": "float32"}, {"effective_max_context_tokens": 2048}):
            with self.subTest(change=change):
                different = copy.deepcopy(description)
                different["critic"].update(change)
                self.assertNotEqual(original, condition_identity(config, different))
        for change in ({"critic_mode": "shared_backbone_adapter_disabled"},
                       {"critic_roles": ["semantic_critic"]}, {"critic_adapter_enabled": True}):
            with self.subTest(change=change):
                self.assertNotEqual(original, condition_identity(config, {**description, **change}))

    def test_critic_switching_and_storage_changes_preserve_resolved_condition(self):
        config, router = self._critic_router()
        request = GenerationRequest(role="draft", messages=[{"role": "user", "content": "Test"}], max_new_tokens=8)
        router.generate(request)
        before = router.describe()
        router.generate(GenerationRequest(role="semantic_critic", messages=request.messages, max_new_tokens=8))
        after = router.describe()
        self.assertNotEqual(before["resident_backend"], after["resident_backend"])
        self.assertNotEqual(before["routing_statistics"], after["routing_statistics"])
        after["critic"].update(adapter_path="/unused/critic/adapter", adapter_sha256="9" * 64,
                               adapter_loaded=False, cuda_device={"name": "other device"},
                               last_generation_cuda_memory={"peak_allocated_bytes": 999})
        after["adapter_path"] = "/moved/writer/adapter"
        after["critic_roles"] = list(reversed(after["critic_roles"]))
        changed_config = {**config, "seed": 99, "output_dir": "/other/run",
                          "critic_revision": "c" * 40, "local_files_only": False}
        one = condition_provenance(config, before)
        two = condition_provenance(changed_config, after)
        self.assertEqual(one["condition_id"], two["condition_id"])
        self.assertFalse(two["provenance_unresolved"])
        self.assertIsNone(two["identity"]["critic"]["adapter"])
        self.assertNotIn("/unused/critic/adapter", json.dumps(two["identity"]))

    def test_shared_critic_identity_excludes_the_disabled_writer_adapter(self):
        config, router = self._critic_router(shared=True)
        description = router.describe()
        provenance = condition_provenance(config, description)
        self.assertFalse(provenance["provenance_unresolved"])
        self.assertEqual(provenance["identity"]["adapter"]["sha256"], "e" * 64)
        critic = provenance["identity"]["critic"]
        self.assertEqual(critic["revision"], "a" * 40)
        self.assertFalse(critic["adapter_enabled"])
        self.assertIsNone(critic["adapter"])
        description["critic"].update(adapter_sha256="f" * 64, adapter_path="/another/physical/adapter")
        self.assertEqual(provenance["condition_id"], condition_identity(config, description))

    def test_unresolved_critic_metadata_is_flagged_explicitly(self):
        config, router = self._critic_router()
        description = router.describe()
        description["critic"].update(resolved_commit=None, template_sha256=None)
        provenance = condition_provenance(config, description)
        self.assertTrue(provenance["provenance_unresolved"])
        self.assertIn("critic model revision is not an immutable commit", provenance["provenance_warnings"])
        self.assertIn("critic chat template content hash is unavailable", provenance["provenance_warnings"])
        del description["critic"]
        del description["critic_adapter_enabled"]
        provenance = condition_provenance(config, description)
        self.assertIn("critic backend metadata is unavailable", provenance["provenance_warnings"])
        self.assertIn("critic adapter-enabled policy is unavailable", provenance["provenance_warnings"])
        # An explicitly configured critic cannot disappear merely because an old
        # flat backend report omitted its actual execution metadata.
        provenance = condition_provenance(config, self.backend)
        self.assertTrue(provenance["provenance_unresolved"])
        self.assertEqual(provenance["identity"]["critic"]["model_id"], "example/critic")

    def test_different_critics_stay_separate_in_frozen_cohorts_and_tampering_fails(self):
        config, router = self._critic_router()
        first = router.describe()
        second = copy.deepcopy(first)
        second["critic"]["resolved_commit"] = "f" * 40
        outputs = []
        for row, backend in zip(records()[:2], (first, second)):
            outputs.append({**row, "configuration": config, "backend": backend,
                            "condition_id": condition_identity(config, backend)})
        manifest = freeze_outputs(outputs)
        cohorts = aggregate_detector_results([], manifest)
        self.assertEqual(len(cohorts), 2)
        self.assertEqual({row["n_designated"] for row in cohorts}, {1})
        manifest[0]["backend"]["critic"]["resolved_commit"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "condition_id conflicts"):
            freeze_outputs(manifest)

    def test_legacy_flat_backend_keeps_the_original_identity_shape(self):
        identity = condition_provenance(self.config, self.backend)["identity"]
        self.assertEqual(identity["schema_version"], 1)
        self.assertNotIn("critic", identity)


if __name__ == "__main__":
    unittest.main()
