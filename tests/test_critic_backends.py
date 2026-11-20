"""Routing/resource-lifecycle tests; no claim of GPU fit or model quality."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from creativity_pipeline.backends import BackendError, TransformersBackend
from creativity_pipeline.critic_backends import SequentialCriticBackend
from creativity_pipeline.interfaces import GenerationRequest, GenerationResult


class Clock:
    now = 0.0

    def read(self):
        return self.now


class Residency:
    def __init__(self):
        self.clock = Clock()
        self.resident = set()
        self.trace = []
        self.max_resident = 0


class MeasuredBackend:
    """Independent test backend that refuses concurrent residency itself."""
    def __init__(self, name, pool, prompt_tokens, revision, adapter=None):
        self.name, self.pool = name, pool
        self.prompt_tokens, self.revision = prompt_tokens, revision
        self.adapter = adapter
        self.loaded = False
        self.requests = []
        self.fail_unload = False
        self.fail_generation = False

    def count_input_tokens(self, request):
        self.pool.trace.append(("count", self.name, request.role))
        return self.prompt_tokens

    def generate(self, request):
        if not self.loaded:
            if self.pool.resident:
                raise AssertionError("A second model was loaded before releasing the first")
            self.pool.resident.add(self.name)
            self.loaded = True
            self.pool.max_resident = max(self.pool.max_resident, len(self.pool.resident))
            self.pool.clock.now += 4.0  # Separately observable model preparation.
            self.pool.trace.append(("load", self.name))
        self.requests.append(request)
        self.pool.trace.append(("generate", self.name, request.role))
        self.pool.clock.now += 2.0
        if self.fail_generation:
            raise RuntimeError("deliberate child generation failure")
        return GenerationResult(
            text=self.name + " response", input_tokens=self.prompt_tokens,
            output_tokens=7, elapsed_seconds=2.0,
            metadata={"model_id": self.name, "resolved_commit": self.revision,
                      "use_adapter": request.metadata.get("use_adapter", bool(self.adapter))},
        )

    def unload(self):
        self.pool.trace.append(("unload", self.name))
        self.pool.clock.now += 0.5
        if self.fail_unload:
            raise RuntimeError("deliberate model release failure")
        was_loaded = self.loaded
        self.loaded = False
        self.pool.resident.discard(self.name)
        return {"unloaded": was_loaded, "elapsed_seconds": 0.5, "cuda_memory": None}

    def describe(self):
        return {"backend": "test_backend", "model_id": self.name,
                "resolved_commit": self.revision, "loaded": self.loaded,
                "adapter_sha256": self.adapter, "adapter_loaded": self.loaded and bool(self.adapter)}


def request(role="draft", **changes):
    return replace(GenerationRequest(
        role=role, messages=[{"role": "user", "content": "Inspect the supplied draft."}],
        max_new_tokens=32, metadata={"use_adapter": True},
    ), **changes)


class CriticRoutingTests(unittest.TestCase):
    def setUp(self):
        self.pool = Residency()
        self.writer = MeasuredBackend("writer", self.pool, 11, "a" * 40, adapter="c" * 64)
        self.critic = MeasuredBackend("critic", self.pool, 23, "b" * 40)
        timer = patch("creativity_pipeline.critic_backends.time.perf_counter", self.pool.clock.read)
        timer.start()
        self.addCleanup(timer.stop)

    def test_token_accounting_routes_without_loading_or_switching_models(self):
        router = SequentialCriticBackend(self.writer, self.critic)
        self.assertEqual(router.count_input_tokens(request()), 11)
        self.assertEqual(router.count_input_tokens(request("semantic_critic")), 23)
        self.assertEqual(router.count_input_tokens(request("review_evidence")), 23)
        self.assertFalse(self.pool.resident)
        self.assertTrue(all(event[0] == "count" for event in self.pool.trace))
        self.assertEqual(router.describe()["routing_statistics"]["child_generation_calls"], 0)

    def test_separate_models_switch_once_per_transition_with_exact_counts_and_timing(self):
        router = SequentialCriticBackend(self.writer, self.critic)
        results = []
        for role in ("draft", "semantic_critic", "review_evidence", "repair"):
            prompt_count = router.count_input_tokens(request(role))
            result = router.generate(request(role))
            self.assertEqual(result.input_tokens, prompt_count)
            self.assertEqual(result.output_tokens, 7)
            results.append(result)
        self.assertEqual(self.pool.max_resident, 1)
        self.assertEqual(self.pool.resident, {"writer"})
        operations = [event[:2] for event in self.pool.trace if event[0] != "count"]
        self.assertEqual(operations, [
            ("load", "writer"), ("generate", "writer"),
            ("unload", "writer"), ("load", "critic"), ("generate", "critic"),
            ("generate", "critic"), ("unload", "critic"),
            ("load", "writer"), ("generate", "writer"),
        ])
        self.assertEqual([r.elapsed_seconds for r in results], [6.0, 6.5, 2.0, 6.5])
        self.assertTrue(all(not item.metadata["use_adapter"] for item in self.critic.requests))
        info = router.describe()
        self.assertEqual(info["model_id"], "writer")
        self.assertEqual(info["resolved_commit"], "a" * 40)
        self.assertEqual(info["adapter_sha256"], "c" * 64)
        self.assertEqual(info["critic"]["resolved_commit"], "b" * 40)
        self.assertEqual(info["routing_statistics"], {
            "generation_requests": 4, "child_generation_calls": 4,
            "completed_requests": 4, "failed_requests": 0,
            "model_switches": 2, "model_switch_attempts": 2,
            "total_seconds": 21.0, "unload_seconds": 1.0,
            "completed_preparation_seconds": 12.0, "completed_generation_seconds": 8.0,
        })

    def test_default_critic_reuses_backbone_and_forces_adapter_off_without_mutating_request(self):
        router = SequentialCriticBackend(self.writer)
        original = request("semantic_critic")
        router.generate(request("draft"))
        result = router.generate(original)
        router.generate(request("review_evidence"))
        self.assertTrue(original.metadata["use_adapter"])
        self.assertFalse(result.metadata["use_adapter"])
        self.assertFalse(self.writer.requests[-1].metadata["use_adapter"])
        self.assertFalse(any(event[0] == "unload" for event in self.pool.trace))
        info = router.describe()
        self.assertEqual(info["critic_mode"], "shared_backbone_adapter_disabled")
        self.assertEqual(info["routing_statistics"]["model_switches"], 0)

    def test_unload_failure_never_loads_or_calls_the_next_model(self):
        router = SequentialCriticBackend(self.writer, self.critic)
        router.generate(request())
        self.writer.fail_unload = True
        with self.assertRaisesRegex(RuntimeError, "release failure"):
            router.generate(request("semantic_critic"))
        self.assertEqual(self.critic.requests, [])
        self.assertFalse(self.critic.loaded)
        self.assertEqual(self.pool.resident, {"writer"})
        failed = router.describe()["routing_calls"][-1]
        self.assertFalse(failed["child_generation_started"])
        self.assertEqual(failed["unload_seconds"], 0.5)

    def test_child_failure_is_preserved_without_fallback_and_next_switch_releases_it(self):
        router = SequentialCriticBackend(self.writer, self.critic)
        router.generate(request())
        self.critic.fail_generation = True
        with self.assertRaisesRegex(RuntimeError, "generation failure"):
            router.generate(request("semantic_critic"))
        self.assertEqual(self.pool.resident, {"critic"})
        self.assertEqual(len(self.writer.requests), 1)  # No fallback generation.
        router.generate(request("repair"))
        self.assertEqual(self.pool.resident, {"writer"})
        info = router.describe()
        self.assertEqual(info["routing_statistics"]["child_generation_calls"], 3)
        self.assertEqual(info["routing_statistics"]["failed_requests"], 1)
        self.assertEqual(info["routing_statistics"]["model_switches"], 2)
        self.assertIsNone(info["routing_calls"][1]["generation_seconds"])

    def test_constructor_rejects_two_already_resident_models(self):
        self.writer.loaded = self.critic.loaded = True
        with self.assertRaisesRegex(BackendError, "Both router models are resident"):
            SequentialCriticBackend(self.writer, self.critic)

    def test_same_backend_object_is_shared_even_if_passed_twice(self):
        router = SequentialCriticBackend(self.writer, self.writer)
        router.generate(request())
        router.generate(request("semantic_critic"))
        self.assertFalse(any(event[0] == "unload" for event in self.pool.trace))
        self.assertEqual(router.describe()["critic_mode"], "shared_backbone_adapter_disabled")


class BackendUnloadTests(unittest.TestCase):
    def test_unload_drops_cyclic_model_reference_before_cuda_cache_clear_and_preserves_identity(self):
        class WeightOwner:
            def __init__(self):
                self.cycle = self

        backend = TransformersBackend()
        backend._model = WeightOwner()
        reference = weakref.ref(backend._model)
        events = []

        def empty_cache():
            self.assertIsNone(reference(), "model tensors would still be live when clearing the cache")
            events.append("empty_cache")

        cuda = SimpleNamespace(
            synchronize=lambda index: events.append("synchronize"),
            memory_allocated=lambda index: 100 if reference() is not None else 0,
            memory_reserved=lambda index: 200 if reference() is not None else 0,
            device=lambda index: nullcontext(), empty_cache=empty_cache,
        )
        backend._torch = SimpleNamespace(cuda=cuda)
        backend._resolved_commit = "a" * 40
        backend._template_sha256 = "b" * 64
        backend._adapter_sha256 = "c" * 64
        backend._adapter_loaded = True
        report = backend.unload()
        self.assertTrue(report["unloaded"])
        self.assertEqual(events, ["synchronize", "empty_cache"])
        self.assertEqual(report["cuda_memory"], {
            "before_allocated_bytes": 100, "before_reserved_bytes": 200,
            "after_allocated_bytes": 0, "after_reserved_bytes": 0,
        })
        info = backend.describe()
        self.assertFalse(info["loaded"])
        self.assertFalse(info["adapter_loaded"])
        self.assertEqual(info["resolved_commit"], "a" * 40)
        self.assertEqual(info["adapter_sha256"], "c" * 64)
        self.assertFalse(backend.unload()["unloaded"])


if __name__ == "__main__":
    unittest.main()
