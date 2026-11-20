"""Contract tests use small fakes, not downloaded models or a pretend GPU run."""
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import tempfile
import unittest
from unittest.mock import patch

from creativity_pipeline import backends
from creativity_pipeline.backends import (
    BackendError, ContextLimitError, TransformersBackend, split_thinking_tokens,
)
from creativity_pipeline.interfaces import GenerationRequest


EOS = 900001
COMMIT = "a" * 40


class Tensor:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return Tensor(self.data[item])

    def tolist(self):
        return self.data


class Tokenizer:
    chat_template = "template(enable_thinking)"
    init_kwargs = {"_commit_hash": COMMIT}
    model_max_length = 4096
    eos_token_id = EOS
    pad_token_id = None

    def __init__(self):
        self.template_calls = []
        self.tokenize_calls = []

    def get_chat_template(self):
        return self.chat_template

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(kwargs)
        suffix = "<think>\n" if kwargs["enable_thinking"] else ""
        return "<user>" + messages[-1]["content"] + "</user><assistant>" + suffix

    def __call__(self, text, **kwargs):
        self.tokenize_calls.append(kwargs)
        return {"input_ids": self.encode(text)}

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        return "".join(
            ("" if skip_special_tokens else "<eos>") if i == EOS else chr(i)
            for i in ids
        )


class Model:
    def __init__(self):
        self.config = SimpleNamespace(max_position_embeddings=4096, _commit_hash=COMMIT)
        self.generation_config = SimpleNamespace(eos_token_id=[EOS], top_p=.95, top_k=20, min_p=0, repetition_penalty=1.)
        self.output = list(map(ord, "answer")) + [EOS]
        self.calls = []
        self.adapter_disabled = False
        self.adapter_history = []
        self.raise_on_generate = None
        self.adapter_load_result = SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    @contextmanager
    def disable_adapter(self):
        self.adapter_disabled = True
        try:
            yield
        finally:
            self.adapter_disabled = False

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        self.adapter_history.append(self.adapter_disabled)
        if self.raise_on_generate:
            raise self.raise_on_generate
        return Tensor([kwargs["input_ids"].data[0] + self.output])


def make_runtime():
    tokenizer, model = Tokenizer(), Model()
    record = SimpleNamespace(imports=[], config_loads=[], tokenizer_loads=[], model_loads=[], adapter_loads=[], quantization=[], seeds=[], cuda=[])

    def load_config(name, **kwargs):
        record.config_loads.append((name, kwargs))
        return model.config

    def load_tokenizer(name, **kwargs):
        record.tokenizer_loads.append((name, kwargs))
        return tokenizer

    def load_model(name, **kwargs):
        record.model_loads.append((name, kwargs))
        return model

    def load_adapter(path, **kwargs):
        record.adapter_loads.append((model, path, kwargs))
        return model.adapter_load_result

    model.load_adapter = load_adapter

    def quantization(**kwargs):
        record.quantization.append(kwargs)
        return kwargs

    def cuda_call(name, value=None):
        def call(*args):
            record.cuda.append((name, args))
            return value
        return call

    cuda = SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1,
        device=lambda index: nullcontext(), is_bf16_supported=lambda: True,
        get_device_properties=lambda index: SimpleNamespace(name="FAKE GPU", total_memory=2000, major=8, minor=9),
        synchronize=cuda_call("synchronize"), reset_peak_memory_stats=cuda_call("reset"),
        memory_allocated=cuda_call("allocated", 100),
        max_memory_allocated=cuda_call("peak_allocated", 150),
        max_memory_reserved=cuda_call("peak_reserved", 200),
    )
    torch = SimpleNamespace(
        bfloat16="bf16", float32="fp32", long="long", cuda=cuda,
        version=SimpleNamespace(cuda="fake-cuda"), manual_seed=record.seeds.append,
        inference_mode=lambda: nullcontext(),
        tensor=lambda data, **kwargs: Tensor(data),
        ones_like=lambda value: Tensor([[1] * len(value.data[0])]),
    )
    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=load_config),
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        BitsAndBytesConfig=quantization,
    )
    modules = {
        "torch": torch, "transformers": transformers, "bitsandbytes": SimpleNamespace(),
        "peft": SimpleNamespace(
            PeftConfig=SimpleNamespace(from_pretrained=lambda path, **kwargs: SimpleNamespace(inference_mode=False)),
            get_peft_model=lambda base, config, **kwargs: base,
        ),
    }
    original_import = backends.importlib.import_module

    def import_module(name, *args, **kwargs):
        if name in modules:
            record.imports.append(name)
            return modules[name]
        return original_import(name, *args, **kwargs)

    patcher = patch.object(backends.importlib, "import_module", import_module)
    return SimpleNamespace(tokenizer=tokenizer, model=model, torch=torch, transformers=transformers, record=record, patcher=patcher)


def request(**changes):
    base = GenerationRequest(role="writer", messages=[{"role": "user", "content": "Hi"}], max_new_tokens=128, seed=42)
    return replace(base, **changes)


def cpu_backend(**kwargs):
    return TransformersBackend(model_id="a-local-tiny-model", device="cpu", load_in_4bit=False, **kwargs)


def write_adapter_metadata(directory, tokenizer):
    """Small consistent fixture for completed project adapters; no model weights."""
    records = {
        "adapter_config.json": {"base_model_name_or_path": "a-local-tiny-model", "revision": COMMIT},
        "training_config.json": {"model_id": "a-local-tiny-model", "resolved_model_revision": COMMIT},
        "training_report.json": {
            "status": "completed", "config": {"model_id": "a-local-tiny-model"},
            "resolved_model_revision": COMMIT,
            "chat_template_sha256": sha256(tokenizer.get_chat_template().encode("utf-8")).hexdigest(),
        },
    }
    for name, record in records.items():
        (Path(directory) / name).write_text(json.dumps(record), encoding="utf-8")
    # Only the separate optional real-PEFT tests contain actual tensor artifacts.
    (Path(directory) / "adapter_model.safetensors").write_bytes(b"fake-runtime-only-weight-placeholder")
    return records



class BackendTests(unittest.TestCase):
    def setUp(self):
        self.runtime = make_runtime()
        self.runtime.patcher.start()
        self.addCleanup(self.runtime.patcher.stop)

    def test_constructor_and_describe_do_not_import_or_load_runtime(self):
        runtime = self.runtime
        backend = TransformersBackend()
        assert runtime.record.imports == []
        info = backend.describe()
        assert info["loaded"] is False
        assert info["resolved_commit"] is None
        assert info["cuda_device"] is None
        assert runtime.record.imports == []

    def test_count_uses_exact_template_without_loading_weights(self):
        runtime = self.runtime
        backend = cpu_backend()
        n = backend.count_input_tokens(request(thinking=True))
        assert n == len("<user>Hi</user><assistant><think>\n")
        assert runtime.tokenizer.template_calls[-1]["enable_thinking"] is True
        assert runtime.tokenizer.tokenize_calls[-1] == {"add_special_tokens": False, "truncation": False}
        assert runtime.record.model_loads == []
        assert "torch" not in runtime.record.imports

    def test_context_overflow_fails_before_weights_are_loaded(self):
        runtime = self.runtime
        backend = cpu_backend(max_context_tokens=30)
        with self.assertRaisesRegex(ContextLimitError, "no text was truncated"):
            backend.generate(request(max_new_tokens=10))
        assert not runtime.record.model_loads

    def test_exact_context_limit_is_allowed(self):
        runtime = self.runtime
        n = len("<user>Hi</user><assistant>")
        backend = cpu_backend(max_context_tokens=n + 10)
        result = backend.generate(request(max_new_tokens=10))
        assert result.input_tokens == n
        assert result.text == "answer"

    def test_accounting_includes_thinking_delimiters_and_eos(self):
        runtime = self.runtime
        runtime.model.output = list(map(ord, "reason</think>\nanswer")) + [EOS]
        backend = cpu_backend()
        result = backend.generate(request(thinking=True))
        assert result.text == "answer"
        assert result.output_tokens == len(runtime.model.output)
        assert result.metadata["reasoning_tokens"] == len("reason</think>")
        assert result.metadata["raw_output"] == "reason</think>\nanswer<eos>"
        assert result.finish_reason == "stop"
        assert result.accounting == "tokenizer"
        assert result.metadata["complete"] is True
        assert runtime.record.seeds == [42]

    def test_thinking_budget_exhaustion_is_not_an_answer(self):
        runtime = self.runtime
        runtime.model.output = list(map(ord, "unfinished"))
        result = cpu_backend().generate(request(thinking=True, max_new_tokens=10))
        assert result.text == ""
        assert result.finish_reason == "length"
        assert result.output_tokens == 10
        assert result.metadata["thinking_complete"] is False
        assert result.metadata["incomplete_reason"] == "unterminated_thinking"
        assert result.metadata["complete"] is False

    def test_eos_before_thinking_closes_is_incomplete(self):
        runtime = self.runtime
        runtime.model.output = list(map(ord, "unfinished")) + [EOS]
        result = cpu_backend().generate(request(thinking=True))
        assert result.text == ""
        assert result.finish_reason == "incomplete"
        assert result.metadata["complete"] is False

    def test_final_answer_at_length_limit_is_marked_truncated(self):
        runtime = self.runtime
        runtime.model.output = list(map(ord, "answer"))
        result = cpu_backend().generate(request(max_new_tokens=6))
        assert result.text == "answer"
        assert result.finish_reason == "length"
        assert result.metadata["incomplete_reason"] == "output_token_limit"

    def test_eos_at_exact_limit_is_a_real_stop(self):
        runtime = self.runtime
        result = cpu_backend().generate(request(max_new_tokens=7))
        assert result.finish_reason == "stop"
        assert result.output_tokens == 7

    def test_adapter_disabled_for_reviews_and_restored_after_exception(self):
        runtime = self.runtime
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = temporary.name
        write_adapter_metadata(tmp_path, runtime.tokenizer)
        backend = cpu_backend(adapter_path=str(tmp_path))
        backend.generate(request(role="planner"))
        backend.generate(request(role="review"))
        backend.generate(request(role="analyst"))
        backend.generate(request(role="writer", metadata={"use_adapter": False}))
        backend.generate(request(role="custom", metadata={"use_adapter": True}))
        assert runtime.model.adapter_history == [False, True, True, True, False]
        runtime.model.raise_on_generate = RuntimeError("deliberate generation failure")
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            backend.generate(request(role="review"))
        assert runtime.model.adapter_disabled is False
        assert runtime.record.adapter_loads[0][2]["is_trainable"] is False

    def test_cuda_configuration_and_per_call_memory_provenance(self):
        runtime = self.runtime
        backend = TransformersBackend()
        first = backend.generate(request())
        backend.generate(request())
        options = runtime.record.model_loads[0][1]
        assert options["device_map"] == {"": 0}
        assert options["attn_implementation"] == "sdpa"
        assert options["torch_dtype"] == "bf16"
        assert options["revision"] == COMMIT
        assert options["local_files_only"] is True
        assert options["trust_remote_code"] is False
        assert runtime.record.quantization == [{
            "load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "bf16",
        }]
        assert [name for name, _ in runtime.record.cuda].count("reset") == 2
        assert first.metadata["cuda_memory"]["peak_allocated_bytes"] == 150
        info = backend.describe()
        assert info["cuda_device"]["name"] == "FAKE GPU"
        assert info["resolved_commit"] == COMMIT
        assert info["loaded"] is True

    def test_download_opt_in_is_propagated_to_both_loaders(self):
        runtime = self.runtime
        cpu_backend(local_files_only=False).generate(request())
        assert runtime.record.tokenizer_loads[0][1]["local_files_only"] is False
        assert runtime.record.model_loads[0][1]["local_files_only"] is False

    def test_sampling_overrides_are_applied_logged_and_do_not_mutate_model_defaults(self):
        backend = cpu_backend()
        result = backend.generate(request(top_p=0.8, top_k=7, min_p=0.02))
        expected = {"top_p": 0.8, "top_k": 7, "min_p": 0.02}
        for name, value in expected.items():
            self.assertEqual(self.runtime.model.calls[-1][name], value)
            self.assertEqual(result.metadata["decoding"][name], value)
        inherited = backend.generate(request())
        self.assertNotIn("top_p", self.runtime.model.calls[-1])
        self.assertEqual(inherited.metadata["decoding"]["top_p"], 0.95)
        self.assertEqual(inherited.metadata["decoding"]["top_k"], 20)

    def test_invalid_sampling_overrides_fail_before_loading(self):
        for changes in ({"top_p": 0}, {"top_p": 1.1}, {"top_p": float("nan")},
                        {"min_p": -0.1}, {"min_p": float("inf")}, {"min_p": True},
                        {"top_k": -1}, {"top_k": 1.5}, {"top_k": False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                cpu_backend().generate(request(**changes))
        self.assertEqual(self.runtime.record.imports, [])

    def test_unavailable_cuda_never_silently_falls_back(self):
        runtime = self.runtime
        runtime.torch.cuda.is_available = lambda: False
        with self.assertRaisesRegex(BackendError, "no CPU fallback"):
            TransformersBackend().generate(request())
        assert runtime.record.model_loads == []

    def test_model_context_limit_is_also_enforced(self):
        runtime = self.runtime
        runtime.model.config.max_position_embeddings = 30
        with self.assertRaises(ContextLimitError):
            cpu_backend().generate(request(max_new_tokens=10))
        assert runtime.model.calls == []

    def test_unsupported_sdpa_falls_back_only_for_that_error(self):
        runtime = self.runtime
        original = runtime.transformers.AutoModelForCausalLM.from_pretrained

        def fallback(name, **kwargs):
            if kwargs["attn_implementation"] == "sdpa":
                raise ValueError("This model does not support scaled_dot_product_attention yet")
            return original(name, **kwargs)

        runtime.transformers.AutoModelForCausalLM.from_pretrained = fallback
        backend = cpu_backend()
        backend.generate(request())
        assert backend.describe()["attention_implementation"] == "eager"

    def test_missing_chat_template_fails_without_model_load(self):
        runtime = self.runtime
        runtime.tokenizer.chat_template = None
        with self.assertRaisesRegex(BackendError, "chat_template"):
            cpu_backend().count_input_tokens(request())
        assert runtime.record.model_loads == []

    def test_invalid_request_is_rejected_before_imports(self):
        changes = [
            {"max_new_tokens": 0}, {"max_new_tokens": True}, {"temperature": float("nan")},
            {"temperature": -1}, {"seed": -1}, {"thinking": "false"},
            {"metadata": {"use_adapter": "false"}}, {"messages": []},
        ]
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    cpu_backend().generate(request(**change))
                assert self.runtime.record.imports == []

    def test_thought_parser_preserves_literal_markers_in_non_thinking_prose(self):
        text = "The literal token </think> appears in this example."
        ids = list(map(ord, text))
        result = split_thinking_tokens(ids, open_ids=list(map(ord, "<think>")), close_ids=list(map(ord, "</think>")))
        assert result.visible_ids == ids
        assert result.reasoning_tokens == 0

    def test_thought_parser_does_not_accept_an_unfinished_second_block(self):
        ids = list(map(ord, "<think>done</think><think>unfinished"))
        result = split_thinking_tokens(ids, open_ids=list(map(ord, "<think>")), close_ids=list(map(ord, "</think>")), leading_open=True)
        assert result.complete is False
        assert result.visible_ids == []


if __name__ == "__main__":
    unittest.main()
