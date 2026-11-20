"""Local Transformers inference with exact, unreduced token accounting.

Importing this module does not import torch, Transformers, bitsandbytes, or PEFT.
Loading is local/cache-only unless the caller explicitly opts into downloads.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import gc
import hashlib
import importlib
from importlib import metadata as package_metadata
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Any

from .interfaces import GenerationRequest, GenerationResult


class BackendError(RuntimeError):
    """A local model cannot execute the requested operation."""


class ContextLimitError(ValueError):
    """The complete prompt and reserved completion exceed the context budget."""


_ADAPTED_ROLES = frozenset({
    "plan", "planner", "write", "writer", "draft", "drafter", "realize",
    "realizer", "revise", "reviser", "revision", "repair",
})
_INFERENCE_LOCK = threading.RLock()  # Torch's seed and CUDA peak counters are global.


@dataclass(frozen=True)
class ThoughtSplit:
    """Token partition; reasoning tokens include their generated delimiters."""

    visible_ids: list[int]
    reasoning_tokens: int
    complete: bool


def _last_subsequence(tokens: list[int], marker: list[int]) -> int:
    if not marker:
        return -1
    for index in range(len(tokens) - len(marker), -1, -1):
        if tokens[index:index + len(marker)] == marker:
            return index
    return -1


def split_thinking_tokens(
    tokens: list[int], *, open_ids: list[int], close_ids: list[int],
    expected: bool = False, leading_open: bool = False, prompt_open: bool = False,
) -> ThoughtSplit:
    """Keep the answer after the final completed Qwen-style reasoning block.

    An unfinished reasoning block is never presented as the final answer. Literal
    markers inside an ordinary, non-thinking answer do not trigger this parser.
    """
    if not (expected or leading_open or prompt_open):
        return ThoughtSplit(list(tokens), 0, True)
    close = _last_subsequence(tokens, close_ids)
    opening = _last_subsequence(tokens, open_ids)
    if close < 0 or opening > close:
        return ThoughtSplit([], len(tokens), False)
    boundary = close + len(close_ids)
    return ThoughtSplit(list(tokens[boundary:]), boundary, True)


def _positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_request(request: GenerationRequest) -> None:
    if not isinstance(request.role, str) or not request.role:
        raise ValueError("role must be a nonempty string")
    _positive_int(request.max_new_tokens, "max_new_tokens")
    if (isinstance(request.temperature, bool)
            or not isinstance(request.temperature, (int, float))
            or not math.isfinite(request.temperature) or request.temperature < 0):
        raise ValueError("temperature must be finite and nonnegative")
    if (isinstance(request.seed, bool) or not isinstance(request.seed, int)
            or not 0 <= request.seed < 2**63):
        raise ValueError("seed must be an integer in [0, 2**63)")
    if not isinstance(request.thinking, bool):
        raise ValueError("thinking must be a boolean")
    for name in ("top_p", "min_p"):
        value = getattr(request, name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1
            or (name == "top_p" and value == 0)
        ):
            interval = "(0, 1]" if name == "top_p" else "[0, 1]"
            raise ValueError(f"{name} must be a finite number in {interval}, or None")
    if request.top_k is not None and (
        isinstance(request.top_k, bool) or not isinstance(request.top_k, int) or request.top_k < 0
    ):
        raise ValueError("top_k must be a nonnegative integer, or None")
    if not request.messages or not isinstance(request.messages, list):
        raise ValueError("messages must be a nonempty list")
    for message in request.messages:
        if (not isinstance(message, dict) or not isinstance(message.get("role"), str)
                or not message["role"] or not isinstance(message.get("content"), str)):
            raise ValueError("each message must have string role and content fields")
    if not isinstance(request.metadata, dict):
        raise ValueError("metadata must be a dictionary")
    if "use_adapter" in request.metadata and not isinstance(request.metadata["use_adapter"], bool):
        raise ValueError("metadata.use_adapter must be a boolean")


def _load_complete_adapter(model: Any, adapter_path: str, peft: Any) -> Any:
    """Load once and reject PEFT's otherwise warning-only partial checkpoints.

    PEFT 0.17's load_adapter returns adapter-filtered missing_keys plus
    unexpected_keys; from_pretrained only warns and discards that result.
    """
    config = peft.PeftConfig.from_pretrained(adapter_path, local_files_only=True)
    config.inference_mode = True
    wrapped = peft.get_peft_model(
        model, config, adapter_name="default", autocast_adapter_dtype=False,
    )
    result = wrapped.load_adapter(
        adapter_path, adapter_name="default", is_trainable=False,
        local_files_only=True, autocast_adapter_dtype=False,
        ignore_mismatched_sizes=False,
    )
    if not hasattr(result, "missing_keys") or not hasattr(result, "unexpected_keys"):
        raise BackendError("PEFT did not return adapter loading diagnostics; completeness is unknown")
    missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    if missing or unexpected:
        raise BackendError(
            "Adapter tensor inventory does not match its configuration: "
            f"missing_keys={missing}; unexpected_keys={unexpected}"
        )
    return wrapped


def _adapter_payload_sha256(adapter_path: str) -> str:
    """Identify the exact supported local weight artifact, without loading it."""
    directory = Path(adapter_path)
    weight_name = "adapter_model.safetensors"
    unsupported = sorted(
        item.name for item in directory.glob("adapter_model*")
        if item.name != weight_name or not item.is_file()
    )
    weights = directory / weight_name
    if unsupported or not weights.is_file():
        raise BackendError(
            "Adapter identity requires one adapter_model.safetensors file; "
            f"unsupported or missing weight layout: {unsupported or ['missing adapter_model.safetensors']}"
        )
    try:
        config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("adapter_config.json must contain an object")
        canonical = json.dumps(config, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"), allow_nan=False).encode("utf-8")
        digest = hashlib.sha256(b"creativity-pipeline-adapter-v1\0")
        for name, size in (("adapter_config.json", len(canonical)), (weight_name, weights.stat().st_size)):
            encoded_name = name.encode("utf-8")
            digest.update(len(encoded_name).to_bytes(8, "big"))
            digest.update(encoded_name)
            digest.update(size.to_bytes(8, "big"))
            if name == "adapter_config.json":
                digest.update(canonical)
            else:
                with weights.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
        return digest.hexdigest()
    except (OSError, UnicodeError, ValueError) as exc:
        raise BackendError("Could not hash the local adapter configuration and weights") from exc


class TransformersBackend:
    """One locally resident causal LM, optionally with an unmerged PEFT adapter.

    CUDA defaults to 4-bit NF4, double quantization, and bfloat16 computation.
    CPU mode requires ``load_in_4bit=False`` and uses float32. It is intended for
    an explicitly selected small, already available checkpoint, not an 8B smoke test.
    """

    def __init__(
        self, model_id: str = "Qwen/Qwen3-8B", revision: str = "main",
        device: str = "cuda", load_in_4bit: bool = True,
        adapter_path: str | None = None, max_context_tokens: int = 8192,
        local_files_only: bool = True,
    ) -> None:
        _positive_int(max_context_tokens, "max_context_tokens")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a nonempty model name or local path")
        if not isinstance(revision, str) or not revision:
            raise ValueError("revision must be a nonempty revision or commit")
        if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", device):
            raise ValueError("device must be 'cpu', 'cuda', or 'cuda:N'")
        if not isinstance(load_in_4bit, bool) or not isinstance(local_files_only, bool):
            raise ValueError("load_in_4bit and local_files_only must be booleans")
        if device == "cpu" and load_in_4bit:
            raise ValueError("CPU mode requires load_in_4bit=False")
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.adapter_path = str(Path(adapter_path).expanduser().resolve()) if adapter_path else None
        self.max_context_tokens = max_context_tokens
        self.local_files_only = local_files_only
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._transformers: Any = None
        self._adapter_loaded = False
        self._resolved_commit: str | None = None
        self._template_sha256: str | None = None
        self._effective_context = max_context_tokens
        self._last_memory: dict[str, int] | None = None
        self._attention_implementation: str | None = None
        self._cuda_info: dict[str, Any] | None = None
        self._model_config: Any = None
        self._adapter_metadata: dict[str, Any] | None = None
        self._adapter_template_verified = False
        self._adapter_sha256: str | None = None

    @property
    def _cuda_index(self) -> int:
        return int(self.device.split(":", 1)[1]) if ":" in self.device else 0

    def _read_adapter_metadata(self) -> dict[str, Any] | None:
        """Check project adapter provenance before tokenizer or weight loading."""
        if self.adapter_path is None:
            return None
        if self._adapter_metadata is not None:
            return self._adapter_metadata
        directory = Path(self.adapter_path)
        if not directory.is_dir():
            raise BackendError("adapter_path must be an existing local adapter directory")

        def read_object(name: str) -> dict[str, Any]:
            try:
                value = json.loads((directory / name).read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError) as exc:
                raise BackendError(
                    f"Adapter requires a readable {name}; use the completed adapter output directory"
                ) from exc
            if not isinstance(value, dict):
                raise BackendError(f"Adapter metadata {name} must contain a JSON object")
            return value

        adapter = read_object("adapter_config.json")
        training = read_object("training_config.json")
        report = read_object("training_report.json")
        if report.get("status") != "completed":
            raise BackendError("Adapter training_report.json must describe a completed run")
        report_config = report.get("config")
        if not isinstance(report_config, dict):
            raise BackendError("Adapter training report has no model configuration")
        names = [
            adapter.get("base_model_name_or_path"), training.get("model_id"),
            report_config.get("model_id"),
        ]
        if any(name != self.model_id for name in names):
            raise BackendError(
                "Adapter base model metadata disagrees with the requested model_id; "
                "use the model on which the adapter was trained"
            )
        revisions = [
            adapter.get("revision"), training.get("resolved_model_revision"),
            report.get("resolved_model_revision"),
        ]
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value)
               for value in revisions) or len(set(revisions)) != 1:
            raise BackendError("Adapter metadata must agree on one immutable base-model commit SHA")
        saved_revision = revisions[0]
        if (re.fullmatch(r"[0-9a-f]{40}", self.revision)
                and self.revision != saved_revision):
            raise BackendError("Explicit model revision conflicts with the adapter's trained base revision")
        template_hash = report.get("chat_template_sha256")
        if not isinstance(template_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", template_hash):
            raise BackendError("Adapter training report is missing a valid chat-template hash")
        self._adapter_metadata = {
            "model_id": self.model_id, "revision": saved_revision,
            "chat_template_sha256": template_hash,
        }
        return self._adapter_metadata

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        adapter_metadata = self._read_adapter_metadata()
        selected_revision = (
            adapter_metadata["revision"]
            if adapter_metadata and self.revision == "main" else self.revision
        )
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise BackendError("Install the project's inference dependencies to use TransformersBackend") from exc
        try:
            # Resolve exactly once, then use the same immutable revision for the
            # tokenizer and the eventual model. Tokenizer init_kwargs need not
            # contain a commit hash, so they cannot be the source of this pin.
            model_config = transformers.AutoConfig.from_pretrained(
                self.model_id, revision=selected_revision,
                local_files_only=self.local_files_only, trust_remote_code=False,
            )
            resolved = getattr(model_config, "_commit_hash", None)
            if not resolved and re.fullmatch(r"[0-9a-f]{40}", selected_revision):
                resolved = selected_revision
            if resolved and (not isinstance(resolved, str)
                             or not re.fullmatch(r"[0-9a-f]{40}", resolved)):
                raise BackendError("Resolved model revision must be an immutable commit SHA")
            if not resolved and not Path(self.model_id).expanduser().is_dir():
                raise BackendError("Could not resolve the model revision to a commit before tokenization")
            if adapter_metadata and resolved != adapter_metadata["revision"]:
                raise BackendError("Explicit model revision resolves to a different commit than the adapter's base")
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id, revision=resolved or selected_revision,
                local_files_only=self.local_files_only, trust_remote_code=False,
            )
        except (OSError, ValueError) as exc:
            mode = "local/cache-only" if self.local_files_only else "download-enabled"
            raise BackendError(f"Could not load the model config/tokenizer in {mode} mode") from exc
        if not getattr(tokenizer, "chat_template", None):
            raise BackendError("The tokenizer needs a saved chat_template; no implicit prompt format is substituted")
        template = tokenizer.get_chat_template()
        template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
        if adapter_metadata and template_hash != adapter_metadata["chat_template_sha256"]:
            raise BackendError("Tokenizer chat-template hash differs from the adapter's training template")
        tokenizer_commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        if tokenizer_commit and resolved and tokenizer_commit != resolved:
            raise BackendError("Tokenizer commit does not match the resolved base-model commit")
        self._template_sha256 = template_hash
        self._resolved_commit = resolved
        self._model_config = model_config
        self._adapter_template_verified = adapter_metadata is not None
        limit = getattr(tokenizer, "model_max_length", None)
        if isinstance(limit, int) and 0 < limit < 10**9:
            self._effective_context = min(self._effective_context, limit)
        model_limit = getattr(model_config, "max_position_embeddings", None)
        if isinstance(model_limit, int) and model_limit > 0:
            self._effective_context = min(self._effective_context, model_limit)
        self._transformers = transformers
        self._tokenizer = tokenizer

    def _encode_request(self, request: GenerationRequest) -> tuple[list[int], str]:
        _validate_request(request)
        self._ensure_tokenizer()
        rendered = self._tokenizer.apply_chat_template(
            request.messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=request.thinking,
        )
        # The template already contains its special tokens. Never add them twice.
        encoded = self._tokenizer(rendered, add_special_tokens=False, truncation=False)
        ids = encoded["input_ids"]
        if not isinstance(ids, list) or not ids or not all(isinstance(i, int) for i in ids):
            raise BackendError("The tokenizer did not produce one nonempty token sequence")
        if request.thinking:
            template = self._tokenizer.get_chat_template()
            if "enable_thinking" not in template and not rendered.rstrip().endswith("<think>"):
                raise BackendError("This tokenizer template does not implement the requested thinking switch")
        return ids, rendered

    def count_input_tokens(self, request: GenerationRequest) -> int:
        """Count the complete chat template without loading model weights."""
        with _INFERENCE_LOCK:
            ids, _ = self._encode_request(request)
            return len(ids)

    def _check_context(self, input_tokens: int, output_budget: int) -> None:
        total = input_tokens + output_budget
        if total > self._effective_context:
            raise ContextLimitError(
                f"Prompt ({input_tokens}) + max_new_tokens ({output_budget}) = {total} "
                f"exceeds context limit {self._effective_context}; no text was truncated"
            )

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._ensure_tokenizer()
        if self.adapter_path and not Path(self.adapter_path).is_dir():
            raise BackendError("adapter_path must be an existing local adapter directory")
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise BackendError("PyTorch is required for local model inference") from exc
        is_cuda = self.device.startswith("cuda")
        if is_cuda:
            if not torch.cuda.is_available() or self._cuda_index >= torch.cuda.device_count():
                raise BackendError(f"Requested CUDA device {self._cuda_index} is unavailable; no CPU fallback was used")
            with torch.cuda.device(self._cuda_index):
                if not torch.cuda.is_bf16_supported():
                    raise BackendError("This CUDA backend requires bfloat16 support")
            props = torch.cuda.get_device_properties(self._cuda_index)
            self._cuda_info = {
                "index": self._cuda_index, "name": props.name,
                "total_memory_bytes": int(props.total_memory),
                "compute_capability": f"{props.major}.{props.minor}",
                "torch_cuda_version": getattr(torch.version, "cuda", None),
            }
        transformers = self._transformers
        load_options: dict[str, Any] = {
            "revision": self._resolved_commit or self.revision,
            "config": self._model_config,
            "local_files_only": self.local_files_only,
            "trust_remote_code": False, "use_safetensors": True,
            "torch_dtype": torch.bfloat16 if is_cuda else torch.float32,
            "attn_implementation": "sdpa", "low_cpu_mem_usage": True,
        }
        if is_cuda:
            # No automatic CPU offload or multi-GPU placement is hidden in a run.
            load_options["device_map"] = {"": self._cuda_index}
        if self.load_in_4bit:
            try:
                importlib.import_module("bitsandbytes")
            except ImportError as exc:
                raise BackendError("bitsandbytes is required for load_in_4bit=True") from exc
            load_options["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
            )
        try:
            adapter_sha256 = _adapter_payload_sha256(self.adapter_path) if self.adapter_path else None
            try:
                model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id, **load_options)
                attention = "sdpa"
            except ValueError as exc:
                # Some otherwise valid small CPU checkpoints lack an SDPA implementation.
                message = str(exc).lower()
                mentions_sdpa = "sdpa" in message or "scaled_dot_product_attention" in message
                if not mentions_sdpa or not any(term in message for term in ("not support", "unsupported")):
                    raise
                load_options["attn_implementation"] = "eager"
                model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id, **load_options)
                attention = "eager"
            if not is_cuda:
                model = model.to("cpu")
            model_commit = getattr(model.config, "_commit_hash", None)
            if model_commit and self._resolved_commit and model_commit != self._resolved_commit:
                raise BackendError("Loaded model commit differs from the tokenizer's resolved commit")
            if self.adapter_path:
                peft = importlib.import_module("peft")
                model = _load_complete_adapter(model, self.adapter_path, peft)
            model.eval()
        except ImportError as exc:
            raise BackendError("A required local inference dependency is missing; check the inference installation") from exc
        except OSError as exc:
            raise BackendError("Could not load the requested model or local adapter weights") from exc
        model_limit = getattr(model.config, "max_position_embeddings", None)
        if isinstance(model_limit, int) and model_limit > 0:
            self._effective_context = min(self._effective_context, model_limit)
        self._resolved_commit = getattr(model.config, "_commit_hash", None) or self._resolved_commit
        self._torch, self._model = torch, model
        self._adapter_loaded = self.adapter_path is not None
        self._adapter_sha256 = adapter_sha256
        self._attention_implementation = attention

    def _marker_ids(self, marker: str) -> list[int]:
        ids = self._tokenizer.encode(marker, add_special_tokens=False)
        recovered = self._tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return ids if recovered == marker else []

    def unload(self) -> dict[str, Any]:
        """Release this backend's model, keeping tokenizer and past provenance.

        This drops weights rather than moving them to CPU. empty_cache only
        releases unused allocator blocks after live model references are gone;
        unrelated allocations in the process may remain.
        """
        with _INFERENCE_LOCK:
            started = time.perf_counter()
            if self._model is None:
                return {"unloaded": False, "elapsed_seconds": time.perf_counter() - started,
                        "cuda_memory": None}
            torch = self._torch
            cuda = self.device.startswith("cuda") and torch is not None
            memory = None
            if cuda:
                torch.cuda.synchronize(self._cuda_index)
                memory = {
                    "before_allocated_bytes": int(torch.cuda.memory_allocated(self._cuda_index)),
                    "before_reserved_bytes": int(torch.cuda.memory_reserved(self._cuda_index)),
                }
            # Do not bind the old model to a local variable: that reference would
            # keep its tensors alive while the allocator cache is being cleared.
            self._model = None
            self._adapter_loaded = False
            gc.collect()
            if cuda:
                with torch.cuda.device(self._cuda_index):
                    torch.cuda.empty_cache()
                memory.update(
                    after_allocated_bytes=int(torch.cuda.memory_allocated(self._cuda_index)),
                    after_reserved_bytes=int(torch.cuda.memory_reserved(self._cuda_index)),
                )
            return {"unloaded": True, "elapsed_seconds": time.perf_counter() - started,
                    "cuda_memory": memory}

    def _use_adapter(self, request: GenerationRequest) -> bool:
        wanted = request.metadata.get("use_adapter", request.role.lower() in _ADAPTED_ROLES)
        return bool(self._adapter_loaded and wanted)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate once. Neither context overflow nor unfinished thought is hidden."""
        with _INFERENCE_LOCK:
            prompt_ids, rendered = self._encode_request(request)
            self._check_context(len(prompt_ids), request.max_new_tokens)
            self._ensure_model()
            self._check_context(len(prompt_ids), request.max_new_tokens)
            torch, model, tokenizer = self._torch, self._model, self._tokenizer
            is_cuda = self.device.startswith("cuda")
            input_device = f"cuda:{self._cuda_index}" if is_cuda else "cpu"
            use_adapter = self._use_adapter(request)
            adapter_context = model.disable_adapter() if self._adapter_loaded and not use_adapter else nullcontext()
            eos = getattr(model.generation_config, "eos_token_id", None)
            if eos is None:
                eos = getattr(tokenizer, "eos_token_id", None)
            eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is None and eos_ids:
                pad_id = eos_ids[0]
            options: dict[str, Any] = {
                "max_new_tokens": request.max_new_tokens,
                "do_sample": request.temperature > 0, "num_beams": 1,
                "num_return_sequences": 1, "return_dict_in_generate": False,
                "output_scores": False, "use_cache": True,
            }
            if request.temperature > 0:
                options["temperature"] = float(request.temperature)
            for name in ("top_p", "top_k", "min_p"):
                value = getattr(request, name)
                if value is not None:
                    options[name] = value
            effective_sampling = {
                name: options.get(name, getattr(model.generation_config, name, None))
                for name in ("top_p", "top_k", "min_p")
            }
            if eos_ids:
                options["eos_token_id"] = eos_ids
            if pad_id is not None:
                options["pad_token_id"] = pad_id
            if is_cuda:
                torch.cuda.synchronize(self._cuda_index)
                torch.cuda.reset_peak_memory_stats(self._cuda_index)
                before = int(torch.cuda.memory_allocated(self._cuda_index))
            else:
                before = 0
            started = time.perf_counter()
            inputs = torch.tensor([prompt_ids], dtype=torch.long, device=input_device)
            mask = torch.ones_like(inputs)
            torch.manual_seed(request.seed)
            with adapter_context, torch.inference_mode():
                generated = model.generate(input_ids=inputs, attention_mask=mask, **options)
            if is_cuda:
                torch.cuda.synchronize(self._cuda_index)
                self._last_memory = {
                    "before_allocated_bytes": before,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self._cuda_index)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self._cuda_index)),
                    "after_allocated_bytes": int(torch.cuda.memory_allocated(self._cuda_index)),
                }
            elapsed = time.perf_counter() - started
            sequence = generated[0].tolist()
            if sequence[:len(prompt_ids)] != prompt_ids:
                raise BackendError("The model did not return the original prompt followed by its completion")
            output_ids = sequence[len(prompt_ids):]
            if len(output_ids) > request.max_new_tokens:
                raise BackendError("The model exceeded the requested output token budget")
            raw_output = tokenizer.decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            thought = split_thinking_tokens(
                output_ids, open_ids=self._marker_ids("<think>"), close_ids=self._marker_ids("</think>"),
                expected=request.thinking, leading_open=raw_output.lstrip().startswith("<think>"),
                prompt_open=rendered.rstrip().endswith("<think>"),
            )
            visible = tokenizer.decode(thought.visible_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            stopped = bool(output_ids and output_ids[-1] in eos_ids)
            hit_limit = len(output_ids) >= request.max_new_tokens and not stopped
            finish = "length" if hit_limit else ("stop" if stopped else "incomplete")
            incomplete_reason: str | None = None
            if not thought.complete:
                finish = "length" if hit_limit else "incomplete"
                incomplete_reason = "unterminated_thinking"
            elif not visible:
                finish = "length" if hit_limit else "incomplete"
                incomplete_reason = "empty_final_answer"
            elif hit_limit:
                incomplete_reason = "output_token_limit"
            elif not stopped:
                incomplete_reason = "generation_ended_without_eos"
            return GenerationResult(
                text=visible, input_tokens=len(prompt_ids), output_tokens=len(output_ids),
                finish_reason=finish, accounting="tokenizer", elapsed_seconds=elapsed,
                metadata={
                    "backend": "transformers", "model_id": self.model_id,
                    "requested_revision": self.revision, "resolved_commit": self._resolved_commit,
                    "template_sha256": self._template_sha256, "role": request.role,
                    "thinking_requested": request.thinking, "thinking_complete": thought.complete,
                    "reasoning_tokens": thought.reasoning_tokens, "raw_output": raw_output,
                    "complete": finish == "stop", "incomplete_reason": incomplete_reason,
                    "use_adapter": use_adapter, "seed": request.seed,
                    "adapter_sha256": self._adapter_sha256,
                    "max_context_tokens": self._effective_context,
                    "max_new_tokens": request.max_new_tokens, "cuda_memory": self._last_memory,
                    "decoding": {
                        "temperature": float(request.temperature), "do_sample": request.temperature > 0,
                        **effective_sampling,
                        "repetition_penalty": getattr(model.generation_config, "repetition_penalty", None),
                    },
                },
            )

    def describe(self) -> dict[str, Any]:
        """Return provenance without importing optional runtimes or loading weights."""
        versions: dict[str, str | None] = {}
        for name in ("torch", "transformers", "accelerate", "bitsandbytes", "peft", "safetensors"):
            try:
                versions[name] = package_metadata.version(name)
            except package_metadata.PackageNotFoundError:
                versions[name] = None
        return {
            "backend": "transformers", "model_id": self.model_id,
            "requested_revision": self.revision, "resolved_commit": self._resolved_commit,
            "loaded": self._model is not None, "tokenizer_loaded": self._tokenizer is not None,
            "device": self.device, "load_in_4bit": self.load_in_4bit,
            "quantization": "nf4_double_quant" if self.load_in_4bit else None,
            "compute_dtype": "bfloat16" if self.device.startswith("cuda") else "float32",
            "attention_implementation": self._attention_implementation,
            "adapter_path": self.adapter_path, "adapter_loaded": self._adapter_loaded,
            "adapter_sha256": self._adapter_sha256,
            "adapter_base_revision": self._adapter_metadata["revision"] if self._adapter_metadata else None,
            "adapter_template_verified": self._adapter_template_verified,
            "local_files_only": self.local_files_only,
            "requested_max_context_tokens": self.max_context_tokens,
            "effective_max_context_tokens": self._effective_context,
            "template_sha256": self._template_sha256, "packages": versions,
            "cuda_device": self._cuda_info, "last_generation_cuda_memory": self._last_memory,
            "accounting": "tokenizer_including_template_reasoning_and_eos",
        }
