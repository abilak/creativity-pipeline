"""Optional critic routing with one resident model among the router's children.

No extra LLM call is made here. Each request is counted with the selected
tokenizer, then generates once with that same backend. Separate model switching
reloads weights and therefore has a visible latency cost; no CPU offload or
out-of-memory fallback is performed. Individual-model fit still needs measuring.
"""
from __future__ import annotations

from dataclasses import replace
import time
from typing import Any, Protocol

from .backends import BackendError, _INFERENCE_LOCK, _validate_request
from .interfaces import Backend, GenerationRequest, GenerationResult


class ResidentBackend(Backend, Protocol):
    def unload(self) -> dict[str, Any]: ...


class SequentialCriticBackend:
    """Route dedicated criticism, defaulting to the writer with adapter off.

    With a separate critic, this router owns the residency of both backends;
    callers must not load their models directly or retain their model tensors.
    A second model is never evidence of independent errors or factual authority.
    """

    def __init__(
        self, writer: ResidentBackend, critic: ResidentBackend | None = None,
        critic_roles: frozenset[str] = frozenset({"semantic_critic", "review_evidence"}),
    ) -> None:
        if isinstance(critic_roles, str):
            raise ValueError("critic_roles must be a collection of role names")
        self.critic_roles = frozenset(critic_roles)
        if not self.critic_roles or any(not isinstance(role, str) or not role for role in self.critic_roles):
            raise ValueError("critic_roles must contain nonempty role names")
        self.writer = writer
        self.critic = critic if critic is not None else writer
        self._separate = self.critic is not self.writer
        for backend in (self.writer, self.critic):
            if not callable(getattr(backend, "unload", None)):
                raise TypeError("Sequential critic routing requires backends with unload()")
        self._calls: list[dict[str, Any]] = []
        with _INFERENCE_LOCK:
            self._resident_slot()  # Reject two already-loaded children rather than hide them.

    def _resident_slot(self) -> str | None:
        writer_loaded = bool(self.writer.describe().get("loaded"))
        critic_loaded = self._separate and bool(self.critic.describe().get("loaded"))
        if writer_loaded and critic_loaded:
            raise BackendError("Both router models are resident; unload one before using sequential routing")
        return "writer" if writer_loaded else "critic" if critic_loaded else None

    def _route(self, request: GenerationRequest) -> tuple[str, ResidentBackend, GenerationRequest, bool]:
        _validate_request(request)
        is_critic = request.role in self.critic_roles
        if is_critic:
            # Critic calls never inherit a writer adapter, including caller metadata.
            request = replace(request, metadata={**request.metadata, "use_adapter": False})
        slot = "critic" if is_critic and self._separate else "writer"
        return slot, self.critic if slot == "critic" else self.writer, request, is_critic

    def count_input_tokens(self, request: GenerationRequest) -> int:
        """Route tokenizer accounting only; do not load, unload, or generate."""
        with _INFERENCE_LOCK:
            _, backend, routed, _ = self._route(request)
            return backend.count_input_tokens(routed)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        with _INFERENCE_LOCK:
            slot, backend, routed, is_critic = self._route(request)
            previous = self._resident_slot()
            started = time.perf_counter()
            record: dict[str, Any] = {
                "request_index": len(self._calls) + 1,
                "role": request.role, "route": "critic" if is_critic else "writer",
                "backend_slot": slot, "previous_resident": previous,
                "switch_requested": previous is not None and previous != slot,
                "switched_model": False,
                "child_generation_started": False, "status": "preparing",
                "unload_seconds": 0.0, "preparation_seconds": 0.0,
                "generation_seconds": 0.0,
            }
            child_started: float | None = None
            try:
                if record["switch_requested"]:
                    outgoing = self.writer if previous == "writer" else self.critic
                    unload_started = time.perf_counter()
                    try:
                        record["unload"] = outgoing.unload()
                    finally:
                        record["unload_seconds"] = time.perf_counter() - unload_started
                    if outgoing.describe().get("loaded"):
                        raise BackendError("Previous model is still resident after unload; next model was not loaded")
                record["loaded_before_generation"] = bool(backend.describe().get("loaded"))
                child_started = time.perf_counter()
                record["child_generation_started"] = True
                result = backend.generate(routed)
                child_seconds = time.perf_counter() - child_started
                record["generation_seconds"] = result.elapsed_seconds
                # Child elapsed time measures generation; the remainder includes
                # model loading and prompt/output processing, not just loading.
                record["preparation_seconds"] = max(0.0, child_seconds - result.elapsed_seconds)
                record["switched_model"] = record["switch_requested"]
                record["status"] = "completed"
                record["total_seconds"] = time.perf_counter() - started
                output = replace(
                    result, elapsed_seconds=record["total_seconds"],
                    metadata={**result.metadata, "routing": dict(record)},
                )
                self._calls.append(record)
                return output
            except BaseException as exc:
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["total_seconds"] = time.perf_counter() - started
                record["switched_model"] = record["switch_requested"] and bool(backend.describe().get("loaded"))
                # Failed calls do not invent a decode/preparation breakdown.
                record["generation_seconds"] = None
                record["preparation_seconds"] = None
                self._calls.append(record)
                raise

    def unload(self) -> dict[str, Any]:
        """Release any resident child without making an LLM call."""
        with _INFERENCE_LOCK:
            reports = {"writer": self.writer.unload()}
            if self._separate:
                reports["critic"] = self.critic.unload()
            return reports

    def describe(self) -> dict[str, Any]:
        """Keep writer identity stable even when its weights are currently absent."""
        with _INFERENCE_LOCK:
            writer = self.writer.describe()
            critic = self.critic.describe() if self._separate else dict(writer)
            completed = [call for call in self._calls if call["status"] == "completed"]
            statistics = {
                "generation_requests": len(self._calls),
                "child_generation_calls": sum(call["child_generation_started"] for call in self._calls),
                "completed_requests": len(completed),
                "failed_requests": len(self._calls) - len(completed),
                "model_switches": sum(call["switched_model"] for call in self._calls),
                "model_switch_attempts": sum(call["switch_requested"] for call in self._calls),
                "total_seconds": sum(call["total_seconds"] for call in self._calls),
                "unload_seconds": sum(call["unload_seconds"] for call in self._calls),
                "completed_preparation_seconds": sum(call["preparation_seconds"] for call in completed),
                "completed_generation_seconds": sum(call["generation_seconds"] for call in completed),
            }
            return {
                **writer,
                "backend": "sequential_critic_router", "writer_backend_type": writer.get("backend"),
                "loaded": bool(writer.get("loaded") or (self._separate and critic.get("loaded"))),
                "critic_mode": "separate_model" if self._separate else "shared_backbone_adapter_disabled",
                "critic_roles": sorted(self.critic_roles), "critic_adapter_enabled": False,
                "resident_backend": self._resident_slot(),
                "writer": writer, "critic": critic,
                "routing_statistics": statistics,
                "routing_calls": [dict(call) for call in self._calls],
                "residency_policy": "one resident model among this router's children; reload on switch",
                "timing_note": "preparation_seconds includes loading and prompt/output processing; generation_seconds is child-reported",
            }
