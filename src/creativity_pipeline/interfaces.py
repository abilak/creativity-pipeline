"""Small dependency-free interface shared by real and fixture backends."""
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerationRequest:
    role: str
    messages: list[dict[str, str]]
    max_new_tokens: int
    temperature: float = 0.7
    seed: int = 0
    thinking: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str = "stop"
    accounting: str = "tokenizer"
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    def count_input_tokens(self, request: GenerationRequest) -> int: ...
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
    def describe(self) -> dict[str, Any]: ...

