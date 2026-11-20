"""Validated external contracts. Model-produced content remains untrusted data."""
from __future__ import annotations
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

Identifier = str
Mode = Literal["english", "history"]
ClaimStatus = Literal["attested", "inference", "disputed", "unknown", "fiction"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Source(Model):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    title: str = Field(min_length=1)
    text: Annotated[str, StringConstraints(strip_whitespace=False)] = Field(min_length=1)
    author: str = "unspecified"
    date: str = "unspecified"
    perspective: str = "not assessed"
    document_id: str | None = None
    url: str = ""
    locator: str = ""
    kind: str = "unspecified"
    independence_group: str | None = None
    audience: str = "unspecified"
    purpose: str = "unspecified"
    context: str = "unspecified"

    @field_validator("text")
    @classmethod
    def nonblank_verbatim_text(cls, value):
        if not value.strip():
            raise ValueError("Source text must contain non-whitespace content")
        return value


class Commitment(Model):
    id: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1)


class StateEntry(Commitment):
    evidence: str = ""


class State(Model):
    world: list[StateEntry] = Field(default_factory=list)
    reader: list[StateEntry] = Field(default_factory=list)


class DateConstraint(Model):
    before: str = Field(min_length=1, max_length=80)
    after: str = Field(min_length=1, max_length=80)


class Task(Model):
    task_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    mode: Mode
    prompt: str = Field(min_length=1)
    sources: list[Source] = Field(default_factory=list)
    source_scope: str = Field(default="", max_length=2000, description="Known limits of the supplied or retrieved evidence packet; never imply unseen sources were assessed.")
    commitments: list[Commitment] = Field(default_factory=list)
    initial_state: State = Field(default_factory=State)
    date_constraints: list[DateConstraint] = Field(default_factory=list)
    target_words: int = Field(default=700, ge=100, le=4000)
    min_words: int | None = Field(default=None, ge=1, le=4000, description="Optional hard minimum; target_words alone is advisory.")
    max_words: int | None = Field(default=None, ge=1, le=8000, description="Optional hard maximum, using whitespace-delimited word count.")

    @model_validator(mode="after")
    def integrity(self):
        if self.min_words is not None and self.min_words > self.target_words:
            raise ValueError("min_words cannot exceed target_words")
        if self.max_words is not None and self.max_words < self.target_words:
            raise ValueError("max_words cannot be below target_words")
        for name, rows in (("source", self.sources), ("commitment", self.commitments),
                           ("world state", self.initial_state.world), ("reader state", self.initial_state.reader)):
            ids = [row.id for row in rows]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate {name} IDs")
        if self.mode == "history" and not self.sources:
            raise ValueError("History requires a source packet; no factual evidence is invented by the runner.")
        commitments = {row.id: row.text for row in self.commitments}
        for entry in self.initial_state.world:
            if entry.id in commitments and commitments[entry.id] != entry.text:
                raise ValueError(f"Immutable world state conflicts with commitment {entry.id}")
        ordering: dict[str, set[str]] = {}
        for relation in self.date_constraints:
            ordering.setdefault(relation.before, set()).add(relation.after)
            ordering.setdefault(relation.after, set())
        active, done = set(), set()

        def visit(node):
            if node in active:
                raise ValueError("Date constraints contain an impossible cycle or self-ordering event")
            if node in done:
                return
            active.add(node)
            for successor in ordering[node]:
                visit(successor)
            active.remove(node)
            done.add(node)

        for node in ordering:
            visit(node)
        return self


class SupportQuote(Model):
    source_id: str
    quote: Annotated[str, StringConstraints(strip_whitespace=False)] = Field(min_length=1)


class Move(Model):
    id: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)
    support_quotes: list[SupportQuote] = Field(default_factory=list)
    claim_status: ClaimStatus
    year: int | None = Field(default=None, strict=True)
    depends_on: list[str] = Field(default_factory=list)


class Plan(Model):
    id: str = Field(min_length=1, max_length=80)
    approach: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    moves: list[Move] = Field(min_length=1, max_length=12)
    presentation: list[str] = Field(min_length=1, description="Move IDs in the order presented to the reader; every ID must name an entry in moves.")
    open_options: list[str] = Field(default_factory=list)


class Claim(Model):
    id: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)
    support_quotes: list[SupportQuote] = Field(default_factory=list)
    claim_status: ClaimStatus
    year: int | None = Field(default=None, strict=True)


class Draft(Model):
    text: str = Field(min_length=1)
    claims: list[Claim] = Field(default_factory=list)
    state_updates: State = Field(default_factory=State)


class Issue(Model):
    code: str
    message: str
    severity: Literal["material", "minor"]
    location: str = ""


class Review(Model):
    decision: Literal["accept", "revise", "reject"]
    issues: list[Issue] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def no_false_accept(self):
        if self.decision == "accept" and any(i.severity == "material" for i in self.issues):
            raise ValueError("An accept review cannot contain material issues")
        return self


class Selection(Model):
    plan_id: str
    reason: str = Field(min_length=1)


class Settings(Model):
    backend: Literal["transformers", "demo"] = "transformers"
    model_id: str = "Qwen/Qwen3-8B"
    revision: str = "main"
    device: Literal["cuda", "cpu"] = "cuda"
    load_in_4bit: bool = True
    local_files_only: bool = True
    adapter_path: str | None = None
    semantic_critic: bool = True
    critic_model_id: str | None = None
    critic_revision: str = "main"
    critic_load_in_4bit: bool = True
    critic_max_context_tokens: int | None = Field(default=None, ge=512)
    strategy: Literal["direct", "single_plan", "multi_plan"] = "multi_plan"
    max_context_tokens: int = Field(default=8192, ge=512)
    max_run_tokens: int = Field(default=90000, ge=1000)
    max_output_tokens: int = Field(default=24000, ge=256)
    max_calls: int = Field(default=28, ge=1, le=100)
    candidates: int = Field(default=4, ge=1, le=8)
    selection_policy: Literal["judge", "sample"] = "judge"
    history_evidence_audit: bool = True
    max_repairs: int = Field(default=2, ge=0, le=4)
    schema_retries: int = Field(default=1, ge=0, le=2)
    plan_tokens: int = Field(default=1024, ge=64)
    draft_tokens: int = Field(default=3072, ge=128)
    review_tokens: int = Field(default=768, ge=64)
    critic_tokens: int = Field(default=3072, ge=256)
    selection_tokens: int = Field(default=256, ge=32)
    plan_temperature: float = Field(default=0.9, gt=0, le=2)
    draft_temperature: float = Field(default=0.75, gt=0, le=2)
    review_temperature: float = Field(default=0.3, gt=0, le=2)
    nonthinking_top_p: float = Field(default=0.8, gt=0, le=1)
    thinking_top_p: float = Field(default=0.95, gt=0, le=1)
    top_k: int = Field(default=20, ge=0)
    min_p: float = Field(default=0.0, ge=0, le=1)
    thinking_plan: bool = False
    thinking_review: bool = False
    seed: int = Field(default=42, ge=0)

