"""Curated SFT data, split checks, and completion-only tokenization.

This module deliberately has no torch/transformers dependency. ``accepted`` is a
curator's attestation, backed by a nonempty provenance record; it is not an
automated factual-verification result. Related prompts, source packets, and
revisions must share a group_id before the curator assigns a held-out split.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from .io import parse_json_object
from .models import Draft, Plan


class DatasetValidationError(ValueError):
    """A dataset cannot safely be used for the requested experiment."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__("Dataset validation failed: " + "; ".join(report["errors"]))


class SequenceTooLongError(ValueError):
    def __init__(self, actual_length: int, max_length: int):
        self.actual_length = actual_length
        self.max_length = max_length
        super().__init__(
            f"Example has {actual_length} tokens including the prompt and EOS; "
            f"limit is {max_length}. It was not truncated."
        )


class PromptMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "user"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class TrainingExample(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    approach_id: str = Field(min_length=1)
    mode: Literal["english", "history"]
    split: Literal["train", "validation", "test"]
    messages: list[PromptMessage] = Field(min_length=1)
    completion: str = Field(min_length=1)
    accepted: StrictBool
    provenance: str = Field(min_length=1)

    @field_validator("id", "group_id", "approach_id", "completion", "provenance")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @model_validator(mode="after")
    def prompt_structure(self) -> "TrainingExample":
        if self.messages[-1].role != "user":
            raise ValueError("the prompt must end with a user message")
        if any(message.role == "system" for message in self.messages[1:]):
            raise ValueError("a system message is allowed only at the beginning")
        if self.accepted:
            try:
                target = parse_json_object(self.completion)
            except ValueError as exc:
                raise ValueError(f"Accepted completion must be Plan or Draft JSON: {exc}") from exc
            for contract in (Plan, Draft):
                try:
                    contract.model_validate(target)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError("Accepted completion must validate as the pipeline's Plan or Draft contract")
        return self

    def prompt_messages(self) -> list[dict[str, str]]:
        return [message.model_dump() for message in self.messages]


def _prompt_key(example: TrainingExample) -> str:
    # Whitespace normalization catches literal prompt copies with formatting edits.
    # Semantic paraphrases still require the curator to supply a shared group_id.
    normalized = [
        {"role": message.role, "content": " ".join(message.content.split())}
        for message in example.messages
    ]
    return sha256(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()


def _read_and_report(path: str | Path) -> tuple[list[TrainingExample], dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    report: dict[str, Any] = {
        "path": str(source), "valid": False, "errors": [], "warnings": [],
        "rows": 0, "accepted_rows": 0, "rejected_rows": 0,
        "counts": {}, "accepted_counts": {}, "groups": 0,
        "group_leakage": [], "prompt_leakage": [],
        "verification": "accepted=true is a curator attestation; provenance is recorded, not independently authenticated",
    }
    examples: list[TrainingExample] = []
    try:
        payload = source.read_bytes()
        report["sha256"] = sha256(payload).hexdigest()
        lines = payload.decode("utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        report["errors"].append(f"Cannot read UTF-8 JSONL: {exc}")
        return examples, report

    ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        report["rows"] += 1
        try:
            raw = json.loads(line)
            example = TrainingExample.model_validate(raw)
        except (ValueError, TypeError) as exc:
            report["errors"].append(f"Line {line_number}: {exc}")
            continue
        if example.id in ids:
            report["errors"].append(f"Line {line_number}: duplicate id {example.id!r}")
        ids.add(example.id)
        examples.append(example)

    groups: dict[str, set[str]] = defaultdict(set)
    prompts: dict[str, set[str]] = defaultdict(set)
    prompt_ids: dict[str, list[str]] = defaultdict(list)
    counts: Counter[str] = Counter()
    accepted_counts: Counter[str] = Counter()
    for example in examples:
        groups[example.group_id].add(example.split)
        key = _prompt_key(example)
        prompts[key].add(example.split)
        prompt_ids[key].append(example.id)
        counts[f"{example.mode}/{example.split}"] += 1
        if example.accepted:
            accepted_counts[f"{example.mode}/{example.split}"] += 1
    for group, splits in sorted(groups.items()):
        if len(splits) > 1:
            report["group_leakage"].append({"group_id": group, "splits": sorted(splits)})
            report["errors"].append(f"Group {group!r} crosses splits: {', '.join(sorted(splits))}")
    for key, splits in sorted(prompts.items()):
        if len(splits) > 1:
            report["prompt_leakage"].append({"ids": prompt_ids[key], "splits": sorted(splits)})
            report["errors"].append(f"Copied prompt crosses splits in IDs: {', '.join(prompt_ids[key])}")
    report["counts"] = dict(sorted(counts.items()))
    report["accepted_counts"] = dict(sorted(accepted_counts.items()))
    report["accepted_rows"] = sum(accepted_counts.values())
    report["rejected_rows"] = len(examples) - report["accepted_rows"]
    report["groups"] = len(groups)
    if not examples:
        report["errors"].append("The dataset contains no valid examples")
    if not any(example.accepted and example.split == "train" for example in examples):
        report["errors"].append("At least one accepted training example is required")
    if not any(example.accepted and example.split == "validation" for example in examples):
        report["errors"].append("An accepted, group-held-out validation split is required")
    report["warnings"].append(
        "Only explicit group IDs and normalized exact prompt copies are checked; "
        "curators must group semantic paraphrases, shared source packets, and revisions."
    )
    report["valid"] = not report["errors"]
    return examples, report


def validate_dataset(path: str | Path) -> dict[str, Any]:
    """Return a serializable report; never download a tokenizer or model."""
    return _read_and_report(path)[1]


def load_training_examples(path: str | Path) -> tuple[list[TrainingExample], dict[str, Any]]:
    examples, report = _read_and_report(path)
    if not report["valid"]:
        raise DatasetValidationError(report)
    return examples, report


def accepted_split(
    examples: Sequence[TrainingExample], mode: str, split: str,
) -> list[TrainingExample]:
    return [e for e in examples if e.accepted and e.mode == mode and e.split == split]


def prepare_completion_tokens(
    tokenizer: Any,
    messages: Sequence[dict[str, str] | PromptMessage],
    completion: str,
    max_length: int,
) -> dict[str, list[int]]:
    """Use the actual generation prefix and supervise the completion plus EOS.

    Prefix and completion are encoded separately, then concatenated. This is an
    exact continuation of the tokens used at inference and avoids BPE merges at
    the character boundary changing the mask or swallowing a target token.
    No truncation or automatic context removal is performed. Qwen3's empty
    non-thinking scaffold belongs to the masked prefix, not the target.
    """
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    prompt = [
        message.model_dump() if isinstance(message, PromptMessage) else dict(message)
        for message in messages
    ]
    if not prompt or prompt[-1].get("role") != "user":
        raise ValueError("messages must end in a user message")
    if not isinstance(completion, str) or not completion.strip():
        raise ValueError("completion must be nonempty text")
    if not isinstance(getattr(tokenizer, "eos_token_id", None), int):
        raise ValueError("the tokenizer must define one integer eos_token_id")
    for token in getattr(tokenizer, "all_special_tokens", []):
        if token and (token in completion or any(token in m["content"] for m in prompt)):
            raise ValueError(f"Dataset text contains reserved tokenizer control token {token!r}")
    if "<think>" in completion or "</think>" in completion:
        raise ValueError("Targets must be inspectable outputs, not <think> blocks")
    prefix = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("the chat template did not produce a text generation prefix")
    prompt_ids = list(tokenizer.encode(prefix, add_special_tokens=False))
    completion_ids = list(tokenizer.encode(completion, add_special_tokens=False))
    if not prompt_ids or not completion_ids:
        raise ValueError("prompt and completion must each contain tokens")
    eos = tokenizer.eos_token_id
    if eos in completion_ids:
        raise ValueError("completion already contains EOS; provide only its text")
    target_ids = completion_ids + [eos]
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_length:
        raise SequenceTooLongError(len(input_ids), max_length)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


def pad_completion_batch(
    features: Sequence[dict[str, list[int]]], pad_token_id: int,
) -> dict[str, list[list[int]]]:
    """Right-pad by position, preserving supervised EOS even when pad==EOS."""
    if not features:
        raise ValueError("cannot collate an empty batch")
    maximum = max(len(feature["input_ids"]) for feature in features)
    batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for feature in features:
        length = len(feature["input_ids"])
        if length == 0 or len(feature["labels"]) != length or len(feature["attention_mask"]) != length:
            raise ValueError("tokenized features must have equal, nonempty field lengths")
        padding = maximum - length
        batch["input_ids"].append(feature["input_ids"] + [pad_token_id] * padding)
        batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
        batch["labels"].append(feature["labels"] + [-100] * padding)
    return batch


def balanced_epoch_indices(
    examples: Sequence[TrainingExample], *, seed: int, epoch: int,
    num_samples: int | None = None,
) -> list[int]:
    """Balance prompt families, then accepted approaches within each family.

    Each group_id is a prompt family. Shuffled round-robin cycles make group
    exposure differ by at most one sample, and likewise approach exposure within
    each group. Rows cycle within each approach. Only accepted training examples
    may enter; rejected examples never receive a diversity bonus.
    """
    if not examples:
        raise ValueError("cannot sample an empty training set")
    if any(not example.accepted or example.split != "train" for example in examples):
        raise ValueError("the sampler accepts only accepted train-split examples")
    count = len(examples) if num_samples is None else num_samples
    if count < 1:
        raise ValueError("num_samples must be positive")
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, example in enumerate(examples):
        groups[example.group_id][example.approach_id].append(index)
    rng = random.Random(f"{seed}:{epoch}")
    group_cycle: list[str] = []
    approach_cycles: dict[str, list[str]] = {}
    row_cycles: dict[tuple[str, str], list[int]] = {}
    result: list[int] = []
    while len(result) < count:
        if not group_cycle:
            group_cycle = sorted(groups)
            rng.shuffle(group_cycle)
        group = group_cycle.pop()
        if not approach_cycles.get(group):
            approach_cycles[group] = sorted(groups[group])
            rng.shuffle(approach_cycles[group])
        approach = approach_cycles[group].pop()
        key = (group, approach)
        if not row_cycles.get(key):
            row_cycles[key] = list(groups[group][approach])
            rng.shuffle(row_cycles[key])
        result.append(row_cycles[key].pop())
    return result
