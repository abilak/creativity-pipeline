"""Explicit local IO and strict JSON extraction; no generated text is executed."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_json_object(text: str) -> dict[str, Any]:
    """Allow one fenced object; reject trailing explanations and ambiguous objects."""
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate JSON key: {key}")
            value[key] = item
        return value
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    parsed = json.loads(cleaned, object_pairs_hook=pairs_hook, parse_constant=reject_constant)
    if not isinstance(parsed, dict):
        raise ValueError("Expected exactly one JSON object")
    return parsed

