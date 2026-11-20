#!/usr/bin/env python3
"""Read a run's bounded diagnostic metadata without importing the pipeline or a model."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys


JSON_LIMIT = 8 * 1024 * 1024
EVENT_LIMIT = 64 * 1024 * 1024
LINE_LIMIT = 2 * 1024 * 1024


def _scalar(value, limit=160):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if abs(value) < 2 ** 63 else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:limit] + ("..." if len(value) > limit else "")
    return None


def _fields(value, names):
    value = value if isinstance(value, dict) else {}
    return {name: _scalar(value[name]) for name in names if name in value}


def compact_error(value):
    """Retain error type/field paths, never Pydantic input_value payloads."""
    if not isinstance(value, str) or not value:
        return None
    lines = value[:20000].splitlines()
    headline = re.split(r"\binput_value=", lines[0], maxsplit=1)[0].rstrip(", ")[:260]
    fields = []
    for index, line in enumerate(lines[1:], start=1):
        match = re.search(r"\[type=([a-zA-Z0-9_]+)", line)
        if match:
            # Paths emitted by Pydantic consist of field names and indices.
            path = lines[index - 1].strip()
            if re.fullmatch(r"[a-zA-Z0-9_.-]{1,120}", path):
                fields.append(f"{path}: {match.group(1)}")
            else:
                fields.append(match.group(1))
    return "\n".join([headline, *fields[:6]])[:650]


def _read_json(directory, name, warnings):
    path = directory / name
    if not path.exists():
        warnings.append(f"{name} is missing")
        return {}
    try:
        if not path.resolve().is_relative_to(directory):
            warnings.append(f"{name} points outside this run; skipped")
            return {}
        if path.stat().st_size > JSON_LIMIT:
            warnings.append(f"{name} exceeds the 8 MiB diagnostic limit; skipped")
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            warnings.append(f"{name} is not a JSON object")
            return {}
        return value
    except (OSError, UnicodeError, ValueError) as exc:
        warnings.append(f"Cannot read {name}: {type(exc).__name__}")
        return {}


def _calls(directory, max_calls, warnings):
    path = directory / "events.jsonl"
    rows, seen = {}, set()
    malformed = oversized = 0
    if not path.exists():
        warnings.append("events.jsonl is missing")
        return [], 0
    try:
        if not path.resolve().is_relative_to(directory):
            warnings.append("events.jsonl points outside this run; skipped")
            return [], 0
        consumed = 0
        with path.open("rb") as stream:
            while consumed < EVENT_LIMIT:
                line = stream.readline(LINE_LIMIT + 1)
                if not line:
                    break
                consumed += len(line)
                if len(line) > LINE_LIMIT:
                    oversized += 1
                    while line and not line.endswith(b"\n") and consumed < EVENT_LIMIT:
                        line = stream.readline(LINE_LIMIT + 1)
                        consumed += len(line)
                    continue
                try:
                    event = json.loads(line)
                except (UnicodeError, ValueError):
                    malformed += 1
                    continue
                if not isinstance(event, dict):
                    malformed += 1
                    continue
                call_id = event.get("call_id")
                if type(call_id) is not int or call_id < 1:
                    continue
                seen.add(call_id)
                row = rows.setdefault(call_id, {"call_id": call_id})
                kind = event.get("kind")
                if kind == "request":
                    request = event.get("request")
                    row.update(_fields(request, ("role", "max_new_tokens", "thinking")))
                    row["prompt_tokens"] = _scalar(event.get("estimated_input_tokens"))
                elif kind == "response":
                    response = event.get("response")
                    response = response if isinstance(response, dict) else {}
                    row.update(_fields(response, ("input_tokens", "output_tokens", "finish_reason", "elapsed_seconds")))
                    meta = response.get("metadata")
                    meta = meta if isinstance(meta, dict) else {}
                    row.update(_fields(meta, ("incomplete_reason", "complete")))
                    routing = _fields(meta.get("routing"), ("backend_slot", "switched_model", "generation_seconds", "preparation_seconds", "unload_seconds", "total_seconds"))
                    if routing:
                        row["routing"] = routing
                elif kind in {"schema_error", "backend_error"}:
                    row[kind] = compact_error(event.get("error"))
                if len(rows) > max_calls:
                    del rows[min(rows)]
            if consumed >= EVENT_LIMIT and stream.read(1):
                warnings.append("events.jsonl scan stopped at the 64 MiB limit; later calls may be omitted")
    except OSError as exc:
        warnings.append(f"Cannot read events.jsonl: {type(exc).__name__}")
    if malformed:
        warnings.append(f"Skipped {malformed} malformed event lines")
    if oversized:
        warnings.append(f"Skipped {oversized} event lines exceeding 2 MiB")
    return [rows[key] for key in sorted(rows)], max(0, len(seen) - len(rows))


def inspect_run(directory, max_calls=12):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("Run directory does not exist or is not a directory")
    if not 1 <= max_calls <= 50:
        raise ValueError("max_calls must be between 1 and 50")
    warnings = []
    summary_name = "assessment.json" if not (directory / "summary.json").exists() and (directory / "assessment.json").exists() else "summary.json"
    summary = _read_json(directory, summary_name, warnings)
    task = _read_json(directory, "task.json", warnings)
    config = _read_json(directory, "config.json", warnings)
    backend = summary.get("backend")
    backend = backend if isinstance(backend, dict) else {}
    sources = task.get("sources")
    sources = sources if isinstance(sources, list) else []
    source_ids = [_scalar(source.get("id"), 64) for source in sources if isinstance(source, dict)]
    issues = summary.get("final_issues")
    issues = issues if isinstance(issues, list) else []
    issue_counts = Counter(_scalar(issue.get("code"), 80) for issue in issues if isinstance(issue, dict) and isinstance(issue.get("code"), str))
    calls, omitted = _calls(directory, max_calls, warnings)
    backend_report = _fields(backend, ("backend", "model_id", "device", "load_in_4bit", "compute_dtype", "attention_implementation", "critic_mode"))
    backend_report["cuda_device"] = _fields(backend.get("cuda_device"), ("index", "name", "total_memory_bytes", "compute_capability", "torch_cuda_version"))
    backend_report["packages"] = _fields(backend.get("packages"), ("torch", "transformers", "accelerate", "bitsandbytes", "peft", "safetensors"))
    backend_report["routing_statistics"] = _fields(backend.get("routing_statistics"), ("generation_requests", "model_switches", "total_seconds", "unload_seconds", "completed_preparation_seconds", "completed_generation_seconds"))
    report = {
        "diagnostics_version": 1,
        "selected_run": str(directory),
        "summary_file": summary_name,
        "summary": _fields(summary, ("pipeline_version", "status", "mode", "strategy", "accepted_plan_count", "draft_revision", "repairs", "word_count", "elapsed_seconds")),
        "error": compact_error(summary.get("error")),
        "tokens": _fields(summary.get("tokens"), ("calls", "input_tokens", "output_tokens", "total_tokens", "unaccounted_calls", "complete")),
        "task_mode": _scalar(task.get("mode")),
        "supplied_source_ids": source_ids[:40],
        "source_count": len(sources),
        "source_ids_omitted": max(0, len(source_ids) - 40),
        "issue_counts": dict(issue_counts.most_common(20)),
        "issue_codes_omitted": max(0, len(issue_counts) - 20),
        "backend": backend_report,
        "configuration": _fields(config, ("semantic_critic", "critic_model_id", "max_context_tokens", "critic_max_context_tokens", "draft_tokens", "critic_tokens", "review_tokens", "max_repairs", "schema_retries")),
        "calls": calls,
        "call_rows_omitted": omitted,
        "timing_note": "With the sequential router, elapsed_seconds is total call wall time; generation_seconds includes setup and prefill, not decode alone. preparation_seconds includes loading and other processing. Saved CUDA metadata describes this run, not current GPU activity.",
        "warnings": warnings,
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog="Read-only: no model loading, generation, downloads, or changes to run files. Output excludes prompts, sources, draft prose, model responses, and Pydantic input values.")
    parser.add_argument("run_directory", nargs="?", help="Directory containing summary.json (or assessment.json)")
    parser.add_argument("--latest", metavar="RUNS_DIRECTORY", help="Choose the most recently modified summary/assessment beneath this directory")
    parser.add_argument("--max-calls", type=int, default=12, help="Number of final call rows to show, 1–50 (default: 12)")
    args = parser.parse_args(argv)
    if bool(args.run_directory) == bool(args.latest):
        parser.error("Provide one run directory OR --latest RUNS_DIRECTORY")
    if not 1 <= args.max_calls <= 50:
        parser.error("--max-calls must be between 1 and 50")
    try:
        directory = args.run_directory
        if args.latest:
            root = Path(args.latest).expanduser().resolve()
            candidates = [path for name in ("summary.json", "assessment.json") for path in root.rglob(name)
                          if path.is_file() and path.resolve().is_relative_to(root)]
            if not candidates:
                raise ValueError("No summary.json or assessment.json found beneath --latest")
            directory = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path))).parent
        report = inspect_run(directory, args.max_calls)
    except (OSError, ValueError) as exc:
        print(f"inspect_run: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
