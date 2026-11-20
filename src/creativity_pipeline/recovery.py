"""Load an existing draft for a separate, bounded review and repair attempt."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .checks import validate_plan
from .evaluation import condition_provenance
from .io import parse_json_object, text_hash
from .models import Draft, Plan, Task


@dataclass(frozen=True)
class SavedRun:
    task: Task
    draft: Draft
    plan: dict | None
    source_manifest: dict | None
    config_path: Path
    provenance: dict
    integrity_sha256: str

    def verify(self):
        current = _snapshot_hash(self.task, self.draft, self.plan, self.source_manifest, self.provenance)
        if current != self.integrity_sha256:
            raise ValueError("Saved task, draft, plan, source selection, or provenance changed after loading")


def _snapshot_hash(task, draft, plan, manifest, provenance):
    payload = {"task": task.model_dump(), "draft": draft.model_dump(), "plan": plan,
               "source_manifest": manifest, "provenance": provenance}
    return text_hash(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))


def load_saved_run(directory: str | Path) -> SavedRun:
    """Read, never rewrite, the saved task and exact final draft.

    Hash comparisons catch mismatched artifacts, not coordinated modification
    of a run or the authenticity of a source. Earlier approvals are not loaded.
    """
    root = Path(directory).expanduser().resolve()
    hashes = {}

    def read_object(name):
        data = (root / name).read_bytes()
        value = parse_json_object(data.decode("utf-8"))
        hashes[name] = hashlib.sha256(data).hexdigest()
        return value

    summary = read_object("summary.json")
    task = Task.model_validate(read_object("task.json"))
    raw_draft = read_object("final.json")
    draft = Draft.model_validate(raw_draft)
    config = read_object("config.json")
    if not isinstance(summary.get("run_id"), str) or not summary["run_id"].strip():
        raise ValueError("The saved run needs a stable run_id")
    if summary.get("task_id") != task.task_id or summary.get("mode") != task.mode:
        raise ValueError("Saved summary identity disagrees with task.json")
    expected = summary.get("output_sha256")
    if not expected or text_hash(draft.text) != expected or raw_draft.get("text") != draft.text:
        raise ValueError("Saved final.json does not match the recorded output hash")
    final_text = (root / "final.txt").read_bytes()
    if final_text.decode("utf-8") != draft.text:
        raise ValueError("Saved final.txt and final.json disagree")
    hashes["final.txt"] = hashlib.sha256(final_text).hexdigest()
    plan = None
    if (root / "selected_plan.json").exists():
        plan = Plan.model_validate(read_object("selected_plan.json")).model_dump()
        findings = validate_plan(task.model_dump(), plan)
        if any(row["severity"] == "material" for row in findings):
            raise ValueError("Saved selected plan fails checks against the saved task")
    manifest = read_object("source_selection.json") if (root / "source_selection.json").exists() else None
    backend = summary.get("backend") or {}
    if not isinstance(backend, dict):
        raise ValueError("Saved backend description must be an object")
    parent_config = {**config, "pipeline_version": summary.get("pipeline_version", "unrecorded")}
    if summary.get("continuation"):
        parent = summary["continuation"]
        if not isinstance(parent, dict):
            raise ValueError("Saved continuation provenance must be an object")
        parent_condition = parent.get("parent_condition")
        if not isinstance(parent_condition, dict):
            raise ValueError("Saved parent condition provenance must be an object")
        parent_config["continuation_parent_condition_id"] = parent_condition.get("condition_id")
        parent_config["continuation_parent_strategy"] = parent.get("parent_strategy")
    provenance = {
        "parent_run": str(root),
        "parent_run_id": summary["run_id"],
        "parent_status": summary.get("status"),
        "parent_strategy": summary.get("strategy"),
        "parent_output_sha256": expected,
        "parent_condition": condition_provenance(parent_config, backend),
        "parent_tokens": summary.get("tokens"),
        "parent_elapsed_seconds": summary.get("elapsed_seconds"),
        "parent_is_fixture": bool(summary.get("is_fixture") or backend.get("fixture")),
        "input_file_sha256": hashes,
        "accounting_scope": "This continuation has a fresh bounded budget. Its tokens and elapsed time exclude earlier attempts.",
        "review_policy": "Earlier approvals are discarded; mechanical checks and all configured reviews run again.",
    }
    fingerprint = _snapshot_hash(task, draft, plan, manifest, provenance)
    return SavedRun(task, draft, plan, manifest, root / "config.json", provenance, fingerprint)
