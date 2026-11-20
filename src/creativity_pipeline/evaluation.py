"""Offline output manifests and detector-result analysis.

This module never queries a detector and is not used to select generated text.
Hashes bind manually imported results to the exact, designated output text.
Detector labels are observations, not proof of authorship or writing quality.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO

LABELS = ("ai", "mixed", "human")
CSV_FIELDS = (
    "output_id", "sha256", "detector", "version", "query_date", "label",
    "classified_text_share", "confidence",
)

_HASH = re.compile(r"^[0-9a-fA-F]{64}$")
_COMMIT = re.compile(r"^[0-9a-fA-F]{40,64}$")
_NON_CONDITION_FIELDS = {
    "seed", "seeds", "local_files_only", "allow_download", "use_cache", "cache_dir",
    "cache_path", "output_dir", "output_path", "run_dir", "run_path",
    "path", "paths", "directory", "output_directory",
    "backend", "model_id", "revision", "requested_revision", "resolved_commit",
    "adapter_path", "adapter_sha256", "template_sha256", "device",
    "critic_model_id", "critic_revision", "critic_max_context_tokens", "critic_load_in_4bit",
    "critic_mode", "critic_roles", "critic_adapter_enabled",
}
_INFERENCE_ID_FIELDS = (
    "load_in_4bit", "quantization", "compute_dtype", "attention_implementation",
    "effective_max_context_tokens", "packages",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _generation_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keep experimental settings, excluding sample seeds and storage locations."""
    result = {}
    for key, value in config.items():
        if key in _NON_CONDITION_FIELDS or key.endswith(("_path", "_dir")):
            continue
        if isinstance(value, Mapping):
            value = _generation_settings(value)
        elif isinstance(value, (list, tuple)):
            value = [_generation_settings(item) if isinstance(item, Mapping) else item for item in value]
        result[key] = value
    return result


def _critic_provenance(
    config: Mapping[str, Any], backend: Mapping[str, Any], reasons: list[str],
) -> dict[str, Any] | None:
    """Normalize the effective critic, excluding physical residency and paths."""
    described = backend.get("backend") == "sequential_critic_router" or any(
        key in backend for key in ("critic", "critic_mode", "critic_roles", "critic_adapter_enabled")
    )
    if not described and not config.get("critic_model_id"):
        return None  # Preserve existing flat-backend condition identities.
    mode = backend.get("critic_mode")
    if not isinstance(mode, str) or mode not in {"separate_model", "shared_backbone_adapter_disabled"}:
        reasons.append("critic routing mode is unavailable or unsupported")
        mode = mode if isinstance(mode, str) else "separate_model" if config.get("critic_model_id") else None
    shared = mode == "shared_backbone_adapter_disabled"
    source = backend.get("critic")
    if not isinstance(source, Mapping):
        if shared:
            writer = backend.get("writer")
            source = writer if isinstance(writer, Mapping) else backend
        else:
            source = {}
            reasons.append("critic backend metadata is unavailable")
    model_id = source.get("model_id") or config.get("model_id" if shared else "critic_model_id")
    revision = (source.get("resolved_commit") or source.get("requested_revision")
                or config.get("revision" if shared else "critic_revision"))
    template = source.get("template_sha256")
    if not isinstance(model_id, str) or not model_id.strip():
        reasons.append("critic model identity is unavailable")
    if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
        reasons.append("critic model revision is not an immutable commit")
    if not isinstance(template, str) or not _HASH.fullmatch(template):
        reasons.append("critic chat template content hash is unavailable")
    roles = backend.get("critic_roles")
    if (not isinstance(roles, (list, tuple, set, frozenset)) or not roles
            or any(not isinstance(role, str) or not role for role in roles)):
        roles = None
        reasons.append("critic role routing is unavailable")
    else:
        roles = sorted(set(roles))
    adapter_enabled = backend.get("critic_adapter_enabled")
    if not isinstance(adapter_enabled, bool):
        adapter_enabled = None
        reasons.append("critic adapter-enabled policy is unavailable")
    if shared and adapter_enabled is True:
        reasons.append("critic adapter policy conflicts with shared-backbone adapter-disabled mode")
    # Router descriptions include the physically loaded writer adapter in shared
    # mode. Those tensors are disabled for criticism and are not critic identity.
    adapter = None
    if adapter_enabled is True and not shared:
        adapter_hash = source.get("adapter_sha256")
        if isinstance(adapter_hash, str) and _HASH.fullmatch(adapter_hash):
            adapter = {"sha256": adapter_hash.lower(), "base_revision": source.get("adapter_base_revision")}
        else:
            adapter = {"sha256": None}
            reasons.append("critic adapter content hash is unavailable")
    inference = {key: source.get(key) for key in _INFERENCE_ID_FIELDS}
    if not isinstance(inference["load_in_4bit"], bool):
        reasons.append("critic weight precision is unavailable")
    if inference["load_in_4bit"] is True and not inference["quantization"]:
        reasons.append("critic quantization method is unavailable")
    if not isinstance(inference["compute_dtype"], str) or not inference["compute_dtype"]:
        reasons.append("critic computation precision is unavailable")
    return {
        "mode": mode, "roles": roles, "adapter_enabled": adapter_enabled,
        "backend": source.get("backend") if source is not backend else backend.get("writer_backend_type", backend.get("backend")),
        "model_id": model_id,
        "revision": revision.lower() if isinstance(revision, str) and _COMMIT.fullmatch(revision) else revision,
        "template_sha256": template.lower() if isinstance(template, str) and _HASH.fullmatch(template) else template,
        "adapter": adapter, "inference": inference,
    }


def condition_provenance(config: Mapping[str, Any], backend: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the experimental condition and its reproducibility limitations.

    Adapter ``adapter_sha256`` must identify actual weight content, as recorded
    by the inference backend. A path is retained only when that hash is absent;
    such a condition is explicitly unresolved. Hashes are audit identifiers,
    not signatures or proof that external metadata is truthful.
    """
    if not isinstance(config, Mapping) or not isinstance(backend, Mapping):
        raise ValueError("Condition configuration and backend must be mappings")
    reasons = []
    model_id = backend.get("model_id") or config.get("model_id")
    revision = backend.get("resolved_commit") or backend.get("requested_revision") or config.get("revision")
    template = backend.get("template_sha256") or config.get("template_sha256")
    if not model_id:
        reasons.append("model identity is unavailable")
    if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
        reasons.append("model revision is not an immutable commit")
    if not isinstance(template, str) or not _HASH.fullmatch(template):
        reasons.append("chat template content hash is unavailable")
    adapter_path = backend.get("adapter_path") or config.get("adapter_path")
    adapter_hash = backend.get("adapter_sha256") or config.get("adapter_sha256")
    adapter = None
    if adapter_hash is not None:
        if not isinstance(adapter_hash, str) or not _HASH.fullmatch(adapter_hash):
            raise ValueError("adapter_sha256 must be a SHA-256 content hash")
        adapter = {"sha256": adapter_hash.lower()}
    elif adapter_path:
        adapter = {"unresolved_path": str(adapter_path)}
        reasons.append("adapter content hash is unavailable; path retained as unresolved identity")
    if adapter is not None:
        adapter["base_revision"] = backend.get("adapter_base_revision")
    identity = {
        "schema_version": 1,
        "backend": backend.get("backend") or config.get("backend"),
        "model_id": model_id,
        "revision": revision.lower() if isinstance(revision, str) and _COMMIT.fullmatch(revision) else revision,
        "template_sha256": template.lower() if isinstance(template, str) and _HASH.fullmatch(template) else template,
        "adapter": adapter,
        "inference": {key: backend.get(key) for key in _INFERENCE_ID_FIELDS},
        "generation": _generation_settings(config),
    }
    critic = _critic_provenance(config, backend, reasons)
    if critic is not None:
        identity["schema_version"] = 2
        identity["critic"] = critic
    condition_id = "condition-" + output_hash(_canonical_json(identity))[:24]
    return {
        "condition_id": condition_id,
        "provenance_unresolved": bool(reasons),
        "provenance_warnings": reasons,
        "identity": identity,
    }


def condition_identity(config: Mapping[str, Any], backend: Mapping[str, Any]) -> str:
    """Fingerprint model/template/adapter and generation settings, excluding seed."""
    return condition_provenance(config, backend)["condition_id"]


def _has_fixture_marker(value: Any) -> bool:
    if isinstance(value, Mapping):
        if any(value.get(key) for key in ("fixture", "is_fixture")):
            return True
        return any(_has_fixture_marker(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_fixture_marker(child) for child in value)
    return False


def _freeze_condition(row: dict[str, Any], where: str) -> None:
    declared = row.get("condition_id")
    reasons = row.get("provenance_warnings", [])
    if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
        raise ValueError(f"{where}: provenance_warnings must be a list of strings")
    reasons = list(reasons)
    if declared is None:
        # Do not pool unrelated, unidentified outputs into one apparent system.
        row["condition_id"] = "unresolved-" + output_hash(row["output_id"])[:24]
        reasons.append("condition identity was not supplied")
        row["identity_source"] = "unresolved"
    else:
        row["condition_id"] = _required_string(row, "condition_id", where)
        config = row.get("configuration", row.get("config"))
        if config is not None and not row["condition_id"].startswith("unresolved-"):
            backend = row.get("backend", {})
            provenance = condition_provenance(config, backend)
            if row["condition_id"] != provenance["condition_id"]:
                raise ValueError(f"{where}: condition_id conflicts with configuration/backend provenance")
            reasons.extend(provenance["provenance_warnings"])
            row["identity_source"] = "configuration_fingerprint"
        elif row["condition_id"].startswith("unresolved-"):
            reasons.append("condition identity was not supplied")
            row["identity_source"] = "unresolved"
        else:
            row["identity_source"] = "declared"
            if row["origin"] == "model":
                reasons.append("condition identity is declared without reproducible configuration")
    row["provenance_warnings"] = sorted(set(reasons))
    row["provenance_unresolved"] = bool(reasons) or bool(row.get("provenance_unresolved"))


def output_hash(text: str) -> str:
    """Hash exact UTF-8 text, including whitespace and citation markers."""
    if not isinstance(text, str):
        raise ValueError("Output text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _required_string(row: Mapping[str, Any], field: str, where: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {field} must be an explicit nonempty string")
    return value.strip()


def freeze_outputs(
    records: Iterable[Mapping[str, Any]], path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Freeze designated final texts in a JSON manifest.

    Each record explicitly supplies output_id, task_id, mode, baseline, origin,
    and text. Origin is 'human' or 'model'; it is never inferred from a detector
    label or a baseline name. Failed runs may use empty text and a failure status
    so they remain in the designated denominator. The manifest is an auditable
    snapshot, not a cryptographic signature of authorship.
    """
    result = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"output {index}: record must be a mapping")
        row = copy.deepcopy(dict(record))
        where = f"output {index}"
        if _has_fixture_marker(row):
            raise ValueError(f"{where}: fixture outputs cannot be used for evaluation")
        for field in ("output_id", "task_id", "mode", "baseline", "origin"):
            row[field] = _required_string(row, field, where)
        if row["output_id"] in seen:
            raise ValueError(f"Duplicate output_id: {row['output_id']}")
        seen.add(row["output_id"])
        if row["mode"] not in {"english", "history"}:
            raise ValueError(f"{where}: mode must be english or history")
        if row["origin"] not in {"human", "model"}:
            raise ValueError(f"{where}: origin must explicitly be human or model")
        text = row.get("text")
        digest = output_hash(text)
        if row.get("sha256") is not None and row["sha256"] != digest:
            raise ValueError(f"{where}: supplied sha256 does not match text")
        row["sha256"] = digest
        row.setdefault("status", "unreported")
        _freeze_condition(row, where)
        result.append(row)
    if path is not None:
        destination = Path(path)
        # Validate/serialize before creating a file so bad metadata cannot leave
        # a partially written artifact that looks like a frozen manifest.
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Refuse to replace a different frozen manifest by accident.
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing != result:
                raise ValueError(f"Frozen manifest already exists with different content: {destination}")
        else:
            with destination.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
    return result


export_frozen_outputs = freeze_outputs


def _manifest_rows(
    manifest: Sequence[Mapping[str, Any]] | str | Path,
) -> list[dict[str, Any]]:
    if isinstance(manifest, (str, Path)):
        loaded = json.loads(Path(manifest).read_text(encoding="utf-8"))
    else:
        loaded = list(manifest)
    if not isinstance(loaded, list):
        raise ValueError("A frozen manifest must be a JSON list of output records")
    for index, row in enumerate(loaded):
        if not isinstance(row, Mapping) or not row.get("sha256"):
            raise ValueError(f"Manifest output {index} is missing its frozen sha256")
    # Recomputes every hash, and also checks identity, origins, and duplicates.
    return freeze_outputs(loaded)


def write_detector_csv_template(
    manifest: Sequence[Mapping[str, Any]] | str | Path,
    path: str | Path,
    *,
    detector: str = "",
    version: str = "",
    query_date: str = "",
) -> None:
    """Write a manual entry sheet. Empty detector fields must be filled later."""
    rows = _manifest_rows(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "output_id": row["output_id"], "sha256": row["sha256"],
                "detector": detector, "version": version, "query_date": query_date,
            })


export_detector_csv_template = write_detector_csv_template


def _valid_query_date(value: str, where: str) -> str:
    try:
        if len(value) == 10:
            date.fromisoformat(value)
        else:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{where}: query_date must be an ISO date or datetime") from exc
    return value


def _optional_fraction(value: Any, field: str, where: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: {field} must be a fraction from 0 to 1") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{where}: {field} must be a finite fraction from 0 to 1")
    return number


def import_detector_csv(
    csv_path: str | Path | TextIO,
    manifest: Sequence[Mapping[str, Any]] | str | Path,
) -> list[dict[str, Any]]:
    """Validate manual observations against a frozen manifest.

    Required CSV fields: output_id, sha256 (output_sha256 is also accepted),
    detector, version, query_date, label. Labels are ai/mixed/human; map vendor
    fields deliberately before import. Optional text share and confidence are
    distinct 0..1 fields; neither is inferred from the other. A repeated scan of
    the same output/detector/version is rejected rather than double-counted.
    """
    frozen = _manifest_rows(manifest)
    index = {row["output_id"]: row for row in frozen}
    context = nullcontext(csv_path) if hasattr(csv_path, "read") else Path(csv_path).open("r", encoding="utf-8-sig", newline="")
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    with context as stream:
        reader = csv.DictReader(stream)
        headers = set(reader.fieldnames or [])
        if len(headers) != len(reader.fieldnames or []):
            raise ValueError("Detector CSV contains duplicate column headers")
        required = {"output_id", "detector", "version", "query_date", "label"}
        if not required <= headers or not {"sha256", "output_sha256"} & headers:
            raise ValueError("Detector CSV needs output_id, sha256, detector, version, query_date, and label headers")
        for line, raw in enumerate(reader, start=2):
            if not any(value and str(value).strip() for value in raw.values()):
                continue
            where = f"CSV row {line}"
            if None in raw:
                raise ValueError(f"{where}: more values than column headers")
            output_id = _required_string(raw, "output_id", where)
            if output_id not in index:
                raise ValueError(f"{where}: unknown output_id {output_id!r}")
            output = index[output_id]
            digest = (raw.get("sha256") or raw.get("output_sha256") or "").strip()
            if digest != output["sha256"]:
                raise ValueError(f"{where}: hash mismatch for {output_id!r}; scan the exact frozen text")
            if raw.get("sha256") and raw.get("output_sha256"):
                if raw["sha256"].strip() != raw["output_sha256"].strip():
                    raise ValueError(f"{where}: conflicting sha256 fields")
            detector = _required_string(raw, "detector", where)
            version = _required_string(raw, "version", where)
            query_date = _valid_query_date(_required_string(raw, "query_date", where), where)
            label = _required_string(raw, "label", where).casefold()
            if label not in LABELS:
                raise ValueError(f"{where}: label must be ai, mixed, or human; do not guess a vendor mapping")
            key = (output_id, detector, version)
            if key in seen:
                raise ValueError(f"{where}: repeated scan of the same output/detector/version")
            seen.add(key)
            for field in ("task_id", "mode", "baseline", "origin", "condition_id"):
                if raw.get(field) and raw[field].strip() != output[field]:
                    raise ValueError(f"{where}: {field} conflicts with the frozen manifest")
            result = {
                "output_id": output_id, "sha256": digest, "detector": detector,
                "version": version, "query_date": query_date, "label": label,
                "task_id": output["task_id"], "mode": output["mode"],
                "baseline": output["baseline"], "origin": output["origin"],
                "condition_id": output["condition_id"],
                "provenance_unresolved": output["provenance_unresolved"],
                "status": output.get("status", "unreported"),
                "classified_text_share": _optional_fraction(raw.get("classified_text_share"), "classified_text_share", where),
                "confidence": _optional_fraction(raw.get("confidence"), "confidence", where),
            }
            results.append(result)
    return results


def aggregate_detector_results(
    rows: Iterable[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]] | str | Path,
) -> list[dict[str, Any]]:
    """Report labels by domain, detector version, condition, baseline, and origin.

    Mixed labels stay separate. Rates use explicit scanned and designated
    denominators. All designated outputs, including failed/unscanned runs, stay
    visible. This descriptive summary does not treat multiple samples from one
    prompt as independent confidence-interval observations.
    """
    frozen = _manifest_rows(manifest)
    by_id = {row["output_id"]: row for row in frozen}
    cohorts: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for output in frozen:
        cohorts[(output["mode"], output["baseline"], output["origin"], output["condition_id"])].append(output)
    groups: dict[tuple[str, str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    detector_versions: set[tuple[str, str]] = set()
    seen: set[tuple[str, str, str]] = set()
    for raw in rows:
        output_id = _required_string(raw, "output_id", "detector result")
        if output_id not in by_id:
            raise ValueError(f"Unknown detector output_id: {output_id}")
        output = by_id[output_id]
        if raw.get("sha256") != output["sha256"]:
            raise ValueError(f"Detector hash mismatch for {output_id}")
        detector = _required_string(raw, "detector", "detector result")
        version = _required_string(raw, "version", "detector result")
        _valid_query_date(_required_string(raw, "query_date", "detector result"), "detector result")
        if raw.get("label") not in LABELS:
            raise ValueError("Detector result label must be ai, mixed, or human")
        key = (output_id, detector, version)
        if key in seen:
            raise ValueError("Repeated output/detector/version would double-count a scan")
        seen.add(key)
        for field in ("mode", "baseline", "origin", "task_id", "condition_id"):
            if field in raw and raw[field] != output[field]:
                raise ValueError(f"Detector result {field} conflicts with manifest for {output_id}")
        detector_versions.add((detector, version))
        groups[(output["mode"], detector, version, output["baseline"], output["origin"], output["condition_id"])].append(raw)

    result = []
    # No imported scans must still expose every designated output and failure.
    for detector, version in sorted(detector_versions) if detector_versions else [(None, None)]:
        for (mode, baseline, origin, condition_id), designated in sorted(cohorts.items()):
            scans = groups.get((mode, detector, version, baseline, origin, condition_id), [])
            counts = Counter(scan["label"] for scan in scans)
            n_scanned, n_designated = len(scans), len(designated)
            summary: dict[str, Any] = {
                "mode": mode, "detector": detector, "version": version,
                "baseline": baseline, "origin": origin,
                "condition_id": condition_id,
                "provenance_unresolved": any(output["provenance_unresolved"] for output in designated),
                "provenance_warnings": sorted({warning for output in designated for warning in output["provenance_warnings"]}),
                "n_designated": n_designated, "n_scanned": n_scanned,
                "n_missing": n_designated - n_scanned,
                "n_distinct_prompts": len({output["task_id"] for output in designated}),
                "query_dates": sorted({scan["query_date"] for scan in scans}),
                "label_counts": {label: counts[label] for label in LABELS},
                "label_rates_among_scanned": {label: counts[label] / n_scanned if n_scanned else None for label in LABELS},
                "label_rates_among_designated": {label: counts[label] / n_designated for label in LABELS},
                "designated_status_counts": dict(Counter(str(output.get("status", "unreported")) for output in designated)),
                "mixed_policy": "reported_separately",
            }
            if origin == "human":
                summary["known_human_ai_label_count"] = counts["ai"]
                summary["known_human_mixed_label_count"] = counts["mixed"]
            result.append(summary)
    return result


aggregate_labels = aggregate_detector_results
