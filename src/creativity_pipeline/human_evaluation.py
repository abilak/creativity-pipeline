"""Offline, blinded assessment of frozen outputs; no model or detector calls.

The public packet hides experimental metadata. The private key preserves every
designated output, including failures, and ties ratings to exact text hashes.
Content can itself reveal authorship; removing labels cannot guarantee blinding.
"""

from __future__ import annotations

import csv
import html
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluation import _canonical_json, _manifest_rows, output_hash


DIMENSIONS = (
    "overall_quality", "coherence", "originality", "voice", "evidence_support",
    "historical_reasoning", "uncertainty",
)
MODE_DIMENSIONS = {
    "english": ("overall_quality", "coherence", "originality", "voice"),
    "history": ("overall_quality", "coherence", "evidence_support", "historical_reasoning", "uncertainty"),
}
RATING_FIELDS = (
    "reviewer_id", "item_id", "sha256", *DIMENSIONS,
    "material_error", "error_notes", "comments",
)
RUBRICS = {
    "english": {
        "overall_quality": "The piece fulfils the prompt and works as a finished narrative.",
        "coherence": "Events, motivations, chronology, and established facts fit together; surprise has preparation and consequences.",
        "originality": "Specific, interesting choices feel earned rather than arbitrary. Unusual wording alone is insufficient.",
        "voice": "Diction, rhythm, perspective, and emotional precision support a distinctive, consistent voice.",
    },
    "history": {
        "overall_quality": "The interpretation answers the question clearly and usefully within the supplied evidence.",
        "coherence": "The chronology and argument are internally consistent, and the conclusion follows from the analysis.",
        "evidence_support": "Claims and quotations are supported by the source packet; source attributions are traceable and accurate.",
        "historical_reasoning": "Causal mechanisms, source perspectives, and plausible alternatives receive careful treatment without anachronism.",
        "uncertainty": "The piece distinguishes evidence, inference, and counterfactual speculation, with appropriately qualified conclusions.",
    },
}


def _review_context(row: Mapping[str, Any]) -> dict[str, Any]:
    task = row.get("task")
    if not isinstance(task, Mapping):
        task = row
    # Only task content is public. IDs and experimental metadata stay private.
    return {
        key: task[key] for key in (
            "prompt", "sources", "source_scope", "commitments", "initial_state", "date_constraints",
            "target_words", "min_words", "max_words",
        ) if key in task and task[key] is not None
    }


def _has_text(row: Mapping[str, Any]) -> bool:
    return bool(row["text"].strip())


def _counts(frozen: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    with_text = sum(_has_text(row) for row in frozen)
    return {
        "n_designated": len(frozen), "n_with_text": with_text,
        "n_missing_text": len(frozen) - with_text,
        "designated_status_counts": dict(sorted(Counter(str(row.get("status", "unreported")) for row in frozen).items())),
    }


def _packet(items: Sequence[Mapping[str, Any]], by_id: Mapping[str, Mapping[str, Any]]) -> str:
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Writing review packet</title>',
        '<style>body{font:17px/1.55 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#172333}'
        'h1,h2,h3{line-height:1.2}article{border-top:2px solid #aab6c5;margin-top:3rem;padding-top:1rem;break-before:page}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}code{font-size:.9em}dt{font-weight:600}'
        '.context{background:#f2f5f8;padding:1rem}.missing{border:1px solid #8795a8;padding:1rem}'
        'table{border-collapse:collapse}th,td{padding:.4rem .8rem;text-align:left;border:1px solid #cad1da}</style>',
        '</head><body><h1>Writing review packet</h1>',
        '<p>Read each item with its prompt and available sources. Record your independent judgment in '
        '<code>ratings.csv</code> before discussing items or seeing the private key. Use a consistent '
        '<code>reviewer_id</code>; leave item IDs and text hashes unchanged.</p>',
        '<p>Rate each applicable dimension with an integer: <strong>0 = very poor, 1 = weak, '
        '2 = adequate, 3 = strong, 4 = excellent</strong>. Leave dimensions for the other domain blank. '
        'Mark <code>material_error</code> as <code>yes</code>, <code>no</code>, or <code>uncertain</code>. '
        'For yes, explain the specific contradiction or unsupported claim in <code>error_notes</code>. '
        'For fiction, assess the prompt and established story facts. For history, use the supplied sources; '
        'uncertainty about an external fact is not proof of an error.</p>',
        '<p>Leave an item entirely unrated if you cannot assess it; explain limitations in a separate note '
        'to the study organiser. Missing texts must remain unrated. Scores describe your judgments, '
        'not an objective measure of creativity or authorship.</p>',
    ]
    for mode, dimensions in RUBRICS.items():
        parts.append(f'<h2>{html.escape(mode.title())} rubric</h2><dl>')
        for name, description in dimensions.items():
            parts.append(f'<dt>{html.escape(name)}</dt><dd>{html.escape(description)}</dd>')
        parts.append('</dl>')
    for item in items:
        row = by_id[item["output_id"]]
        parts.append(f'<article><h2>{html.escape(item["item_id"])}</h2>')
        parts.append(f'<p>Domain: {html.escape(row["mode"])}. Text hash: <code>{html.escape(row["sha256"])}</code></p>')
        context = _review_context(row)
        if context:
            parts.append('<section class="context"><h3>Task context</h3>')
            for field, value in context.items():
                text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                parts.append(f'<h4>{html.escape(field.replace("_", " ").title())}</h4><pre>{html.escape(text)}</pre>')
            parts.append('</section>')
        else:
            parts.append('<p>Task context was not supplied. Assess only what the available text permits.</p>')
        parts.append('<h3>Text</h3>')
        if item["has_text"]:
            parts.append(f'<pre>{html.escape(row["text"])}</pre>')
        else:
            parts.append('<p class="missing">No text was available for this item. Leave all ratings blank.</p>')
        parts.append('</article>')
    parts.append('</body></html>\n')
    return "\n".join(parts)


def export_blind_review(
    manifest: Sequence[Mapping[str, Any]] | str | Path,
    output_dir: str | Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Create reviewer/packet.html, reviewer/ratings.csv, and private_key.json.

    Distribute only ``reviewer/``. The private key contains the original frozen
    manifest, conditions, assignment seed, and integrity hashes. No existing
    output directory is overwritten. Randomization is deterministic for a
    manifest's output IDs and seed, independent of input record order.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Review seed must be an integer")
    frozen = _manifest_rows(manifest)
    if not frozen:
        raise ValueError("A review packet needs at least one designated output")
    ordered = sorted(frozen, key=lambda row: row["output_id"])
    random.Random(seed).shuffle(ordered)
    items = [{
        "item_id": f"item-{index:04d}", "output_id": row["output_id"],
        "sha256": row["sha256"], "context_sha256": output_hash(_canonical_json(_review_context(row))),
        "condition_id": row["condition_id"], "mode": row["mode"],
        "baseline": row["baseline"], "origin": row["origin"], "has_text": _has_text(row),
    } for index, row in enumerate(ordered, start=1)]
    by_id = {row["output_id"]: row for row in frozen}
    packet = _packet(items, by_id)
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=RATING_FIELDS)
    writer.writeheader()
    writer.writerows({"item_id": item["item_id"], "sha256": item["sha256"]} for item in items)
    counts = _counts(frozen)
    key = {
        "schema_version": 1, "seed": seed, "manifest": frozen,
        "manifest_sha256": output_hash(_canonical_json(frozen)),
        "items": items, "packet_sha256": output_hash(packet),
        "rating_fields": list(RATING_FIELDS), "counts": counts,
        "blinding": "Experimental metadata is hidden; text content may still reveal authorship.",
    }
    serialized_key = json.dumps(key, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination = Path(output_dir).expanduser().resolve()
    # Build and validate all content before creating the output directory.
    destination.mkdir(parents=True, exist_ok=False)
    reviewer = destination / "reviewer"
    reviewer.mkdir()
    (reviewer / "packet.html").write_bytes(packet.encode("utf-8"))
    (reviewer / "ratings.csv").write_text(csv_buffer.getvalue(), encoding="utf-8", newline="")
    (destination / "private_key.json").write_text(serialized_key, encoding="utf-8")
    return {
        "output_dir": str(destination), "packet": str(reviewer / "packet.html"),
        "ratings": str(reviewer / "ratings.csv"), "private_key": str(destination / "private_key.json"),
        **counts,
    }


def _read_key(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    key_path = Path(path)
    key = json.loads(key_path.read_text(encoding="utf-8"))
    if not isinstance(key, dict) or key.get("schema_version") != 1:
        raise ValueError("Unsupported human review key schema")
    raw_manifest = key.get("manifest")
    if not isinstance(raw_manifest, list):
        raise ValueError("Private key manifest must contain frozen records")
    frozen = _manifest_rows(raw_manifest)
    if key.get("manifest_sha256") != output_hash(_canonical_json(frozen)):
        raise ValueError("Private key manifest hash mismatch")
    if key.get("counts") != _counts(frozen):
        raise ValueError("Private key counts conflict with frozen manifest")
    if key.get("rating_fields") != list(RATING_FIELDS):
        raise ValueError("Private key rating schema does not match this importer")
    items = key.get("items")
    if not isinstance(items, list) or len(items) != len(frozen):
        raise ValueError("Private key must designate every frozen output exactly once")
    by_id = {row["output_id"]: row for row in frozen}
    item_index = {}
    seen_outputs = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("item_id"), str) or not item["item_id"]:
            raise ValueError("Private key has an invalid item")
        output_id = item.get("output_id")
        if not isinstance(output_id, str) or not output_id:
            raise ValueError("Private key has an invalid output_id")
        if item["item_id"] in item_index or output_id in seen_outputs:
            raise ValueError("Private key contains a duplicate item or output")
        if output_id not in by_id:
            raise ValueError("Private key contains an unknown output")
        row = by_id[output_id]
        for field in ("sha256", "condition_id", "mode", "baseline", "origin"):
            if item.get(field) != row[field]:
                raise ValueError(f"Private key item {field} conflicts with frozen output")
        if item.get("context_sha256") != output_hash(_canonical_json(_review_context(row))):
            raise ValueError("Private key task context hash mismatch")
        if item.get("has_text") is not _has_text(row):
            raise ValueError("Private key text availability conflicts with output")
        item_index[item["item_id"]] = item
        seen_outputs.add(output_id)
    # The packet is part of the reviewed artifact; reject edits after freezing.
    packet_path = key_path.parent / "reviewer" / "packet.html"
    actual_packet = packet_path.read_bytes().decode("utf-8")
    if key.get("packet_sha256") != output_hash(actual_packet):
        raise ValueError("Reviewer packet hash mismatch; packet changed after export")
    if actual_packet != _packet(items, by_id):
        raise ValueError("Reviewer packet conflicts with the private key assignments")
    return frozen, item_index


def import_human_ratings(path: str | Path, key_path: str | Path) -> dict[str, Any]:
    """Validate ratings and report separate dimensions, errors, and missingness.

    Required dimensions depend on the item's domain. Every supplied score must
    be an integer 0..4. A reviewer may rate an item once; another reviewer may
    independently rate it. Blank template rows are not zero-quality judgments.
    No-text outputs remain designated but cannot receive quality ratings.
    """
    frozen, items = _read_key(key_path)
    observations = []
    seen = set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if len(set(reader.fieldnames or [])) != len(reader.fieldnames or []):
            raise ValueError("Ratings CSV contains duplicate column headers")
        if not set(RATING_FIELDS) <= set(reader.fieldnames or []):
            raise ValueError("Ratings CSV must preserve all exported column headers")
        for line, raw in enumerate(reader, start=2):
            if not any(value and str(value).strip() for value in raw.values()):
                continue
            where = f"Ratings CSV row {line}"
            if None in raw:
                raise ValueError(f"{where}: more values than column headers")
            item_id = (raw.get("item_id") or "").strip()
            if item_id not in items:
                raise ValueError(f"{where}: unknown item_id")
            item = items[item_id]
            if (raw.get("sha256") or "").strip() != item["sha256"]:
                raise ValueError(f"{where}: text hash mismatch")
            if not any((raw.get(field) or "").strip() for field in (*DIMENSIONS, "material_error", "error_notes", "comments")):
                continue
            if not item["has_text"]:
                raise ValueError(f"{where}: no-text items must remain unrated")
            reviewer_id = (raw.get("reviewer_id") or "").strip()
            if not reviewer_id:
                raise ValueError(f"{where}: reviewer_id is required for a rating")
            pair = (reviewer_id, item_id)
            if pair in seen:
                raise ValueError(f"{where}: duplicate reviewer/item rating")
            seen.add(pair)
            scores = {}
            for dimension in DIMENSIONS:
                value = (raw.get(dimension) or "").strip()
                if dimension in MODE_DIMENSIONS[item["mode"]]:
                    if value not in {"0", "1", "2", "3", "4"}:
                        raise ValueError(f"{where}: {dimension} requires an integer from 0 to 4")
                    scores[dimension] = int(value)
                elif value:
                    raise ValueError(f"{where}: {dimension} is not applicable to {item['mode']}")
            material_error = (raw.get("material_error") or "").strip().casefold()
            if material_error not in {"yes", "no", "uncertain"}:
                raise ValueError(f"{where}: material_error must be yes, no, or uncertain")
            error_notes = (raw.get("error_notes") or "").strip()
            if material_error == "yes" and not error_notes:
                raise ValueError(f"{where}: a material error requires specific error_notes")
            observations.append({
                "reviewer_id": reviewer_id, "item_id": item_id, "output_id": item["output_id"],
                "sha256": item["sha256"], "scores": scores,
                "material_error": material_error, "error_notes": error_notes,
                "comments": (raw.get("comments") or "").strip(),
            })
    ratings_by_output = defaultdict(list)
    for rating in observations:
        ratings_by_output[rating["output_id"]].append(rating)
    cohorts = defaultdict(list)
    for row in frozen:
        cohorts[(row["mode"], row["baseline"], row["origin"], row["condition_id"])].append(row)
    conditions = []
    for (mode, baseline, origin, condition_id), designated in sorted(cohorts.items()):
        ratings = [rating for row in designated for rating in ratings_by_output[row["output_id"]]]
        n_reviewed = sum(bool(ratings_by_output[row["output_id"]]) for row in designated)
        dimensions = {}
        for dimension in MODE_DIMENSIONS[mode]:
            values = [rating["scores"][dimension] for rating in ratings]
            counts = Counter(values)
            dimensions[dimension] = {
                "n_ratings": len(values), "mean": sum(values) / len(values) if values else None,
                "score_counts": {str(score): counts[score] for score in range(5)},
            }
        counts = _counts(designated)
        conditions.append({
            "mode": mode, "baseline": baseline, "origin": origin, "condition_id": condition_id,
            "provenance_unresolved": any(row["provenance_unresolved"] for row in designated),
            "provenance_warnings": sorted({reason for row in designated for reason in row["provenance_warnings"]}),
            **counts, "n_ratings": len(ratings), "n_reviewed_outputs": n_reviewed,
            "n_unreviewed": len(designated) - n_reviewed,
            "n_with_text_unreviewed": counts["n_with_text"] - n_reviewed,
            "n_reviewers": len({rating["reviewer_id"] for rating in ratings}),
            "n_distinct_prompts": len({row["task_id"] for row in designated}),
            "dimensions": dimensions,
            "material_error_counts": {label: sum(rating["material_error"] == label for rating in ratings) for label in ("yes", "no", "uncertain")},
        })
    reviewed_outputs = len({rating["output_id"] for rating in observations})
    return {
        "schema_version": 1, **_counts(frozen), "n_reviewed_outputs": reviewed_outputs,
        "n_ratings": len(observations), "n_unreviewed": len(frozen) - reviewed_outputs,
        "conditions": conditions, "ratings": observations,
        "limitations": [
            "Dimension means are descriptive ordinal ratings, not a composite creativity measure.",
            "Repeated ratings and outputs from the same prompt are not independent observations.",
            "Missing and failed outputs remain designated; missing ratings are not zero scores.",
            "Experimental labels are hidden, but text content may reveal authorship.",
            "Hashes detect accidental mismatches; they are not signatures or proof of authorship.",
        ],
    }
