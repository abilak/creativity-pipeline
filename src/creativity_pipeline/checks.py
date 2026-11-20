"""Deterministic checks for declared provenance, structure, and state.

These checks inspect the supplied annotations. They cannot discover every
unannotated factual assertion, establish that a source is truthful, or decide
whether a passage entails an interpretation. Passing them is not certification.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Mapping, Sequence

Issue = dict[str, Any]
HISTORY_STATUSES = {"attested", "inference", "disputed", "unknown"}
CLAIM_STATUSES = HISTORY_STATUSES | {"fiction"}
_CITATION = re.compile(
    r"\[([A-Za-z][A-Za-z0-9_.:-]*(?:\s*,\s*[A-Za-z][A-Za-z0-9_.:-]*)*)\]"
)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace only; do not silently alter quotation wording."""
    return " ".join(text.split())


def cited_source_ids(text: str) -> list[str]:
    """Source-marker IDs in first-appearance order; this is not an entailment test."""
    return list(dict.fromkeys(identifier.strip() for match in _CITATION.finditer(text)
                              for identifier in match.group(1).split(",")))


def _issue(code: str, message: str, location: str = "", severity: str = "material") -> Issue:
    result: Issue = {"code": code, "message": message, "severity": severity}
    if location:
        result["location"] = location
    return result


def _duplicate_ids(items: Sequence[Mapping[str, Any]], location: str) -> list[Issue]:
    issues = []
    counts = Counter(item.get("id") for item in items if item.get("id"))
    for identifier, count in counts.items():
        if count > 1:
            issues.append(_issue("duplicate_id", f"ID {identifier!r} appears {count} times.", location))
    for index, item in enumerate(items):
        if not isinstance(item.get("id"), str) or not item["id"].strip():
            issues.append(_issue("missing_id", "An explicit nonempty ID is required.", f"{location}[{index}]"))
    return issues


def _sources(task: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], list[Issue]]:
    rows = task.get("sources") or []
    issues = _duplicate_ids(rows, "sources")
    source_map = {row["id"]: row for row in rows if row.get("id")}
    if task.get("mode") == "history" and not any(
        isinstance(row.get("text"), str) and row["text"].strip() for row in rows
    ):
        issues.append(_issue("history_sources_required", "History requires at least one nonempty supplied source.", "sources"))
    for index, row in enumerate(rows):
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            issues.append(_issue("empty_source", "Source passages cannot be empty.", f"sources[{index}].text"))
    return source_map, issues


def _provenance(
    item: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]], history: bool, location: str
) -> list[Issue]:
    issues: list[Issue] = []
    status = item.get("claim_status")
    if status not in CLAIM_STATUSES:
        issues.append(_issue("invalid_claim_status", "Use an explicit supported claim status.", f"{location}.claim_status"))
    elif history and status == "fiction":
        issues.append(_issue("fiction_in_history", "A historical claim cannot be marked as fiction.", f"{location}.claim_status"))

    source_ids = item.get("source_ids") or []
    if len(set(source_ids)) != len(source_ids):
        issues.append(_issue("duplicate_source_reference", "The same source is listed more than once.", f"{location}.source_ids", "minor"))
    for source_id in source_ids:
        if source_id not in source_map:
            issues.append(_issue("unknown_source", f"Source {source_id!r} is not supplied.", f"{location}.source_ids"))

    if history and status in {"attested", "inference", "disputed"} and not source_ids:
        issues.append(_issue("missing_provenance", f"A history claim marked {status!r} requires source IDs.", location))

    quotes = item.get("support_quotes") or []
    valid_quotes = 0
    for index, support in enumerate(quotes):
        quote_location = f"{location}.support_quotes[{index}]"
        source_id, quote = support.get("source_id"), support.get("quote")
        if source_id not in source_ids:
            issues.append(_issue("quote_source_not_cited", "A supporting quote must use a source in this item's source_ids.", quote_location))
        if source_id not in source_map:
            issues.append(_issue("unknown_quote_source", f"Quote source {source_id!r} is not supplied.", quote_location))
            continue
        if not isinstance(quote, str) or not normalize_whitespace(quote):
            issues.append(_issue("empty_support_quote", "A supporting quote must contain text.", quote_location))
            continue
        source_text = source_map[source_id].get("text") or ""
        if normalize_whitespace(quote) not in normalize_whitespace(source_text):
            issues.append(_issue("quote_mismatch", "The quote does not appear in its source after whitespace normalization.", quote_location))
        elif source_id in source_ids:
            valid_quotes += 1

    if history and status == "attested" and not valid_quotes:
        issues.append(_issue("attested_quote_required", "An attested claim requires at least one matching source passage; matching is not a truth or entailment check.", location))
    elif history and status in {"inference", "disputed"} and source_ids and not valid_quotes:
        issues.append(_issue("support_passage_missing", "Citations are present, but no matching passage is annotated for review of this interpretation.", location, "minor"))
    return issues


def _integer_year(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _length_findings(task: Mapping[str, Any], text: str) -> list[Issue]:
    """Enforce explicit bounds; approximate targets remain advisory.

    Counts are whitespace-delimited tokens, matching the run summary. This is a
    reproducible length check, not evidence of substantive development or quality.
    """
    count = len(text.split())
    minimum, maximum, target = task.get("min_words"), task.get("max_words"), task.get("target_words")
    issues: list[Issue] = []
    if type(minimum) is int and count < minimum:
        issues.append(_issue("below_min_words", f"Draft has {count} words; the explicit minimum is {minimum}.", "text"))
    if type(maximum) is int and count > maximum:
        issues.append(_issue("above_max_words", f"Draft has {count} words; the explicit maximum is {maximum}.", "text"))
    if not issues and type(target) is int and target > 0 and not 0.5 * target <= count <= 1.5 * target:
        issues.append(_issue("target_length_deviation", f"Draft has {count} words, outside the broad advisory range of 50%-150% of target_words={target}; assess task fulfillment and development.", "text", "minor"))
    return issues


def _claim_occurrences(text: str, claim_text: str) -> list[tuple[int, int]]:
    """Return exact spans, including repeated occurrences of the same claim."""
    result = []
    start = text.find(claim_text)
    while start >= 0:
        result.append((start, start + len(claim_text)))
        start = text.find(claim_text, start + 1)
    return result


def _citation_gap_only(text: str) -> bool:
    """Allow punctuation/space and a contiguous cluster of source markers.

    Removing other citation markers permits 'claim [S1] [S2]' without treating
    intervening words or an unannotated sentence as part of the same claim.
    """
    gap = _CITATION.sub("", text)
    return all(character.isspace() or unicodedata.category(character).startswith("P") for character in gap)


def _marker_bound_to_claim(
    marker_start: int,
    marker_end: int,
    source_id: str,
    text: str,
    claim_spans: Sequence[tuple[int, int, set[str]]],
) -> bool:
    for start, end, source_ids in claim_spans:
        if source_id not in source_ids:
            continue
        if start <= marker_start and marker_end <= end:
            return True
        if end <= marker_start and _citation_gap_only(text[end:marker_start]):
            return True
    # A citation before a claim must be included in claim.text. Binding a marker
    # to a following sentence could launder the unannotated preceding sentence.
    return False


def _date_constraint_findings(
    task: Mapping[str, Any], items: Sequence[Mapping[str, Any]], item_kind: str
) -> list[Issue]:
    """Check declared years for stable event IDs, never infer dates from prose.

    Constraints describe event chronology, independently of presentation order.
    A final claim must preserve a constrained event ID even in direct generation.
    """
    issues: list[Issue] = []
    by_id = {item["id"]: item for item in items if item.get("id")}
    for index, constraint in enumerate(task.get("date_constraints") or []):
        before, after = constraint.get("before"), constraint.get("after")
        location = f"date_constraints[{index}]"
        if before not in by_id or after not in by_id:
            missing = [str(identifier) for identifier in (before, after) if identifier not in by_id]
            issues.append(_issue(
                f"date_constraint_unknown_{item_kind}",
                f"Date constraint refers to missing stable event IDs in {item_kind}s: " + ", ".join(missing),
                location,
            ))
            continue
        left, right = by_id[before].get("year"), by_id[after].get("year")
        if not _integer_year(left) or not _integer_year(right):
            issues.append(_issue("date_constraint_unresolved", "This order constraint cannot be checked without both structured integer years.", location, "minor"))
        elif left > right:
            issues.append(_issue("date_order_violation", f"{before!r} is dated {left}, after {after!r} ({right}), despite the required order.", location))
        elif left == right:
            issues.append(_issue("date_constraint_unresolved", "Both events have the same year; year-level precision cannot establish their order.", location, "minor"))
    return issues


def validate_plan(task: Mapping[str, Any], plan: Mapping[str, Any]) -> list[Issue]:
    """Check annotated provenance and plan graph constraints, not literary value."""
    source_map, issues = _sources(task)
    moves = plan.get("moves") or []
    if not moves:
        issues.append(_issue("empty_plan", "A plan needs at least one declared move.", "moves"))
    issues.extend(_duplicate_ids(moves, "moves"))
    move_map = {move["id"]: move for move in moves if move.get("id")}
    history = task.get("mode") == "history"
    graph: dict[str, list[str]] = {}
    for index, move in enumerate(moves):
        location = f"moves[{index}]"
        issues.extend(_provenance(move, source_map, history, location))
        dependencies = move.get("depends_on") or []
        for dependency in dependencies:
            if dependency not in move_map:
                issues.append(_issue("unknown_dependency", f"Dependency {dependency!r} is not a move ID.", f"{location}.depends_on"))
        if move.get("id"):
            graph[move["id"]] = [dep for dep in dependencies if dep in move_map]

    # Iterative elimination avoids recursive depth limits on long plans.
    remaining = {identifier: set(dependencies) for identifier, dependencies in graph.items()}
    ready = [identifier for identifier, dependencies in remaining.items() if not dependencies]
    removed: set[str] = set()
    while ready:
        identifier = ready.pop()
        if identifier in removed:
            continue
        removed.add(identifier)
        for other, dependencies in remaining.items():
            if identifier in dependencies:
                dependencies.remove(identifier)
                if not dependencies:
                    ready.append(other)
    if len(removed) < len(remaining):
        issues.append(_issue("dependency_cycle", "The dependency graph contains a cycle; affected moves: " + ", ".join(sorted(set(remaining) - removed)), "moves"))

    presentation = plan.get("presentation") or []
    if len(set(presentation)) != len(presentation):
        issues.append(_issue("duplicate_presentation_id", "Presentation lists a move more than once.", "presentation"))
    for identifier in presentation:
        if identifier not in move_map:
            issues.append(_issue("unknown_presentation_id", f"Presentation ID {identifier!r} is not a move ID.", "presentation"))
    issues.extend(_date_constraint_findings(task, moves, "move"))
    return issues


def validate_state_update(
    existing: Mapping[str, Any], updates: Mapping[str, Any], draft_text: str
) -> list[Issue]:
    """Allow new evidence-linked entries, but never overwrite an existing ID.

    World and reader ledgers have independent namespaces. Evidence must be an
    exact substring of this draft; its semantic relationship still needs review.
    """
    issues: list[Issue] = []
    for ledger in ("world", "reader"):
        old_rows, new_rows = existing.get(ledger) or [], updates.get(ledger) or []
        issues.extend(_duplicate_ids(old_rows, f"existing.{ledger}"))
        issues.extend(_duplicate_ids(new_rows, f"state_updates.{ledger}"))
        old = {item["id"]: item for item in old_rows if item.get("id")}
        for index, item in enumerate(new_rows):
            location = f"state_updates.{ledger}[{index}]"
            identifier, text, evidence = item.get("id"), item.get("text"), item.get("evidence")
            if identifier in old and text != old[identifier].get("text"):
                issues.append(_issue("state_overwrite", f"Existing {ledger} entry {identifier!r} cannot be silently overwritten.", location))
            if not isinstance(evidence, str) or not evidence.strip() or evidence not in draft_text:
                issues.append(_issue("state_evidence_missing", "State-update evidence must be a nonempty exact substring of the draft.", location))
    return issues


def validate_draft(
    task: Mapping[str, Any], draft: Mapping[str, Any], plan: Mapping[str, Any] | None = None
) -> list[Issue]:
    """Check declared claims and citations; do not certify full claim coverage."""
    source_map, issues = _sources(task)
    history = task.get("mode") == "history"
    text = draft.get("text") or ""
    if not isinstance(text, str) or not text.strip():
        issues.append(_issue("empty_draft", "The draft must contain text.", "text"))
        text = text if isinstance(text, str) else ""
    issues.extend(_length_findings(task, text))
    claims = draft.get("claims") or []
    if history and not claims:
        issues.append(_issue("history_claims_required", "A historical draft requires explicit claim annotations; completeness still needs review.", "claims"))
    issues.extend(_duplicate_ids(claims, "claims"))
    claim_spans: list[tuple[int, int, set[str]]] = []
    for index, claim in enumerate(claims):
        location = f"claims[{index}]"
        claim_text = claim.get("text")
        if not isinstance(claim_text, str) or not claim_text.strip() or claim_text not in text:
            issues.append(_issue("claim_text_missing", "Claim text must be a nonempty exact substring of the draft.", location))
        else:
            sources_for_claim = set(claim.get("source_ids") or [])
            claim_spans.extend((start, end, sources_for_claim) for start, end in _claim_occurrences(text, claim_text))
        issues.extend(_provenance(claim, source_map, history, location))
    # This also runs for direct generation, where no plan exists to check.
    issues.extend(_date_constraint_findings(task, claims, "claim"))

    # In history, bracketed identifier tokens are reserved for source markers.
    # In fiction, only supplied IDs and S<number> tokens are treated as markers.
    for match in _CITATION.finditer(text):
        for source_id in re.split(r"\s*,\s*", match.group(1)):
            if not history and source_id not in source_map and not re.fullmatch(r"S\d+", source_id):
                continue
            if source_id not in source_map:
                issues.append(_issue("unknown_citation_marker", f"Citation marker [{source_id}] does not name a supplied source.", f"text[{match.start()}]"))
            elif not _marker_bound_to_claim(match.start(), match.end(), source_id, text, claim_spans):
                issues.append(_issue("unreferenced_citation_marker", f"Citation marker [{source_id}] is not locally linked to an exact annotated claim listing that source; annotate the supported claim rather than borrowing a citation from another sentence.", f"text[{match.start()}]"))

    updates = draft.get("state_updates") or {}
    if history and updates.get("world"):
        issues.append(_issue("history_world_state_update", "History generation cannot add or overwrite authoritative world facts through a draft state update.", "state_updates.world"))
    existing = dict(task.get("initial_state") or {})
    # Commitments also reserve their IDs against silent replacement.
    world = list(existing.get("world") or [])
    indexed_world = {item.get("id"): item for item in world}
    for commitment in task.get("commitments") or []:
        if commitment.get("id") not in indexed_world:
            world.append(commitment)
            indexed_world[commitment.get("id")] = commitment
        elif indexed_world[commitment.get("id")].get("text") != commitment.get("text"):
            issues.append(_issue("conflicting_initial_commitment", "Initial world state and commitment disagree for the same ID.", "initial_state.world"))
    existing["world"] = world
    issues.extend(validate_state_update(existing, updates, text))

    if plan is not None:
        # A shared ID is an explicit link, not a guessed semantic equivalence.
        planned = {move.get("id"): move for move in plan.get("moves") or []}
        for index, claim in enumerate(claims):
            prior = planned.get(claim.get("id"))
            if prior and _integer_year(prior.get("year")):
                if not _integer_year(claim.get("year")):
                    issues.append(_issue("planned_year_missing", "A claim sharing a dated plan move ID omits its structured year; consistency remains unresolved.", f"claims[{index}].year", "minor"))
                elif prior["year"] != claim["year"]:
                    issues.append(_issue("planned_year_changed", "A claim sharing a plan move ID changes its declared year.", f"claims[{index}].year"))
    return issues


def plan_duplicate_diagnostics(
    plans: Sequence[Mapping[str, Any]], lexical_threshold: float = 0.9
) -> list[Issue]:
    """Warn about exact/lexical repetition; never call this a creativity reward.

    This does not normalize away names, infer topics, or prove that two plans
    have the same plot. Such semantic judgments require independent review.
    """
    if not 0 <= lexical_threshold <= 1:
        raise ValueError("lexical_threshold must be between zero and one")
    representations = []
    for plan in plans:
        parts = [str(plan.get("approach") or ""), str(plan.get("intent") or "")]
        parts.extend(str(move.get("description") or "") for move in plan.get("moves") or [])
        tokens = re.findall(r"\w+", " ".join(parts).casefold())
        representations.append((" ".join(tokens), set(tokens)))
    issues: list[Issue] = []
    for index, (normalized, tokens) in enumerate(representations):
        for earlier in range(index):
            previous, previous_tokens = representations[earlier]
            if normalized and normalized == previous:
                issues.append(_issue("duplicate_plan_text", f"Plans {earlier} and {index} repeat the same normalized descriptive text.", f"plans[{index}]", "minor"))
            elif len(tokens | previous_tokens) >= 10:
                overlap = len(tokens & previous_tokens) / len(tokens | previous_tokens)
                if overlap >= lexical_threshold:
                    issues.append(_issue("high_plan_lexical_overlap", f"Plans {earlier} and {index} have lexical Jaccard overlap {overlap:.3f}; inspect their substantive difference.", f"plans[{index}]", "minor"))
    return issues


check_plan_duplicates = plan_duplicate_diagnostics
