"""Local document ingestion and auditable lexical passage retrieval.

One input record represents one complete document. Original text is retained
verbatim; retrieved Source records are exact, bounded slices with locators.
Retrieval does not establish truth, corroboration, entailment, or independence.
The caller opts into retrieval and decides how to merge it with existing sources.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import re
import sqlite3
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_SCHEMA_VERSION = 1
_SOURCE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_FIELDS = {
    "id", "title", "text", "author", "date", "perspective", "document_id",
    "url", "locator", "kind", "independence_group", "audience", "purpose", "context",
}
_DEFAULTS = {
    "author": "unspecified", "date": "unspecified", "perspective": "not assessed",
    "url": "", "locator": "", "kind": "unspecified", "audience": "unspecified",
    "purpose": "unspecified", "context": "unspecified",
}
_STOPWORDS = frozenset("a an and are as at be been but by did do does for from had has have how if in into is it its of on or that the their them these they this those to was were what when where which who why will with would".split())


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}; provenance fields must not be overwritten")
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON value {value!r} is not permitted")


def _records(records_or_jsonl: Iterable[Mapping[str, Any]] | str | Path):
    if isinstance(records_or_jsonl, (str, Path)):
        with Path(records_or_jsonl).open("r", encoding="utf-8-sig") as stream:
            for line, value in enumerate(stream, 1):
                if not value.strip():
                    continue
                try:
                    yield json.loads(value, object_pairs_hook=_unique_json_object, parse_constant=_reject_json_constant)
                except ValueError as exc:
                    raise ValueError(f"Source JSONL line {line}: invalid JSON: {exc}") from exc
    else:
        if isinstance(records_or_jsonl, Mapping):
            raise ValueError("Supply an iterable of document records, not one mapping")
        yield from records_or_jsonl


def _document(raw: Mapping[str, Any], number: int) -> dict[str, Any]:
    where = f"Source document {number}"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be an object")
    unknown = set(raw) - _FIELDS
    if unknown:
        raise ValueError(f"{where} has unsupported fields: {sorted(unknown)}")
    row = dict(raw)
    for field in ("id", "title", "text"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"{where}: {field} must be a nonempty string")
    row["id"], row["title"] = row["id"].strip(), row["title"].strip()
    if not _SOURCE_ID.fullmatch(row["id"]):
        raise ValueError(f"{where}: id must match the Source identifier format")
    for field, default in _DEFAULTS.items():
        value = row.get(field, default)
        if not isinstance(value, str):
            raise ValueError(f"{where}: {field} must be a string")
        row[field] = value.strip()
    for field in ("document_id", "independence_group"):
        value = row.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{where}: {field} must be null or a nonempty string")
        row[field] = value.strip() if isinstance(value, str) else None
    row["document_id"] = row["document_id"] or row["id"]
    return row


def _chunks(text: str, chunk_chars: int, overlap_chars: int):
    """Yield exact Unicode character offsets, preferring natural boundaries."""
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            floor = start + max(overlap_chars + 1, chunk_chars // 2)
            if floor < end:
                boundary = max(text.rfind("\n\n", floor, end), text.rfind(". ", floor, end))
                if boundary >= floor:
                    end = boundary + (2 if text[boundary:boundary + 2] == "\n\n" else 1)
                else:
                    spaces = [match.start() for match in re.finditer(r"\s", text[floor:end])]
                    if spaces:
                        end = floor + spaces[-1]
        left, right = start, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            yield left, right, text[left:right]
        if end == len(text):
            break
        start = max(start + 1, end - overlap_chars)


def _catalog_identity(connection: sqlite3.Connection, chunk_chars: int, overlap_chars: int) -> str:
    records = [dict(row) for row in connection.execute(
        "SELECT document_id, record_sha256 FROM documents ORDER BY document_id"
    )]
    return _hash(_canonical({"schema_version": _SCHEMA_VERSION, "chunk_chars": chunk_chars,
                             "overlap_chars": overlap_chars, "documents": records}))


def build_source_library(
    records_or_jsonl: Iterable[Mapping[str, Any]] | str | Path,
    output_path: str | Path,
    *,
    chunk_chars: int = 1800,
    overlap_chars: int = 200,
) -> dict[str, Any]:
    """Create an exclusive SQLite library from source records or a JSONL file.

    Text is not fetched from URLs. Supply lawfully obtained complete documents
    and their provenance. ``document_id`` defaults to ``id`` and must be unique;
    excerpts of one document must not become several independent documents.
    IDs and library content hashes are deterministic across input order/paths.
    """
    _integer(chunk_chars, "chunk_chars")
    _integer(overlap_chars, "overlap_chars", 0)
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be smaller than chunk_chars")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Source library already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=".source-library-", suffix=".sqlite", dir=destination.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    connection = sqlite3.connect(temporary)
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY, source_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL, text TEXT NOT NULL, metadata_json TEXT NOT NULL,
                text_sha256 TEXT NOT NULL, record_sha256 TEXT NOT NULL
            );
            CREATE TABLE passages (
                id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(document_id),
                ordinal INTEGER NOT NULL, start_char INTEGER NOT NULL, end_char INTEGER NOT NULL,
                text TEXT NOT NULL, text_sha256 TEXT NOT NULL
            );
            CREATE INDEX passages_document ON passages(document_id, ordinal);
        """)
        fts_enabled = True
        try:
            connection.execute("CREATE VIRTUAL TABLE passage_fts USING fts5(id UNINDEXED, text, title, tokenize='unicode61')")
        except sqlite3.OperationalError as exc:
            if "no such module" not in str(exc).lower():
                raise
            fts_enabled = False
        n_documents = n_passages = 0
        for number, raw in enumerate(_records(records_or_jsonl), 1):
            row = _document(raw, number)
            record_hash, text_hash = _hash(_canonical(row)), _hash(row["text"])
            metadata = {key: value for key, value in row.items() if key != "text"}
            try:
                connection.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (row["document_id"], row["id"], row["title"], row["text"], _canonical(metadata), text_hash, record_hash))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Duplicate source id or document_id: {row['document_id']!r}; one record must contain the complete document") from exc
            n_documents += 1
            for ordinal, (start, end, text) in enumerate(_chunks(row["text"], chunk_chars, overlap_chars)):
                passage_id = "P" + _hash(_canonical([row["document_id"], text_hash, start, end]))[:40]
                connection.execute("INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (passage_id, row["document_id"], ordinal, start, end, text, _hash(text)))
                if fts_enabled:
                    connection.execute("INSERT INTO passage_fts VALUES (?, ?, ?)", (passage_id, text, row["title"]))
                n_passages += 1
        if not n_documents:
            raise ValueError("Source library needs at least one nonempty document")
        info = {
            "schema_version": _SCHEMA_VERSION, "chunk_chars": chunk_chars, "overlap_chars": overlap_chars,
            "n_documents": n_documents, "n_passages": n_passages, "fts5_enabled": fts_enabled,
            "library_sha256": _catalog_identity(connection, chunk_chars, overlap_chars),
        }
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", ((key, _canonical(value)) for key, value in info.items()))
        connection.commit()
        connection.close()
        # A hard link installs the completed file atomically and cannot overwrite
        # a path another process created after the initial existence check.
        os.link(temporary, destination)
        return {"library": str(destination), **info,
                "retrieval_engine": "sqlite_fts5_bm25" if fts_enabled else "lexical_fallback"}
    finally:
        connection.close()
        temporary.unlink(missing_ok=True)


@contextmanager
def _open_library(library: str | Path):
    path = Path(library).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source library is not a file: {path}")
    try:
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"Cannot open source library: {exc}") from exc
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        info = {row["key"]: json.loads(row["value"]) for row in connection.execute("SELECT key, value FROM metadata")}
        required = {"schema_version", "chunk_chars", "overlap_chars", "n_documents", "n_passages", "fts5_enabled", "library_sha256"}
        if not required <= info.keys() or info.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("Unsupported source library schema")
        if not isinstance(info["fts5_enabled"], bool):
            raise ValueError("Invalid source library index metadata")
        expected = _catalog_identity(connection, info["chunk_chars"], info["overlap_chars"])
        if info.get("library_sha256") != expected:
            raise ValueError("Source library catalog hash mismatch")
        if info.get("n_documents") != connection.execute("SELECT count(*) FROM documents").fetchone()[0]:
            raise ValueError("Source library document count mismatch")
        if info.get("n_passages") != connection.execute("SELECT count(*) FROM passages").fetchone()[0]:
            raise ValueError("Source library passage count mismatch")
        yield connection, info
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"Invalid or unreadable source library: {exc}") from exc
    finally:
        connection.close()


def _terms(query: str) -> tuple[list[str], bool]:
    terms = list(dict.fromkeys(token.casefold() for token in _TOKEN.findall(query)))
    meaningful = [term for term in terms if term not in _STOPWORDS]
    terms = meaningful or terms
    return terms[:64], len(terms) > 64


def _fts_candidates(connection: sqlite3.Connection, terms: Sequence[str], limit: int, min_documents: int):
    expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
    sql = """SELECT p.id, p.document_id, bm25(passage_fts, 0.0, 1.0, 0.25) AS relevance
        FROM passage_fts JOIN passages p ON p.id = passage_fts.id
        WHERE passage_fts MATCH ?"""
    candidates = [dict(row) for row in connection.execute(sql + " ORDER BY relevance, p.id LIMIT ?", (expression, limit))]
    documents = {row["document_id"] for row in candidates}
    while len(documents) < min_documents:
        excluded = sorted(documents)
        exclusion = " AND p.document_id NOT IN (" + ",".join("?" for _ in excluded) + ")" if excluded else ""
        row = connection.execute(sql + exclusion + " ORDER BY relevance, p.id LIMIT 1", (expression, *excluded)).fetchone()
        if row is None:
            break
        candidates.append(dict(row))
        documents.add(row["document_id"])
    n_matches = connection.execute("SELECT count(*) FROM passage_fts WHERE passage_fts MATCH ?", (expression,)).fetchone()[0]
    n_documents = connection.execute("SELECT count(DISTINCT p.document_id) FROM passage_fts JOIN passages p ON p.id=passage_fts.id WHERE passage_fts MATCH ?", (expression,)).fetchone()[0]
    return candidates, n_matches, n_documents


def _lexical_candidates(connection: sqlite3.Connection, terms: Sequence[str], limit: int, min_documents: int):
    """Streaming fallback: retain a bounded heap plus one best hit per document."""
    best, by_document = [], {}
    n_matches = 0
    for row in connection.execute("SELECT p.id, p.document_id, p.text, d.title FROM passages p JOIN documents d USING(document_id) ORDER BY p.id"):
        words = _TOKEN.findall(row["text"].casefold())
        frequencies = Counter(words)
        title_words = Counter(_TOKEN.findall(row["title"].casefold()))
        score = sum((frequencies[term] + 0.25 * title_words[term]) /
                    (frequencies[term] + 0.25 * title_words[term] + 0.5 + 1.5 * len(words) / 180)
                    for term in terms if frequencies[term] or title_words[term])
        if not score:
            continue
        n_matches += 1
        item = (score, row["id"], row["document_id"])
        if len(best) < limit:
            heapq.heappush(best, item)
        elif item > best[0]:
            heapq.heapreplace(best, item)
        if row["document_id"] not in by_document or item > by_document[row["document_id"]]:
            by_document[row["document_id"]] = item
    chosen = {item[1]: item for item in best}
    documents = {item[2] for item in chosen.values()}
    for item in sorted(by_document.values(), reverse=True):
        if len(documents) >= min_documents:
            break
        if item[2] not in documents:
            chosen[item[1]] = item
            documents.add(item[2])
    candidates = [{"id": item[1], "document_id": item[2], "relevance": -item[0]}
                  for item in sorted(chosen.values(), key=lambda item: (-item[0], item[1]))]
    return candidates, n_matches, len(by_document)


def _source_for_passage(connection: sqlite3.Connection, source_id: str):
    passage = connection.execute("SELECT * FROM passages WHERE id=?", (source_id,)).fetchone()
    if passage is None:
        raise ValueError(f"Unknown source passage: {source_id}")
    document = connection.execute("SELECT * FROM documents WHERE document_id=?", (passage["document_id"],)).fetchone()
    metadata = json.loads(document["metadata_json"])
    original = {**metadata, "text": document["text"]}
    if _hash(document["text"]) != document["text_sha256"] or _hash(_canonical(original)) != document["record_sha256"]:
        raise ValueError("Original source document hash mismatch")
    start, end = passage["start_char"], passage["end_char"]
    expected_id = "P" + _hash(_canonical([passage["document_id"], document["text_sha256"], start, end]))[:40]
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(document["text"]):
        raise ValueError("Source passage has invalid original-document offsets")
    if passage["id"] != expected_id:
        raise ValueError("Source passage identity does not match original document offsets")
    if document["text"][start:end] != passage["text"] or _hash(passage["text"]) != passage["text_sha256"]:
        raise ValueError("Source passage does not match original document offsets/hash")
    location = f"characters {passage['start_char']}:{passage['end_char']} (0-based, end-exclusive)"
    if metadata["locator"]:
        location = metadata["locator"] + "; " + location
    source = {**metadata, "id": passage["id"], "text": passage["text"], "locator": location}
    provenance = {
        "id": passage["id"], "document_id": passage["document_id"], "original_source_id": metadata["id"],
        "title": metadata["title"], "url": metadata["url"], "locator": location,
        "start_char": passage["start_char"], "end_char": passage["end_char"],
        "text_sha256": passage["text_sha256"], "document_text_sha256": document["text_sha256"],
        "document_record_sha256": document["record_sha256"], "independence_group": metadata["independence_group"],
    }
    return source, provenance


def retrieve_sources(
    library: str | Path,
    query: str,
    *,
    max_chars: int = 8000,
    max_passages: int = 6,
    per_document_limit: int = 2,
    min_documents: int = 3,
    additional_queries: Sequence[str] = (),
) -> dict[str, Any]:
    """Retrieve diverse passages and a complete selected/omitted ID manifest.

    ``max_chars`` caps the compact JSON source list, including provenance, not
    model tokens. Exact tokenizer preflight belongs to the inference backend.
    ``min_documents`` is a coverage target, never a reason to add irrelevant
    documents. Additional explicit queries can search for incompatible evidence;
    they receive coverage priority but do not certify an exhaustive search.
    """
    for name, value in (("max_chars", max_chars), ("max_passages", max_passages), ("per_document_limit", per_document_limit)):
        _integer(value, name, 2 if name == "max_chars" else 1)
    _integer(min_documents, "min_documents", 0)
    if isinstance(additional_queries, (str, bytes)):
        raise ValueError("additional_queries must be a sequence of query strings")
    queries = [query, *additional_queries]
    if any(not isinstance(value, str) or not value.strip() for value in queries):
        raise ValueError("Every retrieval query must be a nonempty string")
    queries = list(dict.fromkeys(queries))
    if len(queries) > 9:
        raise ValueError("Use at most eight additional targeted queries per retrieval")
    terms = [_terms(value) for value in queries]
    if any(not value[0] for value in terms):
        raise ValueError("Retrieval queries need searchable letters or numbers")
    limit = min(5000, max(100, max_passages * 20, min_documents * 20))
    warnings = []
    with _open_library(library) as (connection, info):
        candidates, query_reports = {}, []
        engine = "sqlite_fts5_bm25" if info["fts5_enabled"] else "lexical_fallback"
        for query_index, ((words, truncated), value) in enumerate(zip(terms, queries)):
            if info["fts5_enabled"]:
                try:
                    matches, n_matches, n_matching_documents = _fts_candidates(connection, words, limit, min(min_documents, max_passages))
                except sqlite3.OperationalError as exc:
                    if "no such module" not in str(exc).lower():
                        raise
                    matches, n_matches, n_matching_documents = _lexical_candidates(connection, words, limit, min(min_documents, max_passages))
                    engine = "lexical_fallback"
            else:
                matches, n_matches, n_matching_documents = _lexical_candidates(connection, words, limit, min(min_documents, max_passages))
            query_reports.append({"query": value, "terms": words, "terms_capped_at_64": truncated,
                                  "n_matching_passages": n_matches, "n_matching_documents": n_matching_documents,
                                  "n_candidate_passages": len(matches)})
            for rank, match in enumerate(matches, 1):
                item = candidates.setdefault(match["id"], {"id": match["id"], "document_id": match["document_id"], "score": 0.0, "query_indices": []})
                item["score"] += 1 / (60 + rank)
                item["query_indices"].append(query_index)
        if engine == "lexical_fallback":
            warnings.append("FTS5 unavailable: lexical fallback scans passage text; ranking can differ from BM25.")
        ranked = sorted(candidates.values(), key=lambda item: (-item["score"], item["id"]))
        selected, provenance, selected_ids = [], [], set()
        doc_counts, selected_text_hashes = Counter(), set()
        covered_queries = set()
        rejected, cache = {}, {}

        def select(item):
            if item["id"] in selected_ids:
                return True
            if len(selected) >= max_passages:
                rejected[item["id"]] = "passage_limit"
                return False
            if doc_counts[item["document_id"]] >= per_document_limit:
                rejected[item["id"]] = "per_document_limit"
                return False
            if item["id"] not in cache:
                cache[item["id"]] = _source_for_passage(connection, item["id"])
            source, source_provenance = cache[item["id"]]
            # Repeated text within one document adds no new document provenance.
            # Across documents, identical wording can still have distinct dates,
            # authors, purposes, or transmission histories: retain those records.
            text_identity = (item["document_id"], source_provenance["text_sha256"])
            if text_identity in selected_text_hashes:
                rejected[item["id"]] = "duplicate_passage_text_in_document"
                return False
            if len(_canonical([*selected, source])) > max_chars:
                rejected[item["id"]] = "character_budget"
                return False
            selected.append(source)
            provenance.append({**source_provenance, "query_indices": item["query_indices"], "reciprocal_rank_score": item["score"]})
            selected_ids.add(item["id"])
            selected_text_hashes.add(text_identity)
            doc_counts[item["document_id"]] += 1
            covered_queries.update(item["query_indices"])
            rejected.pop(item["id"], None)
            return True

        # Reserve coverage for each explicit query before filling on relevance.
        for query_index in range(len(queries)):
            if query_index in covered_queries:
                continue
            for item in ranked:
                if query_index in item["query_indices"] and select(item):
                    break
        # Then favour different documents; do not infer witness independence.
        for item in ranked:
            if len(doc_counts) >= min_documents:
                break
            if item["document_id"] not in doc_counts:
                select(item)
        for item in ranked:
            select(item)

        all_passages = [dict(row) for row in connection.execute("SELECT id, document_id FROM passages ORDER BY document_id, ordinal")]
        omitted_passages = [{**row, "reason": rejected.get(row["id"], "outside_candidate_pool_or_no_lexical_match")}
                            for row in all_passages if row["id"] not in selected_ids]
        documents = [dict(row) for row in connection.execute("SELECT document_id, source_id, title, text_sha256, metadata_json FROM documents ORDER BY document_id")]
        omitted_documents = [{"document_id": row["document_id"], "original_source_id": row["source_id"], "title": row["title"],
                              "reason": "no_passage_selected"} for row in documents if row["document_id"] not in doc_counts]
        if len(doc_counts) < min_documents:
            warnings.append(f"Document coverage shortfall: selected {len(doc_counts)} of the requested {min_documents}; relevance, budgets, duplicates, or candidate limits prevented more.")
        uncovered = sorted(set(range(len(queries))) - covered_queries)
        if uncovered:
            warnings.append("Some explicit queries contributed no selected passage; inspect uncovered_query_indices.")
        if any(report["terms_capped_at_64"] for report in query_reports):
            warnings.append("A query exceeded 64 unique search terms; excess terms were not searched.")
        groups = sorted({source["independence_group"] for source in selected if source["independence_group"] is not None})
        unknown_groups = sorted({source["document_id"] for source in selected if source["independence_group"] is None})
        text_groups = defaultdict(list)
        for row in documents:
            text_groups[row["text_sha256"]].append(row["document_id"])
        duplicate_documents = sorted(group for group in text_groups.values() if len(group) > 1)
        if duplicate_documents:
            warnings.append("Identical full-document text occurs in distinct records; inspect provenance. Identical wording alone does not establish copying or independence.")
        manifest = {
            "schema_version": 1, "library": str(Path(library).expanduser().resolve()),
            "library_sha256": info["library_sha256"], "retrieval_engine": engine,
            "queries": query_reports, "candidate_limit_per_query": limit,
            "limits": {"max_chars": max_chars, "max_passages": max_passages, "per_document_limit": per_document_limit, "min_documents": min_documents},
            "character_budget_scope": "compact JSON Source list including provenance; not model tokens",
            "selected_serialized_chars": len(_canonical(selected)),
            "selected_text_chars": sum(len(source["text"]) for source in selected),
            "n_library_documents": info["n_documents"], "n_library_passages": info["n_passages"],
            "n_selected_documents": len(doc_counts), "n_selected_passages": len(selected),
            "n_omitted_documents": len(omitted_documents), "n_omitted_passages": len(omitted_passages),
            "selected_documents": sorted(doc_counts), "selected_passages": provenance,
            "omitted_documents": omitted_documents, "omitted_passages": omitted_passages,
            "declared_independence_groups": groups, "independence_unknown_documents": unknown_groups,
            "duplicate_document_text_groups": duplicate_documents,
            "uncovered_query_indices": uncovered, "document_coverage_shortfall": max(0, min_documents - len(doc_counts)),
            "warnings": warnings,
            "limitations": [
                "Lexical relevance is not truth, corroboration, or exhaustive recall; synonyms and unqueried accounts can be missed.",
                "Only selected passages are supplied. Omitted material may contain incompatible evidence or context that changes an interpretation.",
                "Document counts do not establish independent accounts. Unset independence groups remain unknown, even for different author names.",
                "An exact quotation verifies presence in a document, not the truth of its author's assertion or support for an interpretation.",
                "Character budgeting is not tokenizer budgeting; perform exact final prompt-token preflight after merging sources.",
            ],
        }
        return {"sources": selected, "manifest": manifest}


def verify_source_quote(library: str | Path, source_id: str, quote: str) -> dict[str, Any]:
    """Verify an exact quotation against a selected passage and original text.

    Whitespace and spelling are preserved. The result is a provenance check,
    not a finding that the quoted statement is true or entails a proposed claim.
    """
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be a nonempty string")
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError("quote must be a nonempty string")
    with _open_library(library) as (connection, info):
        if connection.execute("SELECT 1 FROM passages WHERE id=?", (source_id,)).fetchone() is None:
            return {"valid": False, "reason": "unknown_source_id", "source_id": source_id, "library_sha256": info["library_sha256"]}
        source, provenance = _source_for_passage(connection, source_id)
        offset = source["text"].find(quote)
        result = {"valid": offset >= 0, "reason": "exact_match" if offset >= 0 else "quote_not_in_passage",
                  "source_id": source_id, "library_sha256": info["library_sha256"], "provenance": provenance,
                  "match_policy": "exact_text", "establishes_truth_or_entailment": False}
        if offset >= 0:
            result["quote_start_char"] = provenance["start_char"] + offset
            result["quote_end_char"] = provenance["start_char"] + offset + len(quote)
        return result
