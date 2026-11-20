"""Local source provenance and retrieval invariants; no model/network calls."""

import copy
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from creativity_pipeline.sources import build_source_library, retrieve_sources, verify_source_quote


def documents():
    return [
        {"id": "Official", "document_id": "report-1850", "title": "Official harvest report",
         "text": "The harvest was successful. The minister reported full granaries.\r\nNo district was described as hungry.",
         "author": "Minister A", "date": "1850", "perspective": "Government official defending policy",
         "audience": "Parliament", "purpose": "Explain the budget", "context": "A contested relief bill",
         "url": "https://example.invalid/report", "locator": "page 7", "kind": "primary",
         "independence_group": "ministry-bulletin"},
        {"id": "Letter", "document_id": "letter-1850", "title": "Letter about starvation protests",
         "text": "The district experienced starvation. Crowds protested outside the granary. Relief did not reach the village.",
         "author": "Witness B", "date": "1850", "perspective": "Private testimony", "kind": "primary"},
        {"id": "Market", "document_id": "market-1850", "title": "Market journal",
         "text": "The granary price was three coins. Farmers described a mixed harvest and relief queues.",
         "author": "Trader C", "kind": "primary"},
        {"id": "Botany", "document_id": "botany-1900", "title": "A botany manual",
         "text": "Blue flowers open at dawn beside the river.", "author": "Naturalist D", "kind": "secondary"},
    ]


class SourceLibraryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.library = self.root / "sources.sqlite"

    def build(self, rows=None, **kwargs):
        return build_source_library(documents() if rows is None else rows, self.library, **kwargs)

    def test_jsonl_ingestion_preserves_original_text_and_context(self):
        original = documents()
        path = self.root / "documents.jsonl"
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in original), encoding="utf-8")
        info = build_source_library(path, self.library)
        self.assertEqual((info["n_documents"], info["n_passages"]), (4, 4))
        with sqlite3.connect(self.library) as connection:
            stored = connection.execute("SELECT text FROM documents WHERE document_id='report-1850'").fetchone()[0]
        self.assertEqual(stored, original[0]["text"])
        result = retrieve_sources(self.library, "harvest", max_chars=10000)
        report = next(row for row in result["sources"] if row["document_id"] == "report-1850")
        for field in ("author", "date", "perspective", "audience", "purpose", "context", "url", "kind", "independence_group"):
            self.assertEqual(report[field], original[0][field])
        self.assertIn("page 7; characters 0:", report["locator"])
        self.assertRegex(report["id"], r"^P[0-9a-f]{40}$")

    def test_ids_and_content_identity_are_stable_across_paths_and_input_order(self):
        first = self.build(chunk_chars=70, overlap_chars=12)
        second_path = self.root / "elsewhere.sqlite"
        second = build_source_library(list(reversed(documents())), second_path, chunk_chars=70, overlap_chars=12)
        self.assertEqual(first["library_sha256"], second["library_sha256"])
        one = retrieve_sources(self.library, "harvest granary starvation", max_chars=10000)
        two = retrieve_sources(second_path, "harvest granary starvation", max_chars=10000)
        self.assertEqual(one["sources"], two["sources"])
        self.assertEqual(one["manifest"]["selected_passages"], two["manifest"]["selected_passages"])

    def test_chunks_cover_every_nonwhitespace_character_and_retain_exact_offsets(self):
        original = "  Départ en été.\r\n\r\n" + "First testimony. Another sentence follows. " * 8 + "\nFinal claim.  "
        self.build([{"id": "One", "title": "Letter", "text": original}], chunk_chars=65, overlap_chars=11)
        with sqlite3.connect(self.library) as connection:
            pieces = connection.execute("SELECT start_char,end_char,text FROM passages ORDER BY ordinal").fetchall()
        covered = set()
        self.assertGreater(len(pieces), 1)
        for start, end, text in pieces:
            self.assertEqual(original[start:end], text)
            self.assertLessEqual(len(text), 65)
            self.assertEqual(text, text.strip())
            covered.update(range(start, end))
        self.assertTrue(all(index in covered for index, char in enumerate(original) if not char.isspace()))

    def test_existing_library_is_never_overwritten(self):
        self.build()
        original = self.library.read_bytes()
        with self.assertRaises(FileExistsError):
            self.build([{"id": "Other", "title": "New", "text": "New text."}])
        self.assertEqual(self.library.read_bytes(), original)

    def test_invalid_partial_ingestion_does_not_publish_a_library(self):
        for rows in ([], [documents()[0], {"id": "Bad", "title": "Bad", "text": "  "}],
                     [documents()[0], {**documents()[1], "document_id": "report-1850"}],
                     [documents()[0], {**documents()[1], "id": "Official"}]):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    self.build(rows)
                self.assertFalse(self.library.exists())

    def test_unknown_metadata_and_malformed_jsonl_are_reported(self):
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            self.build([{**documents()[0], "perspectve": "typo"}])
        path = self.root / "broken.jsonl"
        path.write_text(json.dumps(documents()[0]) + "\nnot json\n")
        with self.assertRaisesRegex(ValueError, "line 2"):
            build_source_library(path, self.library)
        self.assertFalse(self.library.exists())

    def test_missing_or_unrelated_databases_fail_as_reviewable_input_errors(self):
        with self.assertRaises(FileNotFoundError):
            retrieve_sources(self.library, "harvest")
        self.library.write_text("This is not a SQLite library.")
        with self.assertRaisesRegex(ValueError, "source library"):
            retrieve_sources(self.library, "harvest")

    def test_duplicate_jsonl_provenance_keys_cannot_silently_override_metadata(self):
        path = self.root / "ambiguous.jsonl"
        for field in ("id", "title", "author", "document_id", "url"):
            with self.subTest(field=field):
                row = {**documents()[1], "url": "https://example.invalid/original"}
                repeated = json.dumps(row)[:-1] + "," + json.dumps(field) + ':"overridden"}'
                path.write_text(json.dumps(documents()[0]) + "\n" + repeated + "\n")
                with self.assertRaisesRegex(ValueError, "line 2.*duplicate JSON key"):
                    build_source_library(path, self.library)
                self.assertFalse(self.library.exists())

    def test_nonfinite_jsonl_values_are_rejected_while_literal_text_is_preserved(self):
        path = self.root / "nonfinite.jsonl"
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token):
                row = {**documents()[1], "author": None}
                raw = json.dumps(row).replace('"author": null', '"author": ' + token)
                path.write_text(json.dumps(documents()[0]) + "\n" + raw + "\n")
                with self.assertRaisesRegex(ValueError, "line 2.*non-finite JSON value"):
                    build_source_library(path, self.library)
                self.assertFalse(self.library.exists())
        path.write_text(json.dumps({"id": "Literal", "title": "Transcription", "text": "The labels NaN and Infinity occur literally in this document."}) + "\n")
        build_source_library(path, self.library)
        result = retrieve_sources(self.library, "Infinity")
        self.assertIn("NaN and Infinity", result["sources"][0]["text"])

    def test_retrieval_does_not_mutate_original_records(self):
        rows = documents()
        before = copy.deepcopy(rows)
        self.build(rows)
        result = retrieve_sources(self.library, "harvest", max_chars=10000)
        result["sources"][0]["text"] = "Changed result only."
        self.assertEqual(rows, before)
        self.assertNotEqual(retrieve_sources(self.library, "harvest")["sources"][0]["text"], "Changed result only.")

    def test_source_diversity_and_per_document_caps_survive_one_large_document(self):
        dense = " ".join(f"Coal tax archive entry {index} confirms coal tax receipts for a different district." for index in range(180))
        rows = [
            {"id": "Dense", "title": "Coal tax archive", "text": dense},
            {"id": "Independent", "title": "Private note", "text": "Workers contested the coal tax assessment."},
            {"id": "Local", "title": "Local account", "text": "The coal tax raised transport costs."},
            {"id": "Irrelevant", "title": "Cloud study", "text": "Clouds formed over distant mountains."},
        ]
        self.build(rows, chunk_chars=90, overlap_chars=5)
        result = retrieve_sources(self.library, "coal tax", max_chars=12000, max_passages=6, per_document_limit=2, min_documents=3)
        counts = {}
        for row in result["sources"]:
            counts[row["document_id"]] = counts.get(row["document_id"], 0) + 1
        self.assertEqual(set(counts), {"Dense", "Independent", "Local"})
        self.assertLessEqual(max(counts.values()), 2)
        self.assertEqual(result["manifest"]["document_coverage_shortfall"], 0)

    def test_additional_targeted_query_can_add_incompatible_evidence(self):
        self.build()
        result = retrieve_sources(self.library, "successful", additional_queries=("starvation protests",),
                                  max_chars=10000, max_passages=2, per_document_limit=1, min_documents=2)
        self.assertEqual({row["document_id"] for row in result["sources"]}, {"report-1850", "letter-1850"})
        self.assertEqual(result["manifest"]["uncovered_query_indices"], [])
        self.assertEqual(len(result["manifest"]["queries"]), 2)
        self.assertTrue(any("incompatible evidence" in value for value in result["manifest"]["limitations"]))

    def test_retrieval_never_pads_with_irrelevant_documents_to_meet_quota(self):
        self.build()
        result = retrieve_sources(self.library, "successful", min_documents=3)
        self.assertEqual({row["document_id"] for row in result["sources"]}, {"report-1850"})
        self.assertEqual(result["manifest"]["document_coverage_shortfall"], 2)
        empty = retrieve_sources(self.library, "unfindablezephyr", min_documents=3)
        self.assertEqual(empty["sources"], [])
        self.assertEqual(empty["manifest"]["n_omitted_documents"], 4)
        self.assertEqual(empty["manifest"]["n_omitted_passages"], 4)

    def test_character_budget_includes_metadata_and_never_truncates_passages(self):
        self.build()
        one = retrieve_sources(self.library, "successful", max_chars=10000, max_passages=1)
        required = one["manifest"]["selected_serialized_chars"]
        exact = retrieve_sources(self.library, "successful", max_chars=required, max_passages=1)
        self.assertEqual(exact["sources"], one["sources"])
        too_small = retrieve_sources(self.library, "successful", max_chars=required - 1, max_passages=1)
        self.assertEqual(too_small["sources"], [])
        self.assertLessEqual(too_small["manifest"]["selected_serialized_chars"], required - 1)
        self.assertIn("character_budget", {row["reason"] for row in too_small["manifest"]["omitted_passages"]})

    def test_manifest_accounts_for_every_passage_and_document(self):
        self.build(chunk_chars=55, overlap_chars=8)
        result = retrieve_sources(self.library, "harvest", max_passages=1)
        manifest = result["manifest"]
        selected = {row["id"] for row in manifest["selected_passages"]}
        omitted = {row["id"] for row in manifest["omitted_passages"]}
        self.assertFalse(selected & omitted)
        self.assertEqual(len(selected | omitted), manifest["n_library_passages"])
        self.assertEqual(manifest["n_selected_documents"] + manifest["n_omitted_documents"], manifest["n_library_documents"])
        self.assertEqual(selected, {row["id"] for row in result["sources"]})
        for source, provenance in zip(result["sources"], manifest["selected_passages"]):
            self.assertEqual(hashlib.sha256(source["text"].encode()).hexdigest(), provenance["text_sha256"])

    def test_identical_wording_retains_distinct_documents_with_unknown_origins(self):
        text = "A famine was reported near the granary."
        rows = [
            {"id": "AccountA", "title": "First famine account", "text": text, "author": "A"},
            {"id": "AccountB", "title": "Second famine account", "text": text, "author": "B"},
            {"id": "Brief", "title": "Famine bulletin", "text": "A famine bulletin copied an official estimate.", "independence_group": "official-bulletin"},
            {"id": "Reply", "title": "Famine reply", "text": "A famine reply repeated the same estimate.", "independence_group": "official-bulletin"},
        ]
        self.build(rows)
        result = retrieve_sources(self.library, "famine", max_chars=10000, min_documents=4)
        manifest = result["manifest"]
        self.assertEqual(manifest["n_selected_documents"], 4)
        self.assertEqual(manifest["declared_independence_groups"], ["official-bulletin"])
        self.assertEqual(manifest["independence_unknown_documents"], ["AccountA", "AccountB"])
        self.assertEqual(manifest["duplicate_document_text_groups"], [["AccountA", "AccountB"]])
        self.assertEqual(manifest["document_coverage_shortfall"], 0)
        self.assertTrue(any("does not establish copying or independence" in warning for warning in manifest["warnings"]))

    def test_identical_statements_retain_separate_documents_and_declared_origin_groups(self):
        text = "The bridge was closed on Monday."
        self.build([
            {"id": "Letter", "title": "A private letter", "author": "Witness A", "text": text, "independence_group": "account-A"},
            {"id": "Diary", "title": "A diary entry", "author": "Witness B", "text": text, "independence_group": "account-B"},
            {"id": "Reprint", "title": "Reprint of the letter", "author": "Printer", "text": text, "independence_group": "account-A"},
        ])
        result = retrieve_sources(self.library, "bridge", max_chars=10000, max_passages=3, per_document_limit=1, min_documents=3)
        self.assertEqual({source["document_id"] for source in result["sources"]}, {"Letter", "Diary", "Reprint"})
        self.assertEqual({source["author"] for source in result["sources"]}, {"Witness A", "Witness B", "Printer"})
        self.assertEqual(result["manifest"]["declared_independence_groups"], ["account-A", "account-B"])
        self.assertEqual(sum(source["independence_group"] == "account-A" for source in result["sources"]), 2)
        self.assertEqual(result["manifest"]["document_coverage_shortfall"], 0)

    def test_repeated_identical_passages_within_one_document_remain_deduplicated(self):
        self.build([{"id": "One", "title": "Repeated ledger marks", "text": "bridgelevy\n" * 5}],
                   chunk_chars=11, overlap_chars=0)
        result = retrieve_sources(self.library, "bridgelevy", max_chars=10000, max_passages=5, per_document_limit=5, min_documents=1)
        self.assertGreater(result["manifest"]["n_library_passages"], 1)
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["text"], "bridgelevy")
        self.assertEqual({row["reason"] for row in result["manifest"]["omitted_passages"]}, {"duplicate_passage_text_in_document"})

    def test_quotes_verify_exact_original_spans_without_asserting_truth(self):
        self.build()
        source = retrieve_sources(self.library, "successful")["sources"][0]
        quote = "The harvest was successful."
        result = verify_source_quote(self.library, source["id"], quote)
        self.assertTrue(result["valid"])
        self.assertFalse(result["establishes_truth_or_entailment"])
        original = documents()[0]["text"]
        self.assertEqual(original[result["quote_start_char"]:result["quote_end_char"]], quote)
        self.assertFalse(verify_source_quote(self.library, source["id"], "The  harvest was successful.")["valid"])
        self.assertFalse(verify_source_quote(self.library, source["id"], "The district experienced starvation.")["valid"])
        self.assertEqual(verify_source_quote(self.library, "Invented", quote)["reason"], "unknown_source_id")

    def test_modified_original_document_is_not_accepted_under_old_provenance(self):
        self.build()
        source = retrieve_sources(self.library, "successful")["sources"][0]
        with sqlite3.connect(self.library) as connection:
            connection.execute("UPDATE documents SET text = text || ' Added later.' WHERE document_id='report-1850'")
        with self.assertRaisesRegex(ValueError, "document hash mismatch"):
            verify_source_quote(self.library, source["id"], "successful")

    def test_passage_offsets_cannot_be_changed_while_reusing_an_old_id(self):
        self.build()
        source = retrieve_sources(self.library, "successful")["sources"][0]
        changed = source["text"][4:]
        with sqlite3.connect(self.library) as connection:
            connection.execute("UPDATE passages SET start_char=4,text=?,text_sha256=? WHERE id=?",
                               (changed, hashlib.sha256(changed.encode()).hexdigest(), source["id"]))
        with self.assertRaisesRegex(ValueError, "identity does not match"):
            verify_source_quote(self.library, source["id"], "harvest")

    def test_lexical_fallback_works_when_fts5_is_unavailable(self):
        original_connect = sqlite3.connect

        class NoFTSConnection(sqlite3.Connection):
            def execute(self, sql, *args, **kwargs):
                if sql.startswith("CREATE VIRTUAL TABLE"):
                    raise sqlite3.OperationalError("no such module: fts5")
                return super().execute(sql, *args, **kwargs)

        def connect(*args, **kwargs):
            return original_connect(*args, factory=NoFTSConnection, **kwargs)

        with patch("creativity_pipeline.sources.sqlite3.connect", side_effect=connect):
            info = self.build()
            result = retrieve_sources(self.library, "harvest", max_chars=10000)
        self.assertFalse(info["fts5_enabled"])
        self.assertEqual(result["manifest"]["retrieval_engine"], "lexical_fallback")
        self.assertEqual({row["document_id"] for row in result["sources"]}, {"report-1850", "market-1850"})
        self.assertTrue(result["manifest"]["warnings"])

    def test_user_query_is_tokenized_and_not_executed_as_fts_syntax(self):
        self.build()
        result = retrieve_sources(self.library, 'harvest" OR * NOT (missing)')
        self.assertTrue(result["sources"])
        self.assertIn("harvest", result["manifest"]["queries"][0]["terms"])
        self.assertEqual(result["manifest"]["n_library_documents"], 4)

    def test_repeated_queries_do_not_inflate_ranking_and_invalid_limits_fail(self):
        self.build()
        one = retrieve_sources(self.library, "harvest")
        repeated = retrieve_sources(self.library, "harvest", additional_queries=("harvest", "harvest"))
        self.assertEqual(one["sources"], repeated["sources"])
        self.assertEqual(len(repeated["manifest"]["queries"]), 1)
        for change in ({"max_chars": 1}, {"max_passages": 0}, {"per_document_limit": True},
                       {"min_documents": -1}, {"additional_queries": "not a sequence"}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    retrieve_sources(self.library, "harvest", **change)
        with self.assertRaises(ValueError):
            retrieve_sources(self.library, {"bad": "query"})


if __name__ == "__main__":
    unittest.main()
