# Validation record

Validation for **version 0.3.1**, including early mechanical repair, domain-specific prompts, saved-draft continuation, and read-only run diagnostics.

## Executed in the development environment

- Python 3.12.14, Pydantic 2.13.4, PyYAML 6.0.3.
- **332 tests passed; five optional real-library tests were skipped** because their dependencies are absent. There are 337 tests in total. The full output is in `test-results.txt`.
- New routing regressions reproduce missing claim spans and fabricated source IDs, repair them before critic calls, retain a shared repair cap, stop on persistent defects, and verify that old approvals cannot survive an invalid later revision. Prompt regressions distinguish source-free English fiction, dated fiction, supplied literary sources, and historical evidence.
- Recovery regressions verify that a saved draft skips planning/drafting, preserves its source packet and retrieval manifest, rejects mismatched hashes and in-memory mutations, discards old approvals, uses the original initial state, retains fixture labels, refuses overwrites, and separates continuation provenance and costs from fresh generation.
- The standalone diagnostic script runs under `python -I -S` without site packages. Tests cover actual event keys, bounded/redacted output, interrupted logs, missing/corrupt files, automatic selection, symlink boundaries, and unchanged run-file bytes and modification times.
- Both English and history CLI demos completed the four-plan pipeline and produced complete logs, matching final-text/review hashes, and fixture-labeled summaries.
- Version 0.3.1 was reinstalled without dependency upgrades. Both modes then completed `demo` followed by `repair-run`; continuations made only critic and general-review calls, retained fixture labels, and produced matching final/review hashes. The isolated diagnostic script read both runs without warnings. Results are in `recovery-smoke-results.json`.
- The CLI integration suite executed all three planning strategies, retained failed runs, and checked token totals against individual response records.
- Additional tests reproduced and prevented local citation borrowing, loss of accepted plans after malformed candidates, stale evidence-audit approval, pooling of base/adapter conditions, zero-scan denominator loss, silent replacement of narrative state, and incomplete designation after interruption.
- Structured critic tests cover exact draft/source quotations, required rubrics and source coverage, partial assessments, abstention, mandatory material historical errors, shared-origin contradictions, and the distinction between valid negative criticism and an invalid critic report. Pipeline integration confirms that material criticism blocks acceptance, invalid criticism never becomes an automatic rewrite instruction, repairs receive fresh critic hashes, and failed reviews cannot inherit old approvals.
- Source-library tests executed actual SQLite/FTS5 ingestion and retrieval, a lexical fallback without FTS5, query coverage, document diversity, source budgets, original offsets and hashes, quote verification, tampering checks, duplicate JSON-key rejection, and preservation of distinct documents with identical wording. The six-document invented practice dossier was built and retrieved successfully.
- Sequential critic routing tests verify different tokenizer counts, one resident child, explicit adapter disabling, retained checkpoint identity, failure propagation, and release before cache clearing. These use backend/CUDA fakes and do not establish real GPU memory behavior.
- The eight-case critic development set was validated as input. Evaluator tests verify hidden expected judgments, saved designations, false-positive/negative bookkeeping, invalid reports, interruption, fixture rejection, and retained missing results. The actual model has **not** been run on those cases.
- Standalone critique and generated-output export preserve exact UTF-8 text, including CRLF line endings, so input hashes do not silently describe normalized text. Retrieved-source scope notes account for retained inline sources.
- Blinded review export/import was exercised through both its Python APIs and CLI, including changed text/context/packet rejection, domain-specific ratings, and entirely unreviewed manifests.
- Editable package installation succeeded in an isolated virtual environment using the already available base dependencies. The installed `creativity --help` command ran successfully.
- No tokenizer, model weights, or optional GPU packages were downloaded.

## What those tests establish

| Area | Evidence |
|---|---|
| Pipeline | Material mechanical errors are repaired before reviews; accepted revisions have fresh required reviews; stale reviews cannot approve changed text; failed drafts cannot commit state |
| Recovery | Verified saved drafts can continue without planning/drafting; parent artifacts and evidence remain intact; new budgets and parent costs are distinguished |
| Diagnostics | Saved metadata is inspected with no model imports, inference, or file mutation; unavailable records and timing limits remain explicit |
| Critic contract | Evidence quotes, report coverage, decision consistency, serious issue severity, and declared source dependence are checked; valid criticism and invalid reports have different paths |
| Source library | Real lexical retrieval preserves document provenance, exact source offsets, metadata, selected/omitted records, and distinct sources with matching wording |
| Budgets | Retries and malformed outputs remain counted; insufficient budgets block further calls; context is not silently truncated |
| History contracts | Unknown sources, inexact support quotes, missing annotations, and explicit declared-date contradictions are rejected |
| Fiction contracts | Protected state IDs remain immutable; new entries require exact evidence spans |
| Backend contract | Exact fake-tokenizer accounting, incomplete reasoning/output rejection, adapter restoration, revision/template compatibility, sampling overrides, and adapter inventory/hash guards |
| Training data | Split leakage checks, explicit acceptance/provenance, prompt/approach balancing, completion-only loss masks, EOS supervision, no silent truncation |
| Training configuration | Offline preflight works without GPU imports; two-step configuration expresses two optimizer updates with accumulation 1 |
| Evaluation | Frozen hashes, separate writer/critic condition identities, detector versions/dates, explicit human origin, blind human ratings, designated critic cases, and failed/unscanned/unreviewed denominators |
| CLI | Demo commands, strategy comparisons, input validation, export format, fixture exclusion, and overwrite protection |

Executed backend tests use small fakes, not a real Transformers model. Five
additional real-library tests are implemented with tiny, randomly initialized
CPU models and no downloads, but could not run here. They cover adapter/logit
round trips, missing/unexpected tensors, nonfinite adapter parameters, and
Trainer loss normalization. The fixed demo prose does not test writing quality.
A passing software contract test is not historical verification or a creativity
result.

## Still requiring the target workstation

1. Install the optional CUDA stack and verify `creativity doctor --check-cuda`.
   Run the five optional real-library tests; skips must not be treated as successful execution.
2. Diagnose the user's already completed generation from its saved status/error and per-call records. Exercise the revised recovery path on the actual GPU and measure token counts, elapsed time, peak memory, and JSON adherence. The reported annotation errors are real observations; the omitted stopping error and actual hardware remain unverified here.
3. Run `evaluate-critic` on the synthetic development cases, inspect evidence findings and false positives, then assess new real source packets with knowledgeable human readers. Compare any separate critic against the shared-backbone default using measured cost and failures.
4. Supply a curated, grouped training/validation dataset; run offline and tokenizer preflights.
5. Execute the two-optimizer-step GPU smoke configuration and inspect its report.
6. Train an adapter only after the smoke run fits; compare its held-out writing with the untrained baseline.
7. Obtain independent human quality/factual assessments and versioned detector observations on frozen outputs.

No claims are made here about GPU memory fit, throughput, fine-tuning quality,
expert historical reasoning, or detector performance. Those experiments were not
run in this environment.
