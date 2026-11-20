# Source libraries, careful attribution, and the critic

The source library lets a history task draw a bounded evidence packet from many local documents. The dedicated critic then examines the actual prose against the supplied packet. Neither retrieval nor a matching quotation establishes that a source's assertion is true. The distinction between **what a document says**, **what it supports**, and **what remains uncertain** must survive drafting, criticism, and repair.

This document describes the local implementation in [sources.py](../src/creativity_pipeline/sources.py), the [source and task contracts](../src/creativity_pipeline/models.py), and the [generation prompts](../src/creativity_pipeline/prompts.py). It makes no claim that a model has already demonstrated expert historical judgment.

## Try the invented practice dossier

[source_library.jsonl](../examples/source_library.jsonl) contains six entirely invented documents about a fictional quay protest: a prejudiced official memorandum, a payroll register, a dependent reprint, a petition, a clerk's register, and a comparison collection roll. Names, quotations, dates, and events were created for software practice. URLs are deliberately empty; the locators identify exercise documents, not real archives.

[history_library.json](../examples/history_library.json) has no inline sources. It is intended for explicit library retrieval. Running that task without a source library fails because history requires a nonempty evidence packet.

From the project directory, build and inspect a packet without loading a model:

```bash
creativity build-sources examples/source_library.jsonl \
  --output data/source-libraries/merehaven-practice.sqlite

creativity retrieve-sources data/source-libraries/merehaven-practice.sqlite \
  --query "Merehaven quay levy protest March 1812" \
  --counter-query "payroll certified work unpaid wages collection deadline" \
  --counter-query "commissioner memorandum copied reprint arrests installments" \
  --max-chars 8000 --max-passages 6 --per-document 2 --min-documents 3 \
  --output evaluation/merehaven-packet-001.json
```

Use a new library or output filename for another experiment. The builder refuses to overwrite an existing library. Retrieval writes selected Source records and an audit manifest; no model or web service supplies missing documents.

Inspect `sources` and `manifest` in the packet before generation. In particular, check `selected_documents`, `omitted_documents`, `omitted_passages`, `document_coverage_shortfall`, and `uncovered_query_indices`. A query can miss relevant evidence, and a budget can exclude a relevant passage. An extra counter-query records another explicit search; it does not certify that every competing account was found.

Then run the local model with the same selection settings:

```bash
creativity generate examples/history_library.json \
  --config configs/local.yaml \
  --source-library data/source-libraries/merehaven-practice.sqlite \
  --source-query "Merehaven quay levy protest March 1812" \
  --counter-query "payroll certified work unpaid wages collection deadline" \
  --counter-query "commissioner memorandum copied reprint arrests installments" \
  --source-max-chars 8000 --source-max-passages 6 \
  --source-per-document 2 --source-min-documents 3 \
  --output runs/merehaven-library-001
```

The generation command requires the model to be available locally; add `--allow-download` when you explicitly want the initial checkpoint download. It keeps existing inline `Task.sources`, adds retrieved passages, rejects conflicting source IDs, and saves `source_selection.json` in the run. A short `Task.source_scope` statement tells the writer and critic the selection's limits. Existing sources are never silently removed to make room.

The character budget covers the compact JSON source list, including provenance fields. It is **not a token budget** and does not include every surrounding prompt or schema. The backend checks the final templated prompt with the actual tokenizer and rejects an overlong request. Reduce the packet deliberately, make the task narrower, or revise measured context settings when that check fails; there is no silent text truncation.

## Verify a quotation without promoting it to fact

The packet uses stable passage IDs beginning with `P`. Find the payroll passage's ID in the packet, or extract it with:

```bash
SOURCE_PASSAGE_ID=$(python -c 'import json; p=json.load(open("evaluation/merehaven-packet-001.json")); print(next(s["id"] for s in p["sources"] if s["document_id"] == "practice_merehaven_payroll_1812"))')

creativity verify-source-quote data/source-libraries/merehaven-practice.sqlite \
  --source-id "$SOURCE_PASSAGE_ID" \
  --quote "February wages remained unpaid on 14 March."
```

If the payroll passage was omitted, inspect the manifest and adjust the explicit query or budget before using this example. Do not replace the missing passage with an invented citation.

A successful verification checks exact wording, the stored original document, passage identity, hashes, and character offsets. It returns `establishes_truth_or_entailment: false`. It has not determined whether the clerk's entry is accurate, whether all workers are covered, whether households had other resources, or whether delayed wages caused the protest. The quotation API preserves spelling and whitespace exactly; the pipeline's other mechanical quote checks may normalize whitespace, as described in [checks.py](../src/creativity_pipeline/checks.py).

In the practice dossier, a defensible reading should preserve these distinctions:

- The memorandum's insulting generalization is evidence of the commissioner's expressed view. Quoting it correctly does not make it a true description of workers.
- The payroll entries record certified work and unpaid wages for named workers. They challenge the memorandum's explanation, while leaving household resources and wider conditions incompletely observed.
- The printed notice copies the memorandum. Two document IDs do not turn that wording into two independent accounts.
- The petition records the signatories' complaints and requests. Its advocacy purpose matters; the packet does not establish that it represents every household.
- No arrests entered in one office's book does not establish that no protest, coercion, or violence occurred anywhere.
- The comparison roll records installments elsewhere. Without comparable information about other conditions, it supports a question about payment arrangements rather than proving their causal effect.

These are reading goals for the invented exercise, not assertions about real historical events. Inspect actual model output and critic findings against them.

## Prepare a real source library

Supply one JSON object per line. Each record must contain a unique `id`, a `title`, and the complete `text` of that input document. `document_id` defaults to `id` and must also be unique. Keep distinct pages or excerpts of one original document together when constructing a complete document record; do not manufacture apparent source diversity by assigning each excerpt a different original-document identity.

| Field | What to record |
|---|---|
| `author`, `date` | The documented author and date, or `unspecified`; distinguish uncertain or inferred attribution explicitly. |
| `perspective` | A supported account of standpoint and limitations, or `not assessed`. |
| `audience`, `purpose`, `context` | Relevant documented circumstances. Do not ask the model to invent a biography or motive to fill a blank. |
| `url`, `locator` | The actual source URL and edition/page/item locator when available. Leave unknown values empty. |
| `kind` | A descriptive document-type label. This is not a credibility score or an automatic ranking preference. |
| `independence_group` | A known shared account, reproduction, or transmission family; otherwise `null`. Different author names do not establish independence. |

The builder stores the original text and makes deterministic overlapping passages with exact character locators. Retrieval uses SQLite FTS5/BM25 when available and records a lexical fallback otherwise. It prioritizes query coverage and distinct documents within explicit limits. Repeated identical passages within one document are not selected twice. Identical wording in distinct documents is retained with its separate provenance: text equality by itself establishes neither copying nor independence. Matching whole-document text is listed for inspection, and declared shared account families remain visible. Even records known to transmit one account can matter for comparison, while still providing no independent corroboration of that account. Retrieval does not discover every paraphrase, common informant, shared institutional assumption, or hidden dependence.

There is **no universal two-source rule**. An attributed statement about what one document says can be supported by that document. A broad factual or causal interpretation may need comparison, independent corroboration, contextual knowledge, or a narrower conclusion. Adding several dependent citations does not repair an unsupported inference. Judge what the specific claim requires, what the packet actually supplies, and what remains missing.

Source selection remains the researcher's responsibility. For final evaluation, use real, attributed source packets and unseen questions; this bundled synthetic exercise is development material.

## What the dedicated critic contributes

The semantic critic receives raw prose and immutable task context, including all supplied sources and the retrieval-scope note. It does not receive the writer's plan, claim annotations, or earlier praise as authority. It must identify concrete passages and assess support, source assertions, counterevidence, causal steps, and uncertainty. For English it examines the narrative brief, continuity, character knowledge, motivation, structure, and voice while preserving effective unusual choices.

The default configuration uses a separate critic call on the same checkpoint with the writer adapter disabled. This changes the role and available context; it does not create an independent human expert or eliminate shared model errors. An optional separate critic checkpoint can be routed sequentially so that writer and critic weights are not intentionally kept on the GPU together. That is an operational option for the 20 GB workstation target, not a measured guarantee that any chosen checkpoint and context will fit. See [INFERENCE.md](INFERENCE.md) for the runtime measurement boundary.

Criticism should produce traceable findings and actionable revision priorities, along with choices worth preserving. A finding must not turn the critic's own unsupported assertion into replacement evidence. The runner validates the critic's report, keeps it tied to the draft hash, and obtains a fresh report after a repair. A materially invalid report is retained for inspection instead of being treated as reliable editing instructions. If the critic needs absent evidence, history can stop with `needs_sources`; the writer must not fabricate the missing material.

Inspect `critic_00.json` and subsequent revision reports, the corresponding events in `events.jsonl`, and `critic_reviewed_output_sha256` in `summary.json`. A report separates its decision and coverage statement from domain rubric judgments, source assessments, evidence findings, concrete issues, revision priorities, choices to preserve, and missing evidence. Read those findings; an `accept` label alone is not an explanation. History criteria cover attribution fidelity, source context, corroboration and counterevidence, reasoning and uncertainty, and engagement with the sources. English criteria cover task fulfillment, specificity and originality, voice and language, motivation and consequences, and structure and pacing.

### Critique an existing draft

The standalone command reviews an existing draft without generating replacement prose. Supply the matching task and source packet; for the library example, the saved `task.json` contains the exact merged sources:

```bash
creativity critique runs/merehaven-library-001/task.json \
  --draft runs/merehaven-library-001/final.txt \
  --config configs/local.yaml \
  --output runs/merehaven-critique-001
```

For your own work, replace those two input paths with your task JSON and draft text file. The command creates `assessment.json`, `critic.json`, and `input_draft.json`; it does not create a generation run's `final.txt` or `summary.json`. Preserve the input text when assessing the findings, and compare any later edit with the exact version reviewed.

`assessment.json` distinguishes report validity from the critic's verdict. `valid_report: true` means the critic's report passed its contract checks; the report can still say `revise`, `reject`, or `abstain`. The CLI exits with code 0 for a valid report and 2 for an invalid report. A successful command is therefore not an endorsement of the submitted draft.

Review remains fallible. `accepted_by_checks` records acceptance by these checks and model judgments, not proof of historical truth or literary quality. Use the frozen-output and blinded human review workflow in [README.md](../README.md#assess-actual-writing-quality) to evaluate the prose, citations, handling of counterevidence, and residual errors.

## Evaluate the actual critic before trusting it

The eight cases in [critic_cases.jsonl](../examples/critic_cases.jsonl) pair deliberately faulty short drafts with valid controls. They are invented development diagnostics. Case descriptions and expected outcomes do not enter the model prompt; it receives only the ordinary task and supplied prose/source packet.

```bash
creativity evaluate-critic --cases examples/critic_cases.jsonl \
  --config configs/local.yaml --output evaluation/critic-base-001
```

Inspect `report.json`, `schedule.json`, and the individual `assessments/` directories. Each case passes only when the critic returns a valid report, its decision is among the declared expected decisions, and it identifies at least one required issue code when the case specifies them. Invalid or failed reports remain in the designated denominator. False positives and false negatives are counted only for the explicit decision comparisons described in the report; failures and abstentions stay separately visible.

The evaluator assesses fixed drafts; it does not generate improved examples, train a critic, or adapt the expected answers to its outputs. Every case and its initial pending status are saved before any inference. Interruptions retain both the attempted case and the remaining pending cases. Budgets apply separately to each assessment.

The demo backend is rejected by this command. Unit-test scripts exercise the accounting and scoring rules, but are not evidence that Qwen or another real critic recognizes the source problems. Once this visible set is useful for development, test on new real source packets and independent judgments. A pass on eight constructed examples cannot establish general historical competence or demographic fairness.

## Method references

The historical questions behind the prompt follow established practices of sourcing, contextualization, corroboration, and close reading. These references motivate the assessment dimensions; they do not validate an LLM implementation:

- [Digital Inquiry Group: Historical Thinking Chart](https://www.inquirygroup.org/history-lessons/historical-thinking-chart)
- [National Archives: Understanding Perspective in Primary Sources](https://www.archives.gov/files/education/lessons/document-analysis/english/understanding-perspective-in-primary-sources.pdf)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685), for documented limitations of model-based judging.
