# Creativity Pipeline

A local research prototype for **creative English prose** and **source-grounded historical writing**. It proposes approaches, drafts, obtains structured criticism, and makes bounded repairs. The default uses one open-weight backbone sequentially; an optional second critic checkpoint also runs sequentially. Optional domain adapters use the same inference prompts as the curation workflow.

**Status:** working orchestration, deterministic checks, an offline demo, a real Transformers inference implementation, and a QLoRA training implementation. The GPU paths must still be exercised on your workstation. Demo text is a fixed fixture, not evidence of model quality. No trained adapters or model weights are included.

**Version 0.3.2:** makes every required critic field explicit, including `missing_evidence: []` when appropriate, and gives schema retries bounded field-level correction instructions. Missing fields are still rejected; no evidence declaration is silently filled in. Version 0.3.1's early mechanical repairs, domain-specific prompts, read-only diagnostics, and saved-draft `repair-run` remain available. See [docs/RUN_RECOVERY.md](docs/RUN_RECOVERY.md) for recovery commands and limits.

The structured English/history critic, optional sequential critic model, source library, and supplied-metadata references introduced in 0.3.0 remain available. A source's assertion is assessed separately from whether that assertion is true. See [docs/CRITICS_AND_SOURCES.md](docs/CRITICS_AND_SOURCES.md) for examples and limits. The earlier [quality review](docs/QUALITY_REVIEW.md) remains available; current executed checks are recorded in [docs/VALIDATION.md](docs/VALIDATION.md).

## Quick start: no GPU required

From this source directory on Ubuntu:

```bash
cd ~/Documents/creativity-pipeline
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

creativity doctor
python -m unittest discover -s tests -v
creativity demo --task english --output runs/demo-english
creativity demo --task history --output runs/demo-history
```

Each run directory is exclusive: choose a new name for another run. The offline demo exercises the full plan/check/select/draft/review path, but uses deterministic fixtures and estimated token counts. It does not download a model or call a detector.

If the project is not installed, use `PYTHONPATH=src python -m creativity_pipeline` in place of `creativity`. Bundled demo aliases require this source checkout; an installed wheel can use an explicit task JSON path instead.

## Run the local model

The initial recipe targets **Qwen3-8B**, a **20 GB RTX 4000 Ada**, and 48 GB host RAM. The hardware capacity is documented by [NVIDIA](https://www.nvidia.com/en-us/products/workstations/rtx-4000/). Qwen documents its thinking switch and chat-template usage in the [model card](https://huggingface.co/Qwen/Qwen3-8B).

Install the optional stack into the same virtual environment:

```bash
python -m pip install -e '.[gpu]'
creativity doctor --check-cuda
```

The package targets Transformers 4.57.1 and PEFT 0.17.x. Use a CUDA-enabled PyTorch installation compatible with your system; the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) supplies current platform-specific commands. `doctor --check-cuda` reports what this environment actually sees.

Start with one planned sample:

```bash
creativity generate examples/english.json \
  --config configs/local.yaml \
  --strategy single_plan \
  --allow-download \
  --output runs/english-base-001
```

`--allow-download` permits the initial tokenizer and checkpoint downloads. Subsequent runs can omit it when those files are cached. The default inference configuration is cache-only. No cloud inference API is involved.

Then try the full four-plan path:

```bash
creativity generate examples/history.json \
  --config configs/local.yaml \
  --output runs/history-base-001
```

The supplied history packet is **explicitly invented for a software demonstration**. Replace it with real, attributed source passages before assessing historical reasoning. The runner cannot supply evidence absent from the packet.

The initial GPU settings are 4-bit NF4 weights, double quantization, bf16 computation, SDPA, one model on one GPU, an 8,192-token context cap, and sequential calls. This is a starting configuration, not a measured memory-fit guarantee. Details and measurement boundaries are in [docs/INFERENCE.md](docs/INFERENCE.md). Long contexts, large source packets, schema overhead, and repairs all consume memory and tokens.

## What the pipeline does

| Stage | English | History |
|---|---|---|
| Input | Brief, constraints, established world facts, reader knowledge | Question, attributed sources, explicit chronology constraints |
| Plan | Several approaches, event relationships, presentation order, open choices | Several evidence-supported approaches, rival interpretations, uncertainty |
| Check | Protected IDs, dependencies, presentation references | Source IDs, exact quoted passages, declared dates, dependencies |
| Select and draft | Develop one approach with room for local invention | Write with claim annotations, citations, and bounded conclusions |
| Draft checks | Repair material state or annotation defects before reviews | Repair material citation or annotation defects before reviews |
| Dedicated critic | Development, consequential choices, voice, motivation, structure; preserve successful choices | Attribution, sourcing/context, corroboration, causal reasoning, uncertainty, engagement with sources |
| General review | Continuity, commitments, reader knowledge, evidence for state additions | Complete draft and annotation consistency, unresolved defects |
| Repair | Up to two full revisions by default | Up to two full revisions by default |
| Save | Final prose, review provenance, state, exact run log | Final prose, claim ledger, review provenance, exact run log |

The dedicated critic receives the raw prose and task/source context with the writer's annotations, plan, and earlier reviews hidden. Its report includes domain rubric judgments, exact quoted draft spans, source assessments and evidence findings for history, concrete issues, revision priorities, and choices to preserve. The runner checks the critic's quotations and report consistency before using its advice. A materially invalid critic report stops as `needs_review`; its unsupported criticism is not sent to the writer as a repair instruction.

Every repaired draft is checked again. Material mechanical findings go directly to bounded repair; if no repairs remain, the draft is retained as `needs_review` without spending further review calls. Mechanical and editorial repairs share the same `max_repairs` limit. Once the mechanical checks pass, the current text receives fresh criticism and general review. Both required judgments must accept it, with complete critic coverage and no material findings. Material issues cannot be canceled by praise or another model's acceptance. Reviews use the base model with the writing adapter disabled by default; this remains a correlated judge. With `semantic_critic: false`, history can use the earlier `history_evidence_audit` instead. The default does not add that legacy audit as a redundant third reviewer.

The selector sees shuffled, aliased candidates without previous reviewer praise. `selection_policy: sample` chooses uniformly among accepted plans with a recorded seed, providing an ablation for whether judged selection narrows variety. This is a testable alternative, not a promised quality improvement. Malformed candidate or candidate-review responses are logged and skipped after bounded retries; already accepted plans survive. Backend errors and global budget exhaustion still halt the run.

Each final status has a specific meaning:

- `accepted_by_checks`: mechanical checks and the fallible model reviewer accepted the final text. It is not human approval or a factual guarantee.
- `needs_review`: unresolved draft problems remain, the critic abstained on an English task, or the critic's own report failed validation.
- `needs_sources`: the history critic abstained because evidence or context is missing; the draft is retained for inspection.
- `budget_exhausted`: another call would exceed a configured token or call cap.
- `failed`: a backend, schema, context, selection, or other required step failed.
- `interrupted`: a user interrupt stopped generation; earlier drafts, findings, and the planned benchmark schedule remain available.

Unaccepted drafts remain available for inspection. Failed drafts do not commit new narrative state. A review is tied to the SHA-256 of the text it actually reviewed; a failed review of a newer repair cannot inherit the old draft's approval.

## Inspect a run

Read the existing logs without importing Torch or loading a model:

```bash
python scripts/inspect_run.py --latest runs
```

Check the printed `selected_run`. To choose a particular run, use `python scripts/inspect_run.py runs/english-base-001`. The report includes status/error, actual supplied source IDs, recorded hardware, call roles, completion reasons, and timing. It omits source text, draft prose, prompts, and model responses; inspect original logs locally when deeper analysis requires them.

| File | Contents |
|---|---|
| `final.txt` | Last complete structured draft's prose, when one exists |
| `final.json` | Prose plus claim annotations and proposed state changes |
| `summary.json` | Status, errors, final findings, review hash, token counts, timing, backend provenance |
| `events.jsonl` | Every request, response, schema failure, review, selection, and deterministic finding |
| `task.json`, `config.json` | Exact task and resolved settings supplied to the runner |
| `accepted_plans.json`, `selected_plan.json` | Eligible proposals and the selected plan |
| `draft_00.json`, etc. | Each draft that entered the review loop |
| `critic_00.json`, etc. | Structured critic report for each reviewed draft; its hash and validation are in the summary |
| `source_selection.json` | When retrieval is requested: selected and omitted passages, exact offsets, hashes, queries, and known limits |
| `references.json`, `references.txt` | History references using supplied metadata, unresolved source IDs, and uncited supplied sources |
| `state.json` | Accepted accumulated English state, or unchanged initial state on failure |
| `continuation.json`, `input_draft.json` | For `repair-run`: parent provenance and the saved draft used as input |

All generated tokens, including reasoning and malformed JSON attempts, count toward the measured budget. A backend exception can leave work unmeasurable; such a run records incomplete token accounting instead of inventing a count. Logs include source text and model outputs, so treat them as part of your research data.

### Continue from a saved draft

After diagnosing the stopping error, reuse an existing final draft in a new output directory:

```bash
creativity repair-run runs/english-base-001 \
  --output runs/english-repair-001
```

This inherits the saved configuration by default; `--config configs/local.yaml` selects an explicit replacement configuration. It reuses the saved task, source packet, retrieval manifest, selected plan when available, and complete `final.json`. It verifies the prose against the original output hash and `final.txt`, skips planning and initial drafting, and discards all previous approvals. A new run has its own bounded budget; parent costs are retained separately. Repeated continuations add compute and are not part of a single original budget. Continuations use a separate `repair_run` evaluation baseline. See [docs/RUN_RECOVERY.md](docs/RUN_RECOVERY.md).

## Write your own tasks

Use `examples/english.json` and `examples/history.json` as the external JSON contracts. Important fields:

- `task_id`: stable ID, letters/digits/underscore/hyphen.
- `mode`: `english` or `history`.
- `prompt`: the actual writing task, audience, and desired voice.
- `target_words`: an advisory target, with a diagnostic for a large deviation.
- `min_words`, `max_words`: optional hard bounds, counting whitespace-delimited words. Both must be consistent with the target. The quality challenge suite uses them so a trivial fragment cannot pass as a finished task.
- `commitments`: immutable facts or constraints with stable IDs.
- `initial_state.world`: established fictional events/facts.
- `initial_state.reader`: facts already revealed to the reader. Character beliefs should be labeled as beliefs, not silently treated as true world facts.
- `sources`: history passages with ID, title, original text, author, date, and perspective. Optional documented metadata includes `document_id`, `url`, `locator`, `kind`, `audience`, `purpose`, `context`, and `independence_group`.
- `source_scope`: known limits of the supplied packet. Retrieval adds a short disclosure of omitted material; it does not claim to have examined it.
- `date_constraints`: pairs of stable event IDs, such as `{"before":"E1","after":"E2"}`. History plan moves and final claims must reuse these IDs. The check compares declared event years; it does not extract dates from arbitrary prose or prove them against the sources.

For another scene in an English project, provide a new brief with the previous accepted ledger:

```bash
creativity generate my-next-scene.json \
  --config configs/local.yaml \
  --state-in runs/english-base-001/state.json \
  --output runs/english-scene-002
```

Imported state is merged with the new task's initial state, channel by channel. Distinct facts are retained; conflicting definitions of the same ID fail before generation. This is explicit scene continuation, not automatic book generation. There is no summarization or hidden pruning of growing narrative state. History source retrieval is an explicit optional step described below. Reduce a source packet deliberately or split a task if it exceeds context; the runner refuses silent truncation.

## Critic and multiple-source workflow

The critic is enabled by default in `configs/local.yaml`. It assesses the claim a source can support: a prejudiced assertion can establish what its author wrote without establishing the generalization itself. Author demographics alone do not determine reliability. Purpose, access to information, genre, contextual evidence, and dependence on other accounts matter. A discriminatory rule can be evidence of the rule without justifying its assumptions.

To assess an existing draft without rewriting it:

```bash
creativity critique examples/history.json --draft my-draft.txt \
  --config configs/local.yaml --output runs/critique-001
```

Inspect `assessment.json` and `critic.json`. `valid_report` means the report passed its structural and quotation checks; inspect the actual `decision`, findings, and missing evidence. Supplied drafts are recorded as input, not exported as model generations.

For many documents, build a local library from one attributed document per JSONL record:

```bash
creativity build-sources examples/source_library.jsonl \
  --output data/sources.sqlite

creativity generate examples/history_library.json \
  --config configs/local.yaml --strategy single_plan \
  --source-library data/sources.sqlite \
  --source-query "Merehaven quay levy protest March 1812" \
  --counter-query "payroll certified work unpaid wages collection deadline" \
  --counter-query "commissioner memorandum copied reprint arrests installments" \
  --output runs/history-library-001
```

The bundled dossier is invented development material. The library preserves original text and selects exact passages using lexical retrieval, document diversity, and explicit counter-queries. Omitted passages and query shortfalls remain visible. Different documents are not presumed independent, and there is no arbitrary two-source quota for every claim. Input is plain-text JSONL; PDF/OCR conversion and web acquisition must happen before ingestion.

The 8,192-token context cap covers the actual task, schema, draft, and reserved output. The retrieved packet's character budget is only a preliminary size control. A source-heavy task can still exceed the cap and will stop without truncating evidence. See [the full source/critic guide](docs/CRITICS_AND_SOURCES.md) for packet inspection, quotation verification, metadata, and cautious use of the practice dossier.

To use a separately chosen compatible critic, set `critic_model_id` and `critic_revision` in YAML, or pass `--critic-model` and `--critic-revision`. Models switch sequentially and the writer adapter is always disabled for critic calls. Each checkpoint must fit individually; switching adds loading time. Pin revisions and evaluate the critic on known flaws and correct control passages before relying on it. Changing the critic creates a distinct evaluation condition.

The bundled eight-case development set tests exactly that distinction:

```bash
creativity evaluate-critic --cases examples/critic_cases.jsonl \
  --config configs/local.yaml --output evaluation/critic-base-001
```

It includes bad inferences and valid controls: prejudice stated as fact versus accurately attributed rhetoric, discriminatory policy versus endorsement, copied accounts presented as independent, a controlled author-demographic change, and justified uncertainty. These are invented development cases, not a held-out historical benchmark. Expected outcomes are kept out of the critic's inputs. `report.json` preserves every designation, invalid or missing reports, false positives/negatives where defined, and actual token accounting. Read the individual evidence findings as well as the small-set pass count. Fixtures are refused; no real model result is bundled.

## Compare the pipeline before training

```bash
creativity benchmark \
  --tasks examples/english.json examples/history.json \
  --config configs/local.yaml \
  --strategies direct single_plan multi_plan \
  --samples 3 \
  --output runs/baseline-comparison-001
```

The strategies differ in planning: zero, one, or several plans. **All three retain the same draft review/repair loop.** `direct` therefore means direct drafting with that common review loop, not a raw one-call baseline.

`benchmark.json` records the complete task/sample/strategy schedule before generation. Pending, interrupted, and failed runs remain designated when the benchmark directory is exported. It also includes actual tokens, timing, and backend information. The runs have identical configured caps, not identical realized compute. Compare quality against measured cost; more calls are not free evidence of a better method. Loading time can affect the first run's total elapsed time. Per-call generation timing is also recorded.

Use blinded human assessment for voice, specificity, structural purpose, originality, and coherence. For history, assess factual support, chronology, causal restraint, source criticism, counterevidence, and uncertainty separately. Keep human writing controls from comparable tasks and lengths. Multiple samples from one prompt are related observations; do not pretend they are independent test questions.

For a CPU-only benchmark of the plumbing, replace the configuration with `configs/demo.yaml`. Those runs remain marked as fixtures and are refused by the evaluation and training exporters.

## Assess actual writing quality

`examples/quality_tasks.jsonl` contains three English briefs and three explicitly synthetic historical reasoning challenges, with hard word-count bounds. They exercise specificity, restrained structure, character/reader knowledge, causal confounding, conflicting witnesses, and missing records. They are development challenges, not a real-history benchmark or an untouched held-out test set. Reserve separate unseen briefs and real source packets for final comparisons.

After a single real-model run works, a small comparison is:

```bash
creativity benchmark --tasks examples/quality_tasks.jsonl \
  --config configs/local.yaml --samples 1 \
  --output runs/quality-base-001

creativity freeze-outputs runs/quality-base-001 \
  --output evaluation/quality-frozen-001.json

creativity export-review --manifest evaluation/quality-frozen-001.json \
  --output evaluation/human-review-001
```

Give reviewers only `evaluation/human-review-001/reviewer/`, containing `packet.html` and `ratings.csv`. Keep `private_key.json` private until judgments are complete. The packet includes the exact text and task context, but hides model, adapter, condition, seed, authorship labels, and internal model-review status. Text content itself can still reveal authorship.

Reviewers rate applicable dimensions from 0 to 4 and separately mark material errors. English uses overall quality, coherence, originality, and voice. History uses overall quality, coherence, evidence support, historical reasoning, and uncertainty. A material-error judgment requires a concrete note. Items with no text remain designated and unrated; missing ratings are not zero scores.

```bash
creativity import-review evaluation/human-review-001/reviewer/ratings.csv \
  --key evaluation/human-review-001/private_key.json \
  --output evaluation/human-results-001.json
```

The importer checks text, task-context, packet, and assignment integrity. It reports separate dimension distributions and error counts, preserving unreviewed outputs. These are descriptive single-output ratings; no significance test, paired effect, or composite creativity score is implied. Comparable human controls should include the same complete `task` context and sources in their explicit records.

## Curate training data

First collect base-model runs and review them. The exporter copies the exact runtime prompts and schema-valid candidate responses:

```bash
creativity export-training runs/english-base-001 \
  --group-id apartment-family-01 \
  --split train \
  --roles plan repair \
  --output data/candidates/apartment-01.jsonl
```

Exported rows have `accepted: false`. They can include rejected proposals: nothing is promoted to training data by a self-review score. Review/correct each row, assign its actual `approach_id`, and record its provenance and editorial review before setting `accepted` to `true`. Thinking-mode requests are skipped by this non-thinking recipe. Reviewer/selector outputs are not writer-adapter targets.

The JSONL schema is:

```json
{
  "id": "example-unique-id",
  "group_id": "related-task-and-source-family",
  "approach_id": "one-meaningfully-distinct-approach",
  "mode": "english",
  "split": "train",
  "messages": [{"role": "system", "content": "Exact runtime prompt"}, {"role": "user", "content": "Exact task context"}],
  "completion": "A complete Plan or Draft JSON object encoded as a string",
  "accepted": false,
  "provenance": "Record the source, permission, generation method, and human review here"
}
```

Keep all paraphrases, repairs, source packets, and related prompts in one `group_id` and one split. A different row ID does not make a new test case. The validator detects cross-split group overlap and normalized duplicate prompts; it cannot discover every semantic paraphrase automatically. Every accepted completion must also validate as a complete `Plan` or `Draft` object. Raw prose and reviewer responses cannot silently become this structured writer adapter's targets.

Use several **good, meaningfully different** responses per prompt rather than one canonical winner. The sampler balances prompt groups, then approaches, then rows. Near-duplicates should share an approach or be removed; inventing approach IDs does not create real diversity. Keep English and history data in separate files and train separate adapters.

The initial 2,048-token training cap is for the **entire templated prompt plus target**. Compact plans are the most practical initial targets. Full repairs can exceed it because they contain the old draft, instructions, and revised draft. Tokenization preflight reports offending IDs and fails; it never silently truncates the ending or source evidence. Increase sequence length only after a measured GPU run, or curate genuinely compact complete tasks. `overlength_policy: skip` is available, explicit, and logged.

## Validate and train an adapter

Provide `data/curated/english.jsonl` and/or `data/curated/history.jsonl` with accepted examples in both train and validation splits. No real corpus is bundled.

```bash
creativity validate-data data/curated/english.jsonl
creativity train --config configs/train_english.yaml \
  --output checkpoints/english-preflight-001 --dry-run
```

The default dry run is offline and checks configuration, data, and splits. It does not test token lengths or GPU fit. Set `preflight_tokenizer: true` in the YAML to enable tokenizer/config preflight; `local_files_only` controls whether that may download. Real training always tokenizes and checks lengths before loading weights.

Then perform a bounded real GPU smoke run on your curated data:

```bash
creativity train --config configs/train_smoke.yaml \
  --output checkpoints/english-smoke-001
```

This uses two optimizer steps and accumulation 1. It is a compatibility/memory check, not meaningful fine-tuning. A normal first training run uses:

```bash
creativity train --config configs/train_english.yaml \
  --output checkpoints/english-001
```

The normal recipe uses QLoRA, LoRA rank 16/alpha 32, NF4 double quantization, bf16, gradient checkpointing, batch 1, accumulation 16, learning rate `5e-5`, and one initial epoch. These are conservative experimental defaults, not tuned optimums. Loss applies to completion tokens plus EOS; prompt and padding tokens are masked. Test rows never enter tokenization or checkpoint selection. Validation loss selects a checkpoint but is not a creativity score. See the [PEFT quantization guide](https://huggingface.co/docs/peft/en/developer_guides/quantization) for the underlying method.

Use the adapter directly from its output directory:

```bash
creativity generate examples/english.json \
  --config configs/local.yaml \
  --adapter checkpoints/english-001 \
  --output runs/english-adapter-001
```

Training records the resolved base-model commit, tokenizer template hash, dataset provenance, sampled IDs, package versions, and GPU measurements. It stops on nonfinite loss and checks final validation loss and adapter tensors before recording completion. `observed_training` records the sequence lengths actually used in training forwards: a smoke run on short examples does not establish fit at the configured maximum length.

Project-produced adapters must load with the compatible pinned base revision and template. The loader checks missing/unexpected tensor keys and records a content hash of the adapter weights and configuration. Repeat the held-out baseline comparison before concluding that the adapter helped.

## Evaluate Pangram and other detectors

Detector scores are recorded as external outcomes after outputs are frozen. They are not generation rewards, training labels, or a rewriting feedback loop.

```bash
creativity export-detectors runs/baseline-comparison-001 \
  --manifest evaluation/frozen-001.json \
  --csv evaluation/detector-sheet-001.csv \
  --detector Pangram --version YOUR_ACTUAL_VERSION
```

No detector is queried automatically. Scan the exact frozen text using your own service access, then fill the sheet with `query_date` and canonical label `ai`, `mixed`, or `human`. Record the detector's actual model/version; do not replace this with a date if a version is available. Use a separate filled sheet for another detector or version. Keep failures and unscanned designated outputs in the manifest; remove only their unfilled rows from a particular submitted scan sheet. Missing scans remain visible in the report.

```bash
creativity import-detectors evaluation/detector-sheet-001.csv \
  --manifest evaluation/frozen-001.json \
  --output evaluation/detector-report-001.json
```

The importer verifies exact output hashes, dates, versions, duplicate observations, and field meanings. Reports separate generator conditions by model revision, template, adapter content, and generation settings; two systems using `multi_plan` no longer share a cohort automatically. Unknown provenance stays visibly unresolved. A completely unscanned manifest still reports its designated counts. Optional `classified_text_share` and `confidence` are separate fractions from 0 to 1. A fraction of text classified as AI is not a probability that the author used AI.

For known-human controls, supply `--records human-controls.json` when freezing. It must be a JSON list of records with `output_id`, `task_id`, `mode`, `baseline`, `origin: "human"`, and exact `text`. Authorship is an explicit provenance field; a detector's `human` label never changes a model output's origin.

A lower flag rate supports a claim about that detector on those samples. Assess originality, quality, and historical reliability independently before interpreting it as progress. The tool provides no undetectability guarantee. The [RAID study](https://aclanthology.org/2024.acl-long.674/) is useful background on detector evaluation across generators and settings.

## Design and boundaries

Read [docs/DESIGN.md](docs/DESIGN.md) for the reasoning behind this version, deferred research ideas, and the next experiments. Read [docs/INFERENCE.md](docs/INFERENCE.md) for the backend contract and its validation boundary.

This prototype does not include a trained latent planner, SMC search, preference optimization, a retrieval index, automated source acquisition, or an autonomous long-book workflow. Those extensions need evidence that the simpler measured baseline is insufficient. The immediate deliverable is a testable local writing-and-training loop, with explicit records of what it has and has not checked.
