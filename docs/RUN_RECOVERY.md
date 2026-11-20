# Diagnose and recover a stopped run

Version 0.3.1 addresses a confirmed ordering problem in 0.3.0: a draft could have material mechanical errors, then fail during critic generation before the writer ever received the known repair instructions. A real run reported mismatched claim spans, unknown source IDs S6/S7, zero repairs, no completed critic report, and about 63 minutes elapsed. Its pasted tail did not include the stopping error or task mode. That evidence identifies invalid annotations; it does not establish why the critic stopped or how fast the GPU decoded.

## Read the existing records first

From the project directory:

```bash
python scripts/inspect_run.py --latest runs
```

Check `selected_run`; automatic selection uses the most recently modified summary or assessment. Use an explicit directory if a different run was selected:

```bash
python scripts/inspect_run.py runs/the-failed-run
```

This is a standalone standard-library script. It does not import Torch, load a tokenizer, download weights, generate text, or modify the run. It reports the last 12 calls by default and marks omitted or unreadable records. Prompts, source passages, draft prose, model responses, and Pydantic input-value payloads are excluded. Error details and output sizes are bounded.

The important fields are `status`, `error`, `task_mode`, supplied source IDs, and the request/response sequence. A `finish_reason` of `length` establishes that a particular response did not finish within its limit. A context-admission error means the requested prompt plus reserved output would exceed the configured context. Schema errors identify an output-contract problem. These require different remedies; a larger token limit is not a universal fix.

The CUDA information in `summary.backend` describes the saved run. To inspect the currently allocated hardware separately:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version,utilization.gpu,memory.used --format=csv
```

With the sequential router, `response.elapsed_seconds` is routed call wall time. `routing.generation_seconds` covers the child's input setup and model generation, including prompt prefill. It excludes loading weights but is not decode-only timing. `preparation_seconds` includes loading and other processing; `unload_seconds` records outgoing-model release. Overall run time also includes token counting and work outside the recorded generation calls. Do not derive decode throughput by dividing all output tokens by overall run time.

## What the reported validation codes mean

| Code | What it establishes | Correct response |
|---|---|---|
| `claim_text_missing` | The annotated claim is not a nonempty exact span of the current draft | Copy the actual draft span into the annotation; reconsider the prose if it is unsupported |
| `quote_source_not_cited` | A support quote names a source absent from that claim's `source_ids` | Verify the source actually supports the claim, then align the local citation and annotation |
| `unknown_quote_source` | The support quote names a source absent from the supplied task | Use a real supplied passage or narrow/remove the unsupported claim; never manufacture a source entry |

A draft span and a support quotation have different origins: the former comes from the draft; the latter comes from a source passage. Claim IDs identify annotations and cannot create new evidence IDs. Matching text alone does not prove entailment, credibility, corroboration, or historical truth.

In source-free English fiction without date constraints, prompts explicitly request `claims: []` and use evidence-linked narrative state updates. English literary analysis can still annotate actual supplied works. Historical prompts identify the permitted source IDs and require local citations and exact draft spans.

## Review and repair order

Each saved draft first receives deterministic checks. Material defects go directly to the writer's repair stage. The same `max_repairs` limit covers these repairs and later editorial revisions. If defects remain at the limit, the run stops with `needs_review`, retains the draft, and explains why semantic reviews were skipped.

Once mechanical checks pass, the configured critic and general reviewer assess the current draft. Every changed draft invalidates earlier approvals. Invalid critic reports remain visible and are not converted into trusted rewrite instructions. Historical abstention can still stop with `needs_sources`.

This changes orchestration and prompt clarity. It does not add constrained JSON decoding, establish real-model writing quality, or guarantee that a model will repair every defect.

## Reuse the saved draft

After inspecting the stopping error, run:

```bash
python -m creativity_pipeline repair-run runs/the-failed-run \
  --output runs/repaired-001
```

The configuration defaults to the parent's saved `config.json`, preserving its model, precision, context, and budgets. To use explicitly adjusted settings:

```bash
python -m creativity_pipeline repair-run runs/the-failed-run \
  --config configs/local.yaml \
  --output runs/repaired-002
```

`--allow-download`, `--seed`, `--adapter`, `--critic-model`, and `--critic-revision` remain available. There is no replacement task, state-import, or source-retrieval flag: this operation repairs the existing task using the existing packet. If the evidence itself must change, prepare a new task and run a new experiment.

The command:

1. Reads the saved task, configuration, `final.json`, `final.txt`, and summary. It requires matching task identity and a prose hash matching both final files. A saved selected plan is checked against the saved task.
2. Reuses the complete annotated draft, sources, initial narrative state, selected plan when present, and source-selection manifest. It does not treat the parent's final state as the initial state for the same story.
3. Creates a new exclusive output directory, records the parent run and input hashes, and skips planning and initial drafting.
4. Runs mechanical checks and all configured reviews afresh. Previously accepted reports cannot approve a continuation.
5. Counts only this new attempt's calls, tokens, repairs, and elapsed time. Parent counters remain in `continuation.json` and the summary; add earlier attempts when assessing total cost.

Input hashes help detect mismatched artifacts. They are not signatures, proof that a source is authentic, or protection against coordinated edits of all saved metadata. Source/task/draft/provenance snapshots are also checked for accidental in-memory mutation before a continuation starts.

Original files are never rewritten, and an existing output directory is rejected. Missing or inconsistent final files stop recovery before inference. A changed source packet belongs in a new task, not a forged old run. Fixture origin remains a fixture after real-model review. Exported continuations have the separate `repair_run` baseline and retain parent-condition provenance, so they are not silently pooled with fresh planning runs.

## Updating an existing server checkout

Keep the working virtual environment, existing `configs`, tasks, sources, adapters, and `runs`. The 0.3.1 code changes do not require another Torch/CUDA installation. Back up locally edited code before copying the archive's updated `src`, `scripts`, and `pyproject.toml` into the checkout. Update documentation and tests as desired.

With the existing environment activated, refresh only the editable project:

```bash
python -m pip install --no-deps --no-build-isolation -e .
python -c 'import creativity_pipeline; print(creativity_pipeline.__version__)'
python scripts/inspect_run.py --latest runs
```

The version should be `0.3.1`. The editable install uses the environment's existing setuptools/wheel and dependencies; it does not intentionally upgrade the working GPU stack. Diagnose the earlier critic failure before choosing a new expensive attempt.
