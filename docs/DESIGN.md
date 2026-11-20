# Design decisions and next experiments

## What we are trying to improve

The goal is sustained, specific, worthwhile writing with more than one viable way to answer a prompt. English and history both need invention and reasoning, but their admissible inventions differ. A fiction writer may invent an event that respects the story's commitments. A historian may invent an explanatory arrangement or argument; the events, sources, and quotations still require evidence.

We therefore keep three separate questions visible:

1. Is this a valid and well-supported piece for the task?
2. Is its creative approach worthwhile, coherent, and meaningfully different from other acceptable approaches?
3. How do independent readers and external detectors respond to the fixed result?

There is no single scalar that establishes all three. A text can be unusual and bad, conventional and excellent, detector-friendly and inaccurate, or distinctive and still flagged. The code deliberately records those outcomes separately.

## Why start with one 8B model

The workstation constraint favors sequential calls to one quantized backbone, with a small domain adapter optionally active for planning and writing. Using multiple prompt roles does not create multiple independent minds. A base-model review can still share the writer's blind spots, even when the writing adapter is disabled.

Qwen3-8B is a concrete initial implementation target with documented thinking controls and available training tooling, not a claim to be the best model available. A different model can be substituted if it supports the backend's chat-template and adapter assumptions. Test schema adherence, prose quality, historical support, and measured cost on held-out tasks before choosing a replacement.

An 8B adapter can plausibly improve task formatting, recurring planning habits, and domain voice. Fine-tuning does not create missing evidence, verify a causal claim, or guarantee larger-model reasoning capability. This is why history starts from an attributed source packet and keeps evidence attached to claims.

## Why these stages

### Plans describe consequential choices

A plan includes intended effect, events or argumentative moves, dependencies, presentation order, and open choices. It is not only an outline of paragraph topics. For fiction, distinct approaches can differ in what the character attempts, what the reader initially misunderstands, what remains unsaid, or how an ordinary action changes the situation. For history, approaches can differ in causal mechanism, comparison, source disagreement, scale, or the boundary of the evidence.

The model sees previous approach descriptions so it can try a different one. Lexical overlap warnings are merely diagnostics. Neither unusual vocabulary nor a different label establishes conceptual diversity. Human review of a small candidate set remains the practical initial test.

The current implementation selects among complete short plans, then writes one piece. It does not maintain a beam of novel-length continuations. Four sequential plans are the default experiment; use one for initial compatibility testing and compare actual quality per token.

### Checks distinguish existence from meaning

Mechanical checks can answer whether an ID exists, a quoted passage occurs in a source, an annotation occurs in the prose, a declared dependency graph cycles, a state entry overwrites a protected ID, or declared years reverse an explicit relation.

They cannot answer whether a source is truthful, whether the passage entails the claim, whether an unannotated statement is false, whether a new state ID conceals a contradiction, or whether the narrative earns its ending. The reviewer receives those semantic questions, and human evaluation remains necessary. A matching quote is provenance evidence, not a proof of the interpretation.

Historical `attested` means the supplied source contains the cited statement. It must not be silently promoted to settled reality. `inference`, `disputed`, and `unknown` make epistemic status explicit. Inferring from a source still requires an appropriate citation; unusual unsupported causal stories are not rewarded for novelty.

Version 0.3 adds a structured critic with separate English and history rubrics. It sees raw prose and the supplied context without the writer's annotations or earlier praise. Its own quotations, source IDs, rubric coverage, and decision consistency are checked. Invalid criticism stops for review instead of automatically contaminating a repair. A valid negative report supplies targeted changes and choices to preserve. These contracts establish traceability, not an independent test of truth or creativity.

History inputs can now be assembled explicitly from a local SQLite library. Original documents, passage offsets, hashes, selected passages, and omissions remain auditable. Lexical retrieval and explicit counter-queries are bounded evidence-selection tools; unqueried or omitted sources were not assessed. Sources retain documented audience, purpose, context, and known shared origins. A document's usefulness is claim-specific: prejudiced rhetoric can establish a viewpoint, and a discriminatory rule can establish a prescription, without establishing the truth of the prejudice. See [CRITICS_AND_SOURCES.md](CRITICS_AND_SOURCES.md).

### Repairs have a stopping rule

Repeated unrestricted critique can erase voice and unusual but effective choices. The default is two repairs targeting concrete defects. The final revised piece is reviewed again. If it still fails, it is saved as `needs_review`; the system does not declare success by running out of patience.

If a repair completes but its review fails, the final text is retained while the old review is cleared. Review hashes bind judgments to the precise text reviewed. No state is committed from a failed or unresolved draft.

### State is explicit and bounded by evidence

English tasks have immutable commitments plus separate established-world and reader-knowledge ledgers. Only accepted final drafts add entries. Exact evidence spans make additions inspectable. The model reviewer must check that a proposed entry follows from its span and does not contradict prior facts.

This is deliberately modest memory. It does not infer every character belief, detect all semantic contradictions, or compress an entire novel into a trustworthy summary. When the ledger grows, a human-curated compaction or a separately evaluated retrieval layer will be needed.

## Training choices

The first useful adapter should learn good plans and compact, targeted repairs from curated examples. A set of accepted alternatives matters more than a single stylistically uniform preferred answer. Group/approach-balanced sampling prevents a prompt with many nearly identical rows from dominating the training schedule, provided the approach labels reflect real distinctions.

The exporter preserves the exact inference messages and excludes thinking-mode requests from the initial non-thinking SFT recipe. Prompt tokens are masked; the complete target and EOS are supervised. Overlength examples fail preflight rather than losing their conclusion or evidence. Splits are assigned at the related-task/source-family level.

Training is initially supervised QLoRA. There is no online reward optimization, automatic teacher approval, or preference stage yet. Before adding preferences, obtain reliable paired judgments showing what the SFT adapter still gets wrong. A preference dataset that rewards bland correctness over useful unusual choices can undo the diversity objective.

The 2,048-token cap is intentionally modest for the initial single-GPU experiment. Because runtime prompts contain schemas and context, some full repairs will not fit. This is a data-design constraint, not a reason to discard source material invisibly. Train compact complete plans first, measure a two-step GPU smoke run, then assess whether longer examples justify a 4,096-token experiment.

## Evaluation that can change our mind

### Establish the untrained baseline

Use fixed held-out English briefs and real historical source packets. Include tasks requiring restraint: a story that should stay quiet, incomplete evidence, conflicting accounts, a plausible but unsupported causal explanation, and a nonchronological narrative whose event sequence still matters.

Run direct drafting, one plan, and several plans with the same review rules. Record all calls, input and generated tokens, failures, and latency. Same maximum caps do not mean equal compute. Compare quality at multiple cost levels, or allocate saved calls to additional direct samples as a subsequent experiment. The current benchmark is a planning ablation, not a fully matched-compute tournament.

### Judge the right dimensions

| Dimension | English examples | History examples |
|---|---|---|
| Local quality | Specific observation, sentence control, natural voice | Clear claims, precise terms, readable argument |
| Structure | Consequential actions, pacing, purposeful revelation | Argument progression, chronology, causal mechanism |
| Validity | Continuity, motivation, consistency with commitments | Passage support, source criticism, uncertainty |
| Originality | Distinct situation, perspective, or development that earns its place | Worthwhile interpretation or synthesis supported by evidence |
| Diversity across outputs | Different viable narrative choices for the same brief | Different defensible explanatory approaches |

Human judges should not see the condition or model name when possible. Separate preference from factual-error annotation. Keep prompt families as the statistical unit; several generations of the same task do not supply several independent historical questions. This release records raw results but does not fabricate confidence intervals from an unrun study.

### Freeze before detector scans

Hash every designated output before scoring. Preserve failed and unscanned runs in denominators. Record detector name, version, date, exact label, and the meaning of each percentage. Include genuine-human controls of comparable genre and length. Keep `mixed` separate unless a policy was declared before collection.

This tests whether the generator produces text that a particular detector classifies differently. It does not establish human authorship, originality, or accuracy. Improvements are persuasive when independent human quality and historical support stay strong while diversity increases; a detector result is an additional observation.

The code never iteratively edits a text based on detector feedback and never uses detector labels to train the adapter. It exports a manual evaluation sheet so service access, versions, and observations stay visible.

## What would justify adding complexity

| Extension | Evidence required first |
|---|---|
| Semantic retrieval and learned source reranking | The current lexical retrieval misses relevant passages on measured source-recall tests |
| Longer training sequences | Useful complete examples fail 2,048 tokens and a GPU smoke run establishes feasible memory |
| Preference optimization | Curated paired judgments reveal stable shortcomings that SFT does not fix |
| Latent-plan compression | Explicit plans help but their token cost is demonstrably limiting |
| Branch search or SMC | Multiple continuations improve long-range structure enough to justify additional compute |
| Learned segmentation | Fixed scene/section boundaries measurably harm outcomes |
| Choosing a separate evaluator checkpoint | The optional sequential critic route improves measured error detection without unacceptable false positives or loading cost |

The original research ideas remain possible extensions. Their failure modes are easier to isolate after we have a small executable baseline and a held-out dataset.

## Known limits of this release

- The GPU inference and training integrations have not been executed in the development environment; the CPU tests use deterministic fakes and fixtures.
- The included history dossier is invented and cannot assess real historical expertise.
- No curated training corpus or adapter is bundled. More data is not automatically better; provenance and editorial quality matter.
- The self-reviewer is correlated with the generator and can accept unsupported prose.
- Claim annotations can omit facts; mechanical checks cannot prove full factual coverage.
- Quotation and declared-date checks do not establish historical truth or causal validity.
- Same-year or unspecified-year chronology remains unresolved unless a reviewer handles it.
- Word count is a target. The runner does not silently lengthen a weak result merely to meet a number.
- Runtime inference is one piece at a time. Local history passage retrieval is available; book-length orchestration, semantic retrieval, and narrative state compaction remain future work.
- Detector evaluation is manual, versioned, and descriptive; no scanner has been run as part of this implementation.

## Primary background and API references

- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)
- [PEFT quantization guide](https://huggingface.co/docs/peft/en/developer_guides/quantization)
- [Transformers 4.57.1 Trainer API](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer)
- [Re3: recursive reprompting and revision for long story generation](https://aclanthology.org/2022.emnlp-main.296/)
- [DOC: detailed outline control](https://aclanthology.org/2023.acl-long.190/)
- [Diversity preference optimization](https://arxiv.org/abs/2501.18101)
- [RAID: robust AI-text detection benchmark](https://aclanthology.org/2024.acl-long.674/)

These inform the experimental design; this implementation is not a reproduction of their full systems or a claim to match their published results.
