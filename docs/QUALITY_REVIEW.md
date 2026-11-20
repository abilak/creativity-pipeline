# Quality review for version 0.2.0

## Assessment

The design is suitable for a controlled local research prototype: one model,
explicit evidence and continuity records, bounded search and repairs, and a
curation path into adapters. Its strongest remaining uncertainty is empirical:
whether the actual model produces better writing than simpler alternatives at
the same measured cost. Code checks cannot answer that question.

This review reproduced weaknesses in the initial implementation and fixed them.
The changes below should be evaluated as mechanisms with clear behavior, not as
already measured gains in creativity or historical accuracy.

## Concrete findings and changes

| Reproduced problem | Change | What still needs judgment |
|---|---|---|
| An unannotated false sentence reused a source marker from an unrelated valid claim and passed mechanical checks | Bind each citation occurrence to a local claim span that lists that source | A matching passage can still fail to support an interpretation |
| One malformed later candidate discarded an already accepted plan | Skip bounded candidate-local formatting failures and persist accepted plans incrementally | Backend/budget failures remain fatal and visible |
| A four-word response could pass a 700-word task with an approving reviewer | Optional hard min/max word bounds; substantial-development instructions; bounded quality examples | Length is not a measure of literary quality |
| The evidence reviewer could be anchored by the writer's annotations | Fresh history audit sees prose and sources with annotations/plan hidden; repeat after repair and bind its hash | The same backbone remains a correlated judge |
| Selection received original order and previous reviewer praise | Shuffle and alias candidates; omit praise; add seeded sampling among accepted candidates as an ablation | Actual preference/diversity effects require model samples and readers |
| Base and fine-tuned systems sharing a strategy name were pooled | Condition fingerprints include model/template/adapter/generation identity | An unresolved identity is flagged; metadata is not cryptographic attestation |
| With zero detector observations, all denominators vanished | Emit designated and missing counts even without any scans | No unobserved detector outcome is imputed |
| An interrupted benchmark did not retain its intended full schedule | Save every designation before generation; export pending/interrupted outputs | Unknown work after interruption remains unmeasurable |
| Imported story state silently replaced new task state | Merge by channel/ID and reject contradictions before generation | Different IDs can still hide semantic contradictions |
| Accepted training completions could be arbitrary prose | Require complete Plan/Draft JSON for accepted examples | Schema validity does not establish curation quality |
| Dependency adapter loading could warn about missing tensors yet succeed | Inspect actual missing/unexpected key results and reject incomplete adapters | The real-library regression must run in the installed target environment |
| Training could save nonfinite parameters with a completed label | Finite-loss guard, finite final validation, and adapter-tensor checks | Successful finite optimization does not establish useful writing improvements |

## Making creativity an actual experiment

The planner is asked to vary consequential choices rather than labels or
adjectives. English alternatives can vary action, perspective, withheld
information, or the consequences of an ordinary event. Historical alternatives
can vary source comparison, explanatory mechanism, scale, or the limits of an
inference. Novelty alone is not a reason to accept a claim or a plot.

The `judge` and `sample` selection policies make one specific concern testable:
does selecting the apparently strongest plan reduce the variety of worthwhile
outputs? Keep the candidate budget and other generation settings fixed when
comparing those policies. Inspect both the plans and the finished prose.

The human-review exporter removes condition labels and randomizes item order.
The same prompt and source packet remain visible, so readers can assess whether
the piece answers its task. It keeps quality dimensions separate from material
errors and delivery failures. Do not promote a detector label or one mean rating
into a universal creativity reward.

Use the included six-task suite for development checks. Because its briefs and
pitfalls are visible during implementation, it is not an untouched test set.
History examples are explicitly synthetic. A serious claim about historical
reasoning requires unseen real source packets and knowledgeable assessment.

## What to run next on the workstation

1. Install the optional stack and run the existing tests again. The five optional
   real-library tests use tiny CPU models and no downloads; they should run once
   Torch, Transformers, PEFT, Accelerate, and safetensors are available.
2. Generate one actual Qwen3-8B piece. Inspect context, output completeness,
   claims, reviews, latency, and GPU memory in the saved record.
3. Run the small quality suite under direct, one-plan, and multiple-plan settings.
   Those strategies keep a common review loop. Compare measured cost and blinded
   quality; configured caps alone do not equalize compute.
4. Curate multiple good approaches, grouped by related prompt/source family.
   Run tokenizer preflight, then the two-step GPU training smoke configuration.
   Check the lengths actually exercised before making a maximum-context fit claim.
5. Train separate domain adapters, compare against the base system on held-out
   tasks, and retain all designated results. Add detector scans only after the
   texts are frozen.

Keep any model-based self-review calibrated against the human findings. If it
misses important historical errors or repeatedly removes useful unusual choices,
revise the review protocol based on those concrete cases rather than adding more
unmeasured review passes.

## Evidence behind two implementation choices

Qwen's model card recommends different nucleus-sampling settings for thinking and
non-thinking modes. The pipeline now exposes those controls and starts from
top-p 0.95 and 0.8 respectively, with top-k 20 and min-p 0. Temperatures remain
role-specific experimental settings, not claimed tuned optima.

Research on LLM-as-judge evaluation identifies position, verbosity, and
self-enhancement biases. This supports testing shuffled candidate presentation
and blinded human assessment. It does not establish the magnitude of any bias
in this particular 8B pipeline.

- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [PEFT 0.17.1 model implementation](https://github.com/huggingface/peft/blob/v0.17.1/src/peft/peft_model.py)
- [Transformers 4.57.1 Trainer documentation](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer)
