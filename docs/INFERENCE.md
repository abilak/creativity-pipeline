# Local inference backend

`TransformersBackend` implements the dependency-free `Backend` protocol in
`src/creativity_pipeline/interfaces.py`. It loads a real causal language model
locally; it is separate from the fixture/demo backend. The supported API target
is Transformers **4.57.1**. PyTorch, Transformers, bitsandbytes, and PEFT are lazy
imports, so importing the project or running fixture tests does not load them.

## Load an already available model

```python
from creativity_pipeline.backends import TransformersBackend
from creativity_pipeline.interfaces import GenerationRequest

backend = TransformersBackend(
    model_id="Qwen/Qwen3-8B",  # A local checkpoint directory is also accepted.
    revision="main",         # Prefer a pinned commit for reproducible experiments.
    device="cuda",
    load_in_4bit=True,
    adapter_path=None,
    max_context_tokens=8192,
    local_files_only=True,
)

request = GenerationRequest(
    role="writer",
    messages=[{"role": "user", "content": "Write a brief scene about an unopened letter."}],
    max_new_tokens=512,
    temperature=0.7,
    seed=42,
    thinking=False,
    metadata={"use_adapter": False},
)

prompt_tokens = backend.count_input_tokens(request)
result = backend.generate(request)
print(result.text)
print(result.finish_reason, result.input_tokens, result.output_tokens)
print(backend.describe())
```

The default is **local/cache-only** loading. Missing files produce a clear error;
the backend does not silently download weights or fall back to a different model.
The caller can explicitly set `local_files_only=False` to permit Hugging Face
loading/downloading. The project CLI exposes this as `--allow-download`. No cloud
inference service is used by this backend. Authentication, if needed for a gated
checkpoint during an explicitly permitted download, remains with the Hugging
Face environment; the backend does not record credentials.

CUDA loading uses NF4 4-bit weights, double quantization, bfloat16 compute, and
`device_map={"": 0}` by default. `cuda:N` selects a different single GPU. There is
no automatic CPU offload. SDPA is requested; a model that specifically reports
that it does not implement SDPA is loaded with eager attention and this is
recorded. Other loading errors are not treated as reasons to switch models or
attention implementations. Checkpoint loading requires safetensors and does not
enable remote model code.

For a small, already available CPU checkpoint, set `device="cpu"` and
`load_in_4bit=False`. CPU uses float32. Its tokenizer must have a saved chat
template. This option supports a real small-model smoke test; it is not a claim
that Qwen3-8B is practical to run on CPU. No model was downloaded or executed as
part of the unit tests described below.

## Dedicated critic routing

The CLI wraps the writer in `SequentialCriticBackend`. With no
`critic_model_id`, critic calls reuse the writer's backbone and force its
writing adapter off. With a separate checkpoint, the router counts the prompt
using that checkpoint's tokenizer, releases the previously resident model,
and makes exactly one generation call to the selected backend. It does not
load both weight sets together or add a fallback model call after a failure.

`semantic_critic` and the optional legacy `review_evidence` role use this route.
Planning, selection, writing, and the general draft review use the writer
backend. The default structured critic replaces the legacy history audit;
both are not run together. The critic uses the configured review temperature
without thinking mode, and a separate `critic_tokens` output allowance.

The writer's `unload()` synchronizes CUDA when applicable, releases model
ownership, performs garbage collection, and clears unused CUDA allocator
cache. It retains tokenizer/config state and established model/template/adapter
identities for accounting. It does not offload weights into host RAM. The
router requires exclusive ownership of its two backends; callers must not
load them independently or retain model tensors outside the backend.

Configure a separate checkpoint with `critic_model_id`, `critic_revision`,
`critic_load_in_4bit`, and optional `critic_max_context_tokens`. It uses the same
device and explicit download policy as the writer, and has no writing adapter.
Each chosen checkpoint must fit on its own. One resident model is a routing
invariant, not a measured GPU fit guarantee; reloads can be expensive.

Descriptions include separate writer/critic identities, routing roles, the
adapter-disabled policy, current residency, and routing timing. Evaluation
fingerprints use stable checkpoint/template and inference identities, excluding
residency, loading state, timings, and file locations. Missing resolved critic
identity remains visibly unresolved.

Each routed response retains the child's exact token counts. Its total elapsed
time includes switching. `generation_seconds` is child-reported generation
time; `preparation_seconds` is the remaining child wall time, including loading
and prompt/output processing. It is not an isolated weight-loading benchmark.
Failed releases stop before the next model is loaded, and failed generations
are propagated without fallback.

## Context and token budgets

The tokenizer's own chat template is used with `add_generation_prompt=True` and
the request's `enable_thinking` flag. Tokenization then sets
`add_special_tokens=False` and `truncation=False`, preventing doubled template
tokens or silently discarded context.

The backend requires:

```text
actual templated prompt tokens + max_new_tokens <= effective context limit
```

The effective limit is the minimum of the configured cap, a finite tokenizer
limit, and the loaded model's maximum position count. Overflow raises
`ContextLimitError`; the caller must change its prompt or budget deliberately.
Counting resolves the model configuration and loads the tokenizer, but not the model weights. The preflight is repeated
after the model is loaded so that a stricter model limit cannot be missed.

`input_tokens` counts the entire rendered chat prompt, including role and thinking
prefix tokens. `output_tokens` counts every generated token, including reasoning
tokens, reasoning delimiters, and the ending EOS token. It is never estimated
from visible words or recomputed after stripping reasoning. A completion budget
therefore covers **both reasoning and the final answer**.

The request controls temperature and seed. Temperature zero selects greedy
decoding; positive temperatures select sampling. Optional request fields
`top_p`, `top_k`, and `min_p` override the checkpoint's saved settings without
mutating them; leaving a field `None` retains that checkpoint default. Effective
values and repetition penalty are recorded in result metadata. The pipeline
passes explicit top-p 0.8 for non-thinking or 0.95 for thinking, top-k 20, and
min-p 0 by default. All are configurable. Qwen recommends sampling rather than
greedy decoding for thinking mode; a zero-temperature backend request is still
honored explicitly.

## Thinking output and incomplete answers

Qwen-style `<think>...</think>` output is separated at the token level. The visible
`result.text` contains the answer after the completed reasoning block. Metadata
contains `raw_output`, `reasoning_tokens`, and `thinking_complete`, so accounting
and parsing can be audited. `raw_output` is a model-generated experimental trace,
not a verified explanation of the model's internal computation. It is not printed
by the backend automatically.

If the output budget is exhausted during thinking, the result has empty visible
text, `finish_reason="length"`, and `incomplete_reason="unterminated_thinking"`.
An EOS before the reasoning block closes is marked `"incomplete"`. A truncated
final answer is retained for inspection but marked `"length"`, never a successful
completion. An EOS emitted on the last permitted token is a normal `"stop"`.
Empty final answers and unexpected early endings without EOS are incomplete.

Callers must check `finish_reason` or `metadata["complete"]` before treating a
response as a finished JSON object, accepted plan, review, or final draft. An
incomplete call has already consumed tokens and must stay in the experiment's
ledger. A retry is a new charged call, not a replacement for the failed entry.

## Adapter isolation

Adapters are loaded from a completed output directory made by this project's trainer, with `is_trainable=False`.
The directory must contain `adapter_config.json`, `training_config.json`, and
`training_report.json`. Arbitrary external PEFT adapters without that provenance
are not currently supported. The three records must agree on the base model and
immutable commit; the report must say the training completed.

The default `revision="main"` adopts the adapter's saved commit. An explicit SHA
or tag must match that commit. The backend resolves `AutoConfig` before counting,
loads tokenizer and weights at the same revision, and verifies the chat-template
hash against the training report. A mismatch fails before drafting; changing the
configured revision is not a way to silently combine incompatible checkpoints.

The supported adapter weight layout is the trainer's single
`adapter_model.safetensors` file. Loading checks PEFT's actual missing/unexpected
tensor-key result and rejects partial adapters. A streamed hash of those weight
bytes and canonical adapter configuration becomes `adapter_sha256` only after
successful loading. Identical files copied to another directory retain the same
identity. Binary-only and sharded adapter layouts are currently unsupported.

They are **not merged** into the base weights. Set `metadata["use_adapter"]` on
each request to make the choice explicit:

```python
planner_request = GenerationRequest(
    role="planner", messages=[{"role": "user", "content": "Propose three different plans."}],
    max_new_tokens=512, metadata={"use_adapter": True},
)
review_request = GenerationRequest(
    role="review", messages=[{"role": "user", "content": "Review this draft against the evidence."}],
    max_new_tokens=512, metadata={"use_adapter": False},
)
```

With an adapter loaded and no explicit override, the roles `plan`, `planner`,
`write`, `writer`, `draft`, `drafter`, `realize`, `realizer`, `revise`, `reviser`,
`revision`, and `repair` use it. Every other role defaults to unmodified base
weights. A request without a loaded adapter uses the base model and records
`use_adapter=False` even if an adapter was requested.

Base-model calls use PEFT's `disable_adapter()` context manager, restoring the
previous state even when generation raises an exception. The same shared base
model remains a correlated judge; disabling an adapter is not an independent
model-family evaluation.

## Provenance and measurements

`describe()` does not load weights. Before loading, unknown fields are `None`.
After loading, it records the requested model/revision, resolved cached commit
when available, chat-template SHA-256, package versions, quantization and attention
implementation, adapter path, adapter base revision/template verification, effective
context, and CUDA device details.

CUDA peak counters are reset immediately before each generation, after loading.
The result records baseline allocation, peak allocated and reserved bytes, and
allocation after generation. These are **per-call process CUDA allocator
measurements**, not whole-system GPU usage or model-loading peaks. Elapsed time
includes device input preparation and synchronized model generation, and excludes
checkpoint loading. CPU has no fabricated CUDA measurement.

Backend calls serialize generation because Torch seeds, adapter state, and CUDA
peak counters are shared state. `torch.manual_seed(request.seed)` is set for each
call. A seed does not promise bitwise identical results across package versions,
hardware, kernels, or different batching arrangements. Pin revisions and record
the execution environment for comparisons.

## Validation scope

Run the targeted tests with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_backends.py -v
```

The tests use tiny dependency fakes. They validate exact accounting, overflow
rejection, unfinished-thinking handling, adapter restoration, lazy loading,
download opt-in, CUDA configuration, and per-call instrumentation. They do not
measure quality, speed, memory fit, CUDA compatibility, or successful execution of
Qwen3-8B. A separate real-model run on the target machine is required for those
claims.

`tests/test_real_peft.py` adds five actual-library regressions using tiny randomly
initialized CPU Qwen models, without Hub downloads. They test adapter/logit round
trips, incomplete or unexpected tensors, nonfinite parameters, and Trainer loss
normalization. They are skipped when their optional dependencies are absent;
that skip is not a pass. Run them after installing the optional stack:

```bash
python -m unittest discover -s tests -p test_real_peft.py -v
```

## Primary API references

- [Qwen3-8B model card: thinking switch and parsing](https://huggingface.co/Qwen/Qwen3-8B)
- [Transformers 4.57.1: bitsandbytes quantization](https://huggingface.co/docs/transformers/v4.57.1/en/quantization/bitsandbytes)
- [PEFT PeftModel: disable_adapter context manager](https://huggingface.co/docs/peft/en/package_reference/peft_model)
- [Transformers 4.57.1: chat templates](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating)
