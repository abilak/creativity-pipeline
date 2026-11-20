"""Single-GPU Qwen3 QLoRA SFT, with an offline data-only dry run.

Training dependencies are imported lazily. The supported Transformers interface
is >=4.57.1,<5, using Trainer + PEFT directly (no TRL). A normal run resolves the
Hub revision, tokenizes and validates every selected example, and only then loads
model weights. A dry run never downloads anything unless preflight_tokenizer is
explicitly enabled in its YAML config.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml

from .training_data import (
    TrainingExample, SequenceTooLongError, accepted_split, balanced_epoch_indices,
    load_training_examples, pad_completion_batch, prepare_completion_tokens,
)


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    dataset: str = Field(min_length=1)
    mode: Literal["english", "history"]
    model_id: str = "Qwen/Qwen3-8B"
    revision: str = "main"
    local_files_only: bool = False
    preflight_tokenizer: bool = False
    overlength_policy: Literal["error", "skip"] = "error"
    max_length: int = Field(default=2048, ge=64)
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    learning_rate: float = Field(default=5e-5, gt=0)
    num_train_epochs: float = Field(default=1.0, gt=0)
    max_steps: int | None = Field(default=None, ge=1)
    micro_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1)
    seed: int = 42
    logging_steps: int = Field(default=10, ge=1)
    save_total_limit: int = Field(default=2, ge=1)


def _trainer_controls(config: TrainConfig) -> dict[str, Any]:
    """Keep the custom sampler and make bounded smoke runs real optimizer steps."""
    controls: dict[str, Any] = {
        "max_steps": config.max_steps if config.max_steps is not None else -1,
        "accelerator_config": {
            "use_seedable_sampler": False,
            "split_batches": False,
            "dispatch_batches": False,
            "even_batches": False,
        },
    }
    if config.max_steps is not None:
        # Evaluate and save at the final optimizer step even when the data loader
        # has not completed an epoch. The held-out split still selects a checkpoint.
        controls.update(
            eval_strategy="steps", save_strategy="steps",
            eval_steps=config.max_steps, save_steps=config.max_steps,
        )
    else:
        controls.update(eval_strategy="epoch", save_strategy="epoch")
    return controls


class TokenizationPreflightError(ValueError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        overlong = ", ".join(item["id"] for item in report["overlength"])
        detail = f" Overlength IDs: {overlong}." if overlong else ""
        super().__init__("Tokenization preflight failed; no model weights were loaded." + detail)


def _require_finite_scalar(value: Any, label: str) -> float:
    """Inspect a scalar without detaching the original loss used for backward."""
    if hasattr(value, "detach"):
        value = value.detach().item()
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FloatingPointError(f"{label} must be a finite scalar") from exc
    if isinstance(value, bool) or not math.isfinite(number):
        raise FloatingPointError(f"Non-finite {label}; refusing to publish a completed adapter")
    return number


class _FiniteLossTrainerMixin:
    """Preserve Trainer's loss scaling and stop nonfinite loss before backward."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        if model.training and inputs.get("attention_mask") is not None:
            lengths = inputs["attention_mask"].detach().sum(dim=-1)
            self._observed_train_batches = getattr(self, "_observed_train_batches", 0) + 1
            self._observed_train_sequences = getattr(self, "_observed_train_sequences", 0) + lengths.numel()
            self._observed_train_max = max(
                getattr(self, "_observed_train_max", 0), int(lengths.max().item()),
            )
            self._observed_train_padded_max = max(
                getattr(self, "_observed_train_padded_max", 0), int(inputs["attention_mask"].shape[-1]),
            )
        result = super().compute_loss(
            model, inputs, return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch, **kwargs,
        )
        loss = result[0] if return_outputs else result
        _require_finite_scalar(loss, "training loss" if model.training else "evaluation loss")
        return result

    def observed_training_lengths(self) -> dict[str, int]:
        return {
            "forward_batches": getattr(self, "_observed_train_batches", 0),
            "sequences": getattr(self, "_observed_train_sequences", 0),
            "max_sequence_tokens": getattr(self, "_observed_train_max", 0),
            "max_padded_sequence_tokens": getattr(self, "_observed_train_padded_max", 0),
        }


def _finite_adapter_statistics(model: Any) -> dict[str, int]:
    """Check exactly the adapter-only state we save; do not copy base weights."""
    import torch
    from peft import get_peft_model_state_dict

    state = get_peft_model_state_dict(model, save_embedding_layers=False)
    if not state:
        raise ValueError("No adapter tensors are available to save")
    parameters = 0
    for name, tensor in state.items():
        if not torch.isfinite(tensor.detach()).all().item():
            raise FloatingPointError(f"Non-finite adapter tensor {name!r}; refusing to save it")
        parameters += tensor.numel()
    return {"tensors": len(state), "parameters": parameters}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    # Atomic replacement avoids leaving a plausible but partial run manifest.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prepare_output(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Output is not a directory: {path}")
    if path.exists():
        unexpected = [item.name for item in path.iterdir() if item.name != "preflight.json"]
        if unexpected:
            raise FileExistsError(
                f"Refusing to overwrite a nonempty output directory: {path}. "
                "Choose a new run directory."
            )
    path.mkdir(parents=True, exist_ok=True)


def _load_tokenizer(config: TrainConfig) -> tuple[Any, str]:
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Tokenizer preflight requires the optional training dependencies.") from exc
    model_config = AutoConfig.from_pretrained(
        config.model_id, revision=config.revision, trust_remote_code=False,
        local_files_only=config.local_files_only,
    )
    resolved = getattr(model_config, "_commit_hash", None)
    if not resolved and re.fullmatch(r"[0-9a-f]{40}", config.revision):
        resolved = config.revision
    if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise RuntimeError(
            "Could not resolve an immutable Hub model revision. Use a Hub model ID "
            "and a full commit SHA; unversioned local model directories are unsupported."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id, revision=resolved, trust_remote_code=False,
        local_files_only=config.local_files_only, use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer, str(resolved)


def tokenize_splits(
    tokenizer: Any,
    splits: dict[str, list[TrainingExample]],
    *, max_length: int, overlength_policy: Literal["error", "skip"] = "error",
) -> tuple[dict[str, list[dict[str, list[int]]]], dict[str, list[TrainingExample]], dict[str, Any]]:
    """Preflight train/validation only; fail on overlength unless skipping is explicit."""
    if set(splits) != {"train", "validation"}:
        raise ValueError("Only train and validation splits may enter SFT tokenization")
    if overlength_policy not in {"error", "skip"}:
        raise ValueError("overlength_policy must be error or skip")
    tokens: dict[str, list[dict[str, list[int]]]] = {"train": [], "validation": []}
    kept: dict[str, list[TrainingExample]] = {"train": [], "validation": []}
    report: dict[str, Any] = {
        "valid": True, "policy": overlength_policy, "max_length": max_length,
        "overlength": [], "skipped": [], "rejected": [], "counts": {},
        "truncated_rows": 0, "test_rows_tokenized": 0,
        "thinking": False, "loss": "completion tokens and EOS only",
    }
    for split in ("train", "validation"):
        for example in splits[split]:
            if not example.accepted or example.split != split:
                raise ValueError("Tokenization accepts only explicitly accepted rows in their own split")
            try:
                feature = prepare_completion_tokens(
                    tokenizer, example.prompt_messages(), example.completion, max_length,
                )
            except SequenceTooLongError as exc:
                detail = {"id": example.id, "split": split, "tokens": exc.actual_length}
                report["overlength"].append(detail)
                report["skipped" if overlength_policy == "skip" else "rejected"].append(detail)
                continue
            except ValueError as exc:
                raise ValueError(f"Invalid tokenization for example {example.id!r}: {exc}") from exc
            tokens[split].append(feature)
            kept[split].append(example)
        lengths = [len(feature["input_ids"]) for feature in tokens[split]]
        report["counts"][split] = {
            "input_rows": len(splits[split]), "usable_rows": len(kept[split]),
            "total_tokens": sum(lengths), "max_tokens": max(lengths, default=0),
            "supervised_tokens": sum(
                sum(label != -100 for label in feature["labels"]) for feature in tokens[split]
            ),
        }
    if (report["overlength"] and overlength_policy == "error") or not all(kept.values()):
        report["valid"] = False
        if not all(kept.values()):
            report["empty_splits"] = [name for name, rows in kept.items() if not rows]
        raise TokenizationPreflightError(report)
    return tokens, kept, report


def _runtime_versions() -> dict[str, str]:
    versions = {}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Missing {package!r}; install the project's optional training dependencies. "
                "A data-only --dry-run does not require them."
            ) from exc
    numbers = re.match(r"(\d+)\.(\d+)\.(\d+)", versions["transformers"])
    if not numbers or not ( (4, 57, 1) <= tuple(map(int, numbers.groups())) < (5, 0, 0) ):
        raise RuntimeError("This training entrypoint supports transformers>=4.57.1,<5; use the pinned training environment.")
    return versions


def _provenance(examples: list[TrainingExample]) -> list[dict[str, Any]]:
    result = []
    for example in examples:
        encoded = json.dumps(example.model_dump(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        result.append({
            "id": example.id, "group_id": example.group_id, "approach_id": example.approach_id,
            "split": example.split, "mode": example.mode, "accepted": example.accepted,
            "provenance": example.provenance, "example_sha256": sha256(encoded).hexdigest(),
        })
    return result


def train(
    config_path: str | Path, output_dir: str | Path, dry_run: bool = False,
) -> dict[str, Any]:
    """Validate or run real single-GPU adapter SFT.

    ``dry_run=True`` checks data/config only by default: no imports of torch, no
    downloads, no token-length claim. Set YAML ``preflight_tokenizer: true`` to
    additionally download/cache tokenizer+config and check lengths during a dry
    run. Actual training always performs tokenizer preflight before loading the
    base weights. Only accepted train rows optimize parameters; accepted
    validation rows choose the lowest-loss checkpoint; test rows are never
    tokenized, scored, or trained here.
    """
    source = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    config = TrainConfig.model_validate(raw)
    dataset_path = Path(config.dataset).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = source.parent / dataset_path
    dataset_path = dataset_path.resolve()
    examples, dataset_report = load_training_examples(dataset_path)
    splits = {name: accepted_split(examples, config.mode, name) for name in ("train", "validation")}
    if not all(splits.values()):
        raise ValueError(f"Mode {config.mode!r} needs accepted train and validation examples")
    destination = Path(output_dir).expanduser().resolve()
    _prepare_output(destination)
    report: dict[str, Any] = {
        "status": "data_validated", "dry_run": dry_run,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(source), "output_dir": str(destination),
        "config": config.model_dump(), "dataset": dataset_report,
        "selected_rows": {name: len(rows) for name, rows in splits.items()},
        "excluded_unaccepted_rows": sum(not example.accepted for example in examples),
        "test_rows_used": 0, "tokenizer_checked": False, "model_loaded": False,
        "tokenizer_preflight_note": "Data-only dry run does not establish token length or GPU fit.",
    }
    _write_json(destination / "preflight.json", report)
    if dry_run and not config.preflight_tokenizer:
        return report

    torch: Any = None
    if not dry_run:
        report["package_versions"] = _runtime_versions()
        import torch as torch_module
        torch = torch_module
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("This entrypoint intentionally supports one GPU/process only")
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA training requires a CUDA GPU; none was detected")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The configured QLoRA recipe requires a bf16-capable CUDA GPU")
        torch.cuda.set_device(0)
        report["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        }

    tokenizer, resolved_revision = _load_tokenizer(config)
    report["resolved_model_revision"] = resolved_revision
    report["chat_template_sha256"] = sha256(tokenizer.get_chat_template().encode("utf-8")).hexdigest()
    try:
        tokenized, selected, token_report = tokenize_splits(
            tokenizer, splits, max_length=config.max_length,
            overlength_policy=config.overlength_policy,
        )
    except TokenizationPreflightError as exc:
        report.update(status="tokenization_failed", tokenizer_checked=True, tokenization=exc.report)
        _write_json(destination / "preflight.json", report)
        raise
    except ValueError as exc:
        report.update(status="tokenization_failed", tokenizer_checked=True, error=str(exc))
        _write_json(destination / "preflight.json", report)
        raise
    report.update(status="tokenization_validated", tokenizer_checked=True, tokenization=token_report)
    report["selected_rows"] = {name: len(rows) for name, rows in selected.items()}
    _write_json(destination / "preflight.json", report)
    if dry_run:
        return report

    from transformers import (
        AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainerCallback,
        TrainingArguments, set_seed,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    set_seed(config.seed)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, revision=resolved_revision, trust_remote_code=False,
        local_files_only=config.local_files_only, torch_dtype=torch.bfloat16,
        quantization_config=quantization, device_map={"": 0}, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora = LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM", revision=resolved_revision,
    )
    model = get_peft_model(model, lora)
    report["model_loaded"] = True
    report["trainable_parameters"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    train_examples = selected["train"]

    class BalancedSampler(torch.utils.data.Sampler):
        def __init__(self) -> None:
            self.epoch = 0

        def __len__(self) -> int:
            return len(train_examples)

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

        def __iter__(self):
            return iter(balanced_epoch_indices(train_examples, seed=config.seed, epoch=self.epoch))

    class BalancedTrainer(_FiniteLossTrainerMixin, Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return balanced_sampler

    # Accelerate normally propagates set_epoch to the sampler. The callback also
    # sets it explicitly, keeping behavior stable across supported versions.
    balanced_sampler = BalancedSampler()

    class BalanceEpochCallback(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kwargs):
            balanced_sampler.set_epoch(int(state.epoch or 0))
            return control

    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        batch = pad_completion_batch(features, tokenizer.pad_token_id)
        return {key: torch.tensor(values, dtype=torch.long) for key, values in batch.items()}

    epoch0 = balanced_epoch_indices(train_examples, seed=config.seed, epoch=0)
    report["sampling"] = {
        "method": "shuffled round-robin prompt families, then accepted approaches, then rows",
        "samples_per_epoch": len(train_examples), "seed": config.seed,
        "epoch0_ids": [train_examples[index].id for index in epoch0],
        "epoch0_group_counts": dict(Counter(train_examples[index].group_id for index in epoch0)),
    }
    report["estimated_optimizer_steps"] = config.max_steps or math.ceil(
        len(train_examples) / (config.micro_batch_size * config.gradient_accumulation_steps)
    ) * math.ceil(config.num_train_epochs)
    controls = _trainer_controls(config)
    report["trainer_controls"] = controls
    arguments = TrainingArguments(
        output_dir=str(destination / "checkpoints"), overwrite_output_dir=False,
        per_device_train_batch_size=config.micro_batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate, num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio, lr_scheduler_type="cosine",
        bf16=True, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=config.save_total_limit, save_safetensors=True,
        logging_steps=config.logging_steps, logging_first_step=True,
        prediction_loss_only=True, report_to=[], seed=config.seed, data_seed=config.seed,
        remove_unused_columns=False, label_names=["labels"], dataloader_num_workers=0,
        **controls,
    )
    trainer = BalancedTrainer(
        model=model, args=arguments, train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"], data_collator=collate, processing_class=tokenizer,
        callbacks=[BalanceEpochCallback()],
    )
    _write_json(destination / "training_config.json", {
        **config.model_dump(), "resolved_model_revision": resolved_revision,
        "dataset_absolute_path": str(dataset_path), "base_precision": "nf4-double-quant",
        "compute_precision": "bf16", "thinking": False,
        "trainer_controls": controls,
    })
    _write_json(destination / "provenance.json", {
        "dataset_sha256": dataset_report["sha256"],
        "verification": dataset_report["verification"],
        "train": _provenance(selected["train"]),
        "validation": _provenance(selected["validation"]),
        "test_rows_used": 0,
    })
    report["status"] = "training"
    _write_json(destination / "training_report.json", report)
    torch.cuda.reset_peak_memory_stats(0)
    try:
        result = trainer.train()
        validation = trainer.evaluate()
        train_loss = _require_finite_scalar(result.metrics.get("train_loss"), "reported training loss")
        eval_loss = _require_finite_scalar(validation.get("eval_loss"), "final validation loss")
        adapter_statistics = _finite_adapter_statistics(model)
        model.save_pretrained(destination, safe_serialization=True, save_embedding_layers=False)
        tokenizer.save_pretrained(destination)
        trainer.save_state()
        report.update(
            status="completed", training_metrics=result.metrics, validation_metrics=validation,
            numerical_checks={"training_loss": train_loss, "validation_loss": eval_loss,
                              "finite_adapter": adapter_statistics},
            observed_training=trainer.observed_training_lengths(),
            best_checkpoint=trainer.state.best_model_checkpoint,
            optimizer_steps=trainer.state.global_step,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            completed_utc=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      observed_training=trainer.observed_training_lengths())
        _write_json(destination / "training_report.json", report)
        raise
    _write_json(destination / "training_report.json", report)
    return report
