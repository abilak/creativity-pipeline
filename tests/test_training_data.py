"""Data correctness tests use a fake tokenizer; no model download or GPU."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml

from creativity_pipeline.training import (
    TrainConfig, TokenizationPreflightError, _require_finite_scalar, _trainer_controls,
    tokenize_splits, train,
)
from creativity_pipeline.training_data import (
    SequenceTooLongError, TrainingExample, accepted_split, balanced_epoch_indices,
    pad_completion_batch, prepare_completion_tokens, validate_dataset,
)


class FakeTokenizer:
    eos_token_id = 100000
    pad_token_id = 100000  # Deliberately exercise the pad==EOS case.
    all_special_tokens = ["<|im_start|>", "<|im_end|>"]
    chat_template = "fake-non-thinking-template"

    def __init__(self):
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant:\n<think>\n\n</think>\n\n"

    def encode(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("the chat template already supplies special tokens")
        # A fictional cross-boundary merge makes an incorrect full-string masking
        # implementation observable. Our prefix ends with two newlines.
        if text.endswith("\n\nA"):
            return [ord(character) + 10 for character in text[:-3]] + [99999]
        return [ord(character) + 10 for character in text]


def example(identifier="train-1", group="train-group", split="train", **kwargs):
    payload = {
        "id": identifier, "group_id": group, "approach_id": "causal",
        "mode": "history", "split": split,
        "messages": [{"role": "user", "content": f"Evidence packet for {group}."}],
        "completion": json.dumps({"text": "The evidence supports the narrower claim [S1]."}),
        "accepted": True, "provenance": "Reviewer checked this synthetic unit-test example.",
    }
    payload.update(kwargs)
    return payload


class DatasetValidationTests(unittest.TestCase):
    def write_rows(self, directory, rows):
        path = Path(directory) / "examples.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_accepted_selection_excludes_rejected_and_test_rows(self):
        rows = [
            example(), example("bad", "bad-group", accepted=False),
            example("val", "val-group", "validation"), example("test", "test-group", "test"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = validate_dataset(self.write_rows(directory, rows))
        self.assertTrue(report["valid"])
        self.assertEqual(report["accepted_rows"], 3)
        selected = accepted_split([TrainingExample.model_validate(row) for row in rows], "history", "train")
        self.assertEqual([row.id for row in selected], ["train-1"])

    def test_group_leakage_fails_even_for_rejected_row(self):
        rows = [example(), example("val", "val-group", "validation")]
        rows.append(example("rejected", "train-group", "test", accepted=False,
                            messages=[{"role": "user", "content": "A paraphrased variant."}]))
        with tempfile.TemporaryDirectory() as directory:
            report = validate_dataset(self.write_rows(directory, rows))
        self.assertFalse(report["valid"])
        self.assertEqual(report["group_leakage"][0]["group_id"], "train-group")

    def test_copied_prompt_leaks_even_with_different_group_ids(self):
        rows = [example(), example("val", "val-group", "validation",
                                 messages=[{"role": "user", "content": "Evidence  packet for train-group. "}])]
        with tempfile.TemporaryDirectory() as directory:
            report = validate_dataset(self.write_rows(directory, rows))
        self.assertFalse(report["valid"])
        self.assertTrue(report["prompt_leakage"])

    def test_strict_boolean_empty_provenance_and_duplicate_ids_fail(self):
        for change in ({"accepted": "true"}, {"provenance": "   "}, {"id": "val"}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                rows = [example(**change), example("val", "val-group", "validation")]
                report = validate_dataset(self.write_rows(directory, rows))
                self.assertFalse(report["valid"])

    def test_validation_split_required(self):
        with tempfile.TemporaryDirectory() as directory:
            report = validate_dataset(self.write_rows(directory, [example()]))
        self.assertFalse(report["valid"])
        self.assertTrue(any("validation split" in message for message in report["errors"]))

    def test_unknown_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            TrainingExample.model_validate(example(approval="automatic"))

    def test_accepted_targets_must_be_pipeline_plans_or_drafts(self):
        plan = {"id": "p1", "approach": "causal", "intent": "Inspect causation",
                "moves": [{"id": "m1", "description": "Explain the evidence", "claim_status": "inference"}],
                "presentation": ["m1"]}
        TrainingExample.model_validate(example(completion=json.dumps(plan)))
        TrainingExample.model_validate(example(completion='{"text": "A valid draft."}'))
        for target in ("Generic raw prose", '{"decision":"accept"}',
                       '{"text":"one","text":"two"}', '{"text":"ok","claims":[{"id":"c"}]}'):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "Accepted completion"):
                TrainingExample.model_validate(example(completion=target))
            rejected = TrainingExample.model_validate(example(completion=target, accepted=False))
            self.assertFalse(rejected.accepted)

    def test_offline_dry_run_does_not_load_a_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_rows(root, [example(), example("val", "val-group", "validation")])
            config = root / "train.yaml"
            config.write_text("dataset: examples.jsonl\nmode: history\n", encoding="utf-8")
            with patch("creativity_pipeline.training._load_tokenizer") as loader:
                report = train(config, root / "run", dry_run=True)
                loader.assert_not_called()
            self.assertFalse(report["tokenizer_checked"])
            self.assertFalse(report["model_loaded"])
            self.assertEqual(report["test_rows_used"], 0)
            self.assertTrue((root / "run" / "preflight.json").exists())

    def test_training_requires_validation_for_the_selected_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_rows(root, [example(mode="english"), example("val", "val-group", "validation")])
            config = root / "train.yaml"
            config.write_text("dataset: examples.jsonl\nmode: english\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "accepted train and validation"):
                train(config, root / "run", dry_run=True)


class CompletionMaskingTests(unittest.TestCase):
    def test_generation_prefix_is_masked_and_target_and_eos_are_supervised(self):
        tokenizer = FakeTokenizer()
        messages = [{"role": "user", "content": "Inspect this plan."}]
        tokens = prepare_completion_tokens(tokenizer, messages, "A", 200)
        prefix = tokenizer.apply_chat_template(messages)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        self.assertEqual(tokens["input_ids"][:len(prefix_ids)], prefix_ids)
        self.assertEqual(tokens["labels"][:len(prefix_ids)], [-100] * len(prefix_ids))
        self.assertEqual(tokens["labels"][len(prefix_ids):], [ord("A") + 10, tokenizer.eos_token_id])
        # The fake merge spans prompt and answer only when tokenized together.
        self.assertNotEqual(tokens["input_ids"][:-1], tokenizer.encode(prefix + "A", add_special_tokens=False))

    def test_thinking_disabled_and_no_automatic_special_token_duplication(self):
        tokenizer = FakeTokenizer()
        prepare_completion_tokens(tokenizer, [{"role": "user", "content": "Plan"}], "Answer", 200)
        self.assertEqual(tokenizer.template_kwargs,
                         {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False})

    def test_overlength_raises_instead_of_truncating_the_ending(self):
        tokenizer = FakeTokenizer()
        with self.assertRaises(SequenceTooLongError) as caught:
            prepare_completion_tokens(tokenizer, [{"role": "user", "content": "Plan"}], "Ending" * 20, 50)
        self.assertGreater(caught.exception.actual_length, caught.exception.max_length)

    def test_padding_preserves_real_eos_when_eos_is_pad(self):
        tokenizer = FakeTokenizer()
        messages = [{"role": "user", "content": "Plan"}]
        short = prepare_completion_tokens(tokenizer, messages, "A", 200)
        long = prepare_completion_tokens(tokenizer, messages, "Long ending", 200)
        batch = pad_completion_batch([short, long], tokenizer.eos_token_id)
        boundary = len(short["input_ids"])
        self.assertEqual(batch["labels"][0][boundary - 1], tokenizer.eos_token_id)
        self.assertTrue(all(label == -100 for label in batch["labels"][0][boundary:]))
        self.assertTrue(all(mask == 0 for mask in batch["attention_mask"][0][boundary:]))

    def test_control_tokens_and_thinking_blocks_are_rejected(self):
        for target in ("Text<|im_end|>", "<think>private notes</think>Final"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                prepare_completion_tokens(FakeTokenizer(), [{"role": "user", "content": "Plan"}], target, 200)

    def test_preflight_reports_overlength_default_error_and_explicit_skip(self):
        rows = {
            "train": [TrainingExample.model_validate(example()),
                      TrainingExample.model_validate(example("long", "long-group", completion=json.dumps({"text": "x" * 800})))],
            "validation": [TrainingExample.model_validate(example("val", "val-group", "validation"))],
        }
        with self.assertRaises(TokenizationPreflightError) as caught:
            tokenize_splits(FakeTokenizer(), rows, max_length=300)
        self.assertEqual(caught.exception.report["rejected"][0]["id"], "long")
        self.assertEqual(caught.exception.report["skipped"], [])
        tokens, kept, report = tokenize_splits(FakeTokenizer(), rows, max_length=300, overlength_policy="skip")
        self.assertEqual([row.id for row in kept["train"]], ["train-1"])
        self.assertEqual(report["skipped"][0]["id"], "long")
        self.assertEqual(report["truncated_rows"], 0)
        self.assertEqual(len(tokens["validation"]), 1)

    def test_test_split_cannot_enter_tokenizer_preflight(self):
        with self.assertRaisesRegex(ValueError, "Only train and validation"):
            tokenize_splits(FakeTokenizer(), {"train": [], "validation": [], "test": []}, max_length=100)


class BalancedSamplingTests(unittest.TestCase):
    def test_balances_groups_then_approaches_instead_of_row_volume(self):
        rows = [TrainingExample.model_validate(example(f"common-{i}", "group-a", approach_id="common"))
                for i in range(19)]
        rows += [TrainingExample.model_validate(example("rare", "group-a", approach_id="rare")),
                 TrainingExample.model_validate(example("other", "group-b", approach_id="one"))]
        indices = balanced_epoch_indices(rows, seed=5, epoch=0, num_samples=60)
        groups = Counter(rows[index].group_id for index in indices)
        approaches = Counter(rows[index].approach_id for index in indices if rows[index].group_id == "group-a")
        self.assertEqual(groups, {"group-a": 30, "group-b": 30})
        self.assertEqual(approaches, {"common": 15, "rare": 15})
        self.assertEqual(indices, balanced_epoch_indices(rows, seed=5, epoch=0, num_samples=60))

    def test_sampler_refuses_unaccepted_or_heldout_rows(self):
        for kwargs in ({"accepted": False}, {"split": "validation"}, {"split": "test"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                balanced_epoch_indices([TrainingExample.model_validate(example(**kwargs))], seed=1, epoch=0)


class TrainingControlTests(unittest.TestCase):
    def test_nonfinite_or_missing_metrics_cannot_complete_a_training_run(self):
        for value in (float("nan"), float("inf"), float("-inf"), None, True):
            with self.subTest(value=value), self.assertRaises(FloatingPointError):
                _require_finite_scalar(value, "final validation loss")
        self.assertEqual(_require_finite_scalar(0.125, "final validation loss"), 0.125)

    def test_normal_run_uses_epoch_evaluation_and_preserves_the_balanced_sampler(self):
        config = TrainConfig(dataset="reviewed.jsonl", mode="english")
        controls = _trainer_controls(config)
        self.assertIsNone(config.max_steps)
        self.assertEqual(controls["max_steps"], -1)
        self.assertEqual(controls["eval_strategy"], "epoch")
        self.assertEqual(controls["save_strategy"], "epoch")
        self.assertNotIn("eval_steps", controls)
        self.assertEqual(controls["accelerator_config"], {
            "use_seedable_sampler": False, "split_batches": False,
            "dispatch_batches": False, "even_batches": False,
        })

    def test_smoke_run_limits_optimizer_steps_and_saves_after_step_two(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "train_smoke.yaml"
        config = TrainConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        controls = _trainer_controls(config)
        self.assertEqual(config.dataset, "../data/curated/english.jsonl")
        self.assertEqual(config.max_length, 2048)
        self.assertEqual(config.micro_batch_size, 1)
        self.assertEqual(config.gradient_accumulation_steps, 1)
        self.assertEqual(config.overlength_policy, "error")
        self.assertEqual(controls["max_steps"], 2)
        self.assertEqual(controls["eval_strategy"], "steps")
        self.assertEqual(controls["save_strategy"], "steps")
        self.assertEqual(controls["eval_steps"], 2)
        self.assertEqual(controls["save_steps"], 2)

    def test_max_steps_requires_a_positive_integer_when_present(self):
        for value in (0, -1, True, 2.0, "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TrainConfig(dataset="reviewed.jsonl", mode="english", max_steps=value)


if __name__ == "__main__":
    unittest.main()
