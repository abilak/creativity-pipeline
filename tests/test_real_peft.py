"""Optional real-library regressions: tiny CPU Qwen models, no Hub downloads.

These exercise actual tensor serialization/loading and Trainer loss scaling.
They do not establish CUDA, bitsandbytes, 8B-model fit, or writing quality.
"""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from creativity_pipeline.backends import BackendError, _load_complete_adapter
from creativity_pipeline.training import _FiniteLossTrainerMixin, _finite_adapter_statistics


_REQUIRED = ("torch", "transformers", "peft", "accelerate", "safetensors")
_MISSING = [name for name in _REQUIRED if importlib.util.find_spec(name) is None]


@unittest.skipIf(bool(_MISSING), "Optional real-model dependencies absent: " + ", ".join(_MISSING))
class RealPeftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        offline = patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        offline.start()
        cls.addClassCleanup(offline.stop)
        import torch
        import peft
        from safetensors import torch as safe_torch
        from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer, TrainingArguments

        cls.torch, cls.peft, cls.safe_torch = torch, peft, safe_torch
        cls.Qwen3Config, cls.Qwen3ForCausalLM = Qwen3Config, Qwen3ForCausalLM
        cls.Trainer, cls.TrainingArguments = Trainer, TrainingArguments

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)

    def new_base(self):
        self.torch.manual_seed(345)
        config = self.Qwen3Config(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=64,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "eager"
        config._name_or_path = "local-config-created-test-model"
        return self.Qwen3ForCausalLM(config)

    def saved_adapter(self):
        config = self.peft.LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM", revision="a" * 40,
        )
        adapted = self.peft.get_peft_model(self.new_base(), config, autocast_adapter_dtype=False)
        with self.torch.no_grad():
            for name, parameter in adapted.named_parameters():
                if "lora_B" in name:
                    parameter.copy_(self.torch.randn_like(parameter) * 0.03)
        adapted.eval()
        adapted.save_pretrained(self.path, safe_serialization=True, save_embedding_layers=False)
        return adapted

    def test_real_adapter_round_trip_preserves_tensors_and_logits(self):
        original = self.saved_adapter()
        loaded = _load_complete_adapter(self.new_base(), str(self.path), self.peft).eval()
        expected = self.peft.get_peft_model_state_dict(original, save_embedding_layers=False)
        actual = self.peft.get_peft_model_state_dict(loaded, save_embedding_layers=False)
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            self.torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        ids = self.torch.tensor([[1, 7, 9, 5]])
        with self.torch.no_grad():
            self.torch.testing.assert_close(loaded(ids).logits, original(ids).logits, rtol=1e-5, atol=1e-6)
        self.assertTrue(all(not tensor.requires_grad for name, tensor in loaded.named_parameters() if "lora_" in name))

    def test_missing_real_adapter_tensor_is_rejected(self):
        self.saved_adapter()
        file = self.path / "adapter_model.safetensors"
        state = self.safe_torch.load_file(file)
        removed = next(name for name in state if "lora_B" in name)
        del state[removed]
        self.safe_torch.save_file(state, file)
        with self.assertRaisesRegex(BackendError, "tensor inventory.*missing_keys"):
            _load_complete_adapter(self.new_base(), str(self.path), self.peft)

    def test_unexpected_real_adapter_tensor_is_rejected(self):
        self.saved_adapter()
        file = self.path / "adapter_model.safetensors"
        state = self.safe_torch.load_file(file)
        state["base_model.model.nonexistent.lora_A.weight"] = self.torch.zeros(2, 32)
        self.safe_torch.save_file(state, file)
        with self.assertRaisesRegex(BackendError, "unexpected_keys"):
            _load_complete_adapter(self.new_base(), str(self.path), self.peft)

    def test_nonfinite_adapter_tensor_cannot_pass_publication_check(self):
        adapted = self.saved_adapter()
        self.assertGreater(_finite_adapter_statistics(adapted)["tensors"], 0)
        with self.torch.no_grad():
            next(parameter for name, parameter in adapted.named_parameters() if "lora_B" in name).fill_(float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "Non-finite adapter tensor"):
            _finite_adapter_statistics(adapted)

    def test_real_trainer_loss_scaling_and_observed_lengths_are_preserved(self):
        class GuardedTrainer(_FiniteLossTrainerMixin, self.Trainer):
            pass

        model = self.new_base()
        trainer = GuardedTrainer(model=model, args=self.TrainingArguments(
            output_dir=str(self.path / "trainer"), use_cpu=True, report_to=[],
            per_device_train_batch_size=1, max_steps=1,
        ))
        inputs = {
            "input_ids": self.torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 0, 0]]),
            "attention_mask": self.torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
            "labels": self.torch.tensor([[-100, -100, 3, 4, 5], [-100, -100, 6, -100, -100]]),
        }
        # Deliberately differs from this microbatch's target count. Trainer must
        # preserve the caller's accumulated-batch denominator through the guard.
        denominator = self.torch.tensor(11)
        loss, outputs = trainer.compute_loss(model, dict(inputs), return_outputs=True,
                                             num_items_in_batch=denominator)
        expected = self.torch.nn.functional.cross_entropy(
            outputs.logits[:, :-1].float().reshape(-1, 32), inputs["labels"][:, 1:].reshape(-1),
            ignore_index=-100, reduction="sum",
        ) / denominator
        self.torch.testing.assert_close(loss, expected)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(trainer.observed_training_lengths(), {
            "forward_batches": 1, "sequences": 2,
            "max_sequence_tokens": 5, "max_padded_sequence_tokens": 5,
        })
        model.eval()
        trainer.compute_loss(model, dict(inputs))
        self.assertEqual(trainer.observed_training_lengths()["forward_batches"], 1)
        model.train()
        with self.torch.no_grad():
            model.lm_head.weight.fill_(float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "Non-finite training loss"):
            trainer.compute_loss(model, dict(inputs), num_items_in_batch=denominator)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
