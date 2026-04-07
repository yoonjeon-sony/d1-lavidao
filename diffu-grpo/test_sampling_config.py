from collections import defaultdict
from contextlib import contextmanager
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from diffu_grpo_config import DiffuGRPOConfig
from diffu_grpo_trainer import DiffuGRPOTrainer


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def decode(self, ids, skip_special_tokens=True):
        return "decoded"


class _FakeAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.process_index = 0
        self.is_main_process = False

    def gather_for_metrics(self, value):
        return value

    def unwrap_model(self, model):
        return model


class _FakeBaseModel:
    def __init__(self):
        self.embed_tokens = MagicMock(return_value=torch.zeros(1, 3, 4))


class _FakeModel:
    def __init__(self):
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.config = SimpleNamespace()
        self.base_model = _FakeBaseModel()
        self.generate = MagicMock(return_value=torch.tensor([[7, 8]]))
        self.prepare_inputs_labels_for_multimodal = MagicMock()

    def get_model(self):
        return self.base_model


class SamplingConfigTest(unittest.TestCase):
    def _make_rollout_trainer(self):
        trainer = DiffuGRPOTrainer.__new__(DiffuGRPOTrainer)
        trainer.args = SimpleNamespace(
            use_fast_dlm=False,
            temperature=0.7,
            text_rollout_force_prefix_lm=None,
            prefix_lm=True,
            image_edit_sample_policy="multinomial",
            image_edit_confidence_policy="halton",
            image_edit_guidance_scale=0.0,
            image_edit_guidance_scale_image=5.0,
            image_edit_batch_size=1,
            image_edit_resolution=1024,
            image_edit_n_tokens=4096,
            image_edit_shift=5,
            image_edit_n_steps=64,
            image_edit_schedule="shift",
            image_edit_alg_temp=5.0,
            image_edit_dynamic_temperature=True,
            image_edit_schedule_temp="cosine2",
            image_edit_min_temperature=0.5,
            image_edit_micro_cond="",
            image_edit_schedule_temp_samp="linear",
            image_edit_dynamic_temperature_samp=False,
            image_edit_min_temperature_samp=1.0,
            image_edit_cfg_interval_start=0.0,
            image_edit_cfg_interval_end=1.0,
            image_edit_order_cutoff=1.0,
            image_edit_edit_mode=0,
            max_completion_length=8,
            block_length=4,
            text_rollout_step_per_block=None,
            text_rollout_do_sample=False,
            remasking=None,
            cfg_scale=None,
            beta=0.0,
            num_iterations=1,
            random_masking=False,
            p_mask_prompt=0.0,
            output_dir="/tmp/dummy",
            logging_steps=1,
            report_to=[],
        )
        trainer.processing_class = _FakeTokenizer()
        trainer.max_prompt_length = None
        trainer.accelerator = _FakeAccelerator()
        trainer.control = SimpleNamespace(should_evaluate=False)
        trainer.state = SimpleNamespace(global_step=0)
        trainer.log_completions = False
        trainer.reward_processing_classes = [None]
        trainer.reward_weights = [1.0]
        trainer.num_generations = 1
        trainer._metrics = {
            "train": defaultdict(list),
            "eval": defaultdict(list),
        }
        trainer._step = 0
        trainer.model_wrapped = object()

        def dummy_reward(prompts, completions, **kwargs):
            return [0.0] * len(prompts)

        trainer.reward_funcs = [dummy_reward]
        return trainer

    def test_config_defaults(self):
        cfg = DiffuGRPOConfig(output_dir="/tmp/dummy")
        self.assertEqual(cfg.temperature, 0.1)
        self.assertFalse(cfg.use_fast_dlm)
        self.assertEqual(cfg.version, "llada")
        self.assertTrue(cfg.load_vlm)
        self.assertTrue(cfg.prefix_lm)
        self.assertTrue(cfg.unified_gen)
        self.assertEqual(cfg.mm_vision_tower_lr, 2e-6)
        self.assertTrue(cfg.enc_use_image_branch)
        self.assertEqual(cfg.image_edit_sample_policy, "multinomial")
        self.assertEqual(cfg.image_edit_confidence_policy, "halton")
        self.assertEqual(cfg.image_edit_guidance_scale, 0.0)
        self.assertEqual(cfg.image_edit_resolution, 1024)
        self.assertEqual(cfg.image_edit_n_tokens, 4096)
        self.assertEqual(cfg.image_edit_n_steps, 64)
        self.assertEqual(cfg.image_edit_schedule, "shift")
        self.assertEqual(cfg.image_edit_alg_temp, 5.0)
        self.assertTrue(cfg.image_edit_dynamic_temperature)
        self.assertEqual(cfg.image_edit_schedule_temp, "cosine2")
        self.assertEqual(cfg.image_edit_min_temperature, 0.5)
        self.assertEqual(cfg.image_edit_micro_cond, "")

    def test_trainer_sampling_helpers(self):
        trainer = DiffuGRPOTrainer.__new__(DiffuGRPOTrainer)
        trainer.args = SimpleNamespace(
            use_fast_dlm=True,
            temperature=0.7,
            text_rollout_force_prefix_lm=None,
            prefix_lm=True,
            image_edit_sample_policy="multinomial",
            image_edit_confidence_policy="halton",
            image_edit_guidance_scale=0.0,
            image_edit_guidance_scale_image=5.0,
            image_edit_batch_size=1,
            image_edit_resolution=1024,
            image_edit_n_tokens=4096,
            image_edit_shift=5,
            image_edit_n_steps=64,
            image_edit_schedule="shift",
            image_edit_alg_temp=5.0,
            image_edit_dynamic_temperature=True,
            image_edit_schedule_temp="cosine2",
            image_edit_min_temperature=0.5,
            image_edit_micro_cond="",
            image_edit_schedule_temp_samp="linear",
            image_edit_dynamic_temperature_samp=False,
            image_edit_min_temperature_samp=1.0,
            image_edit_cfg_interval_start=0.0,
            image_edit_cfg_interval_end=1.0,
            image_edit_order_cutoff=1.0,
            image_edit_edit_mode=0,
        )
        trainer.processing_class = object()

        trainer._assert_sampling_constraints()
        self.assertFalse(trainer.args.use_fast_dlm)
        self.assertEqual(trainer._get_text_rollout_temperature(), 0.7)
        self.assertTrue(trainer._get_text_rollout_prefix_lm())

        gen_dict = trainer._get_image_edit_gen_dict()
        self.assertEqual(gen_dict["sample_policy"], "multinomial")
        self.assertEqual(gen_dict["confidence_policy"], "halton")
        self.assertEqual(gen_dict["guidance_scale"], 0.0)
        self.assertEqual(gen_dict["batch_size"], 1)
        self.assertEqual(gen_dict["image_resolution"], 1024)
        self.assertEqual(gen_dict["n_tokens"], 4096)
        self.assertEqual(gen_dict["shift"], 5)
        self.assertEqual(gen_dict["n_steps"], 64)
        self.assertEqual(gen_dict["schedule"], "shift")
        self.assertEqual(gen_dict["alg_temp"], 5.0)
        self.assertTrue(gen_dict["dynamic_temperature"])
        self.assertEqual(gen_dict["schedule_temp"], "cosine2")
        self.assertEqual(gen_dict["min_temperature"], 0.5)
        self.assertEqual(gen_dict["micro_cond"], "")

    def test_prefix_lm_output_normalization(self):
        trainer = DiffuGRPOTrainer.__new__(DiffuGRPOTrainer)
        prompt_ids = torch.tensor([[1, 2, 3]])

        full_sequence = torch.tensor([[1, 2, 3, 4, 5]])
        norm_prompt, completion = trainer._normalize_text_rollout(
            generated=full_sequence, prompt_ids=prompt_ids, prefix_lm=False
        )
        self.assertTrue(torch.equal(norm_prompt, prompt_ids))
        self.assertTrue(torch.equal(completion, torch.tensor([[4, 5]])))

        completion_only = torch.tensor([[7, 8, 9]])
        norm_prompt, completion = trainer._normalize_text_rollout(
            generated=completion_only, prompt_ids=prompt_ids, prefix_lm=True
        )
        self.assertTrue(torch.equal(norm_prompt, prompt_ids))
        self.assertTrue(torch.equal(completion, completion_only))

    def test_prepare_text_rollout_image_inputs_uses_pad_to_square_and_resize(self):
        trainer = self._make_rollout_trainer()
        loaded_image = MagicMock()
        converted_image = object()
        processed_image = SimpleNamespace(size=(1024, 1024))
        loaded_image.convert.return_value = converted_image
        model = SimpleNamespace(config=SimpleNamespace(), dtype=torch.float32)

        with patch.object(trainer, "_load_image", return_value=loaded_image), patch(
            "diffu_grpo_trainer.pad_to_square_and_resize",
            return_value=processed_image,
        ) as mock_pad, patch(
            "diffu_grpo_trainer.process_images",
            return_value=[torch.zeros(3, 4, 4)],
        ) as mock_process:
            image_tensor, image_sizes = trainer._prepare_text_rollout_image_inputs(
                {"image": "dummy.png"},
                model,
                object(),
                torch.device("cpu"),
            )

        mock_pad.assert_called_once_with(converted_image, 1024)
        mock_process.assert_called_once_with([processed_image], unittest.mock.ANY, model.config)
        self.assertEqual(image_sizes, [(1024, 1024)])
        self.assertEqual(len(image_tensor), 1)

    def test_rollout_multimodal_text_gen_returns_decoded_context(self):
        trainer = self._make_rollout_trainer()
        fake_model = _FakeModel()
        prompt_ids = torch.tensor([[1, 2, 3]])
        prompt_mask = torch.ones_like(prompt_ids)
        prepared_embeds = torch.randn(1, 4, 5, requires_grad=True)
        fake_model.prepare_inputs_labels_for_multimodal.return_value = (
            prompt_ids,
            None,
            prompt_mask,
            None,
            prepared_embeds,
            None,
        )

        with patch.object(
            trainer,
            "_prepare_text_rollout_image_inputs",
            return_value=([torch.zeros(3, 4, 4)], [(1024, 1024)]),
        ), patch(
            "diffu_grpo_trainer.llada_generate",
            return_value=torch.tensor([[7, 8]]),
        ) as mock_llada:
            generated, context = trainer._rollout_multimodal_text_gen(
                fake_model,
                {"image": "dummy.png"},
                "<image>\nDescribe the image.",
                prompt_ids,
                prompt_mask,
                object(),
                {"prefix_lm": True, "max_new_tokens": 8},
                torch.device("cpu"),
            )

        self.assertTrue(torch.equal(generated, torch.tensor([[7, 8]])))
        self.assertEqual(context["decoded_text"], "decoded")
        self.assertTrue(torch.equal(context["completion_ids"], torch.tensor([7, 8])))
        self.assertTrue(torch.equal(context["prompt_ids"], prompt_ids.squeeze(0)))
        self.assertFalse(context["inputs_embeds"].requires_grad)
        self.assertIsNot(context["inputs_embeds"], prepared_embeds.squeeze(0))
        llada_args, llada_kwargs = mock_llada.call_args
        self.assertIs(llada_args[0], fake_model.get_model())
        self.assertIs(llada_kwargs["inputs_embeds"], prepared_embeds)
        self.assertTrue(torch.equal(llada_kwargs["attention_mask"], prompt_mask))
        self.assertIsNone(llada_kwargs["position_ids"])
        self.assertTrue(llada_kwargs["prefix_lm"])
        self.assertEqual(llada_kwargs["max_new_tokens"], 8)

    def test_multimodal_text_rollout_uses_prepare_inputs_and_llada_generate(self):
        trainer = self._make_rollout_trainer()
        fake_model = _FakeModel()
        prepared_embeds = torch.randn(1, 4, 5)
        prompt_mask = torch.ones(1, 3, dtype=torch.long)
        fake_model.prepare_inputs_labels_for_multimodal.return_value = (
            torch.tensor([[1, 2, 3]]),
            None,
            prompt_mask,
            None,
            prepared_embeds,
            None,
        )

        @contextmanager
        def fake_unwrap(model_wrapped, accelerator):
            yield fake_model

        with patch("diffu_grpo_trainer.unwrap_model_for_generation", fake_unwrap), patch(
            "diffu_grpo_trainer.tokenizer_image_token",
            return_value=torch.tensor([1, 2, 3]),
        ), patch.object(trainer, "_get_image_processor", return_value=object()), patch.object(
            trainer,
            "_prepare_text_rollout_image_inputs",
            return_value=([torch.zeros(3, 4, 4)], [(1024, 1024)]),
        ), patch(
            "diffu_grpo_trainer.llada_generate",
            return_value=torch.tensor([[7, 8]]),
        ) as mock_llada, patch(
            "diffu_grpo_trainer.gather",
            side_effect=lambda x: x,
        ), patch(
            "diffu_grpo_trainer.gather_object",
            side_effect=lambda x: x,
        ):
            output = trainer._generate_and_score_completions(
                [{"task_type": "text", "prompt": "<image>\nDescribe the image.", "image": "dummy.png"}]
            )

        fake_model.generate.assert_not_called()
        fake_model.prepare_inputs_labels_for_multimodal.assert_called_once()
        self.assertEqual(output["text_completion_ids"][0].tolist(), [7, 8])
        self.assertEqual(output["text_prompt_ids"][0].tolist(), [1, 2, 3])
        llada_args, llada_kwargs = mock_llada.call_args
        self.assertIs(llada_args[0], fake_model.get_model())
        self.assertIs(llada_kwargs["inputs_embeds"], prepared_embeds)
        self.assertIs(llada_kwargs["attention_mask"], prompt_mask)

    def test_text_only_rollout_uses_embed_tokens_and_skips_multimodal_prepare(self):
        trainer = self._make_rollout_trainer()
        fake_model = _FakeModel()
        embedded_prompt = torch.randn(1, 3, 4)
        fake_model.base_model.embed_tokens.return_value = embedded_prompt

        @contextmanager
        def fake_unwrap(model_wrapped, accelerator):
            yield fake_model

        with patch("diffu_grpo_trainer.unwrap_model_for_generation", fake_unwrap), patch(
            "diffu_grpo_trainer.tokenizer_image_token",
            return_value=torch.tensor([1, 2, 3]),
        ), patch.object(trainer, "_get_image_processor", return_value=object()), patch.object(
            trainer,
            "_prepare_text_rollout_image_inputs",
            return_value=(None, None),
        ), patch(
            "diffu_grpo_trainer.llada_generate",
            return_value=torch.tensor([[7, 8]]),
        ) as mock_llada, patch(
            "diffu_grpo_trainer.gather",
            side_effect=lambda x: x,
        ), patch(
            "diffu_grpo_trainer.gather_object",
            side_effect=lambda x: x,
        ):
            output = trainer._generate_and_score_completions(
                [{"task_type": "text", "prompt": "Describe the text-only task."}]
            )

        fake_model.generate.assert_not_called()
        fake_model.prepare_inputs_labels_for_multimodal.assert_not_called()
        self.assertEqual(output["text_completion_ids"][0].tolist(), [7, 8])
        embed_args, _ = fake_model.base_model.embed_tokens.call_args
        self.assertTrue(torch.equal(embed_args[0], torch.tensor([[1, 2, 3]])))
        llada_args, llada_kwargs = mock_llada.call_args
        self.assertIs(llada_args[0], fake_model.get_model())
        self.assertIs(llada_kwargs["inputs_embeds"], embedded_prompt)

    def test_multimodal_text_rollout_requires_image_token_in_prompt(self):
        trainer = self._make_rollout_trainer()
        fake_model = _FakeModel()

        @contextmanager
        def fake_unwrap(model_wrapped, accelerator):
            yield fake_model

        with patch("diffu_grpo_trainer.unwrap_model_for_generation", fake_unwrap), patch.object(
            trainer,
            "_get_image_processor",
            return_value=object(),
        ):
            with self.assertRaisesRegex(ValueError, "does not contain '<image>'"):
                trainer._generate_and_score_completions(
                    [{"task_type": "text", "prompt": "Describe the image.", "image": "dummy.png"}]
                )


if __name__ == "__main__":
    unittest.main()
