import unittest
from types import SimpleNamespace

import torch

from diffu_grpo_config import DiffuGRPOConfig
from diffu_grpo_trainer import DiffuGRPOTrainer


class SamplingConfigTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
