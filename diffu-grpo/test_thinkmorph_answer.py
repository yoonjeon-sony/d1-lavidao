import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset

import data_utils
import diffu_grpo_train
from data_utils import COT_PROMPT, get_image_answer_placeholder_questions
from reward_func import correctness_reward_func, soft_format_reward_func, strict_format_reward_func


class ThinkMorphAnswerLoaderTest(unittest.TestCase):
    def test_loader_emits_multimodal_text_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sample = {
                "pid": "sample-1",
                "question": "What number is highlighted?",
                "problem_image_0": "images/q1.png",
                "answer": " 42 ",
            }
            data_path = self._write_jsonl(temp_dir, data_utils.THINKMORPH_LOCAL_JSONL_FILES[0], [sample])
            self._write_empty_jsonl_files(temp_dir, exclude={data_path})

            with patch.object(data_utils, "THINKMORPH_DEFAULT_DATA_ROOT", temp_dir), patch.object(
                data_utils, "THINKMORPH_DEFAULT_IMAGE_ROOT", "/mock-images"
            ):
                dataset = get_image_answer_placeholder_questions("train")

        self.assertEqual(len(dataset), 1)
        row = dataset[0]
        instruction = f"{COT_PROMPT} {sample['question']}"
        self.assertEqual(row["task_type"], "text")
        self.assertEqual(
            row["prompt"],
            [{"role": "user", "content": f"<image>\n{instruction}"}],
        )
        self.assertEqual(row["instruction"], instruction)
        self.assertEqual(row["image"], "/mock-images/images/q1.png")
        self.assertEqual(row["answer_gt"], "42")
        self.assertNotIn("image_gen_enc", row)
        self.assertNotIn("image_gen", row)
        self.assertNotIn("image_gt", row)

    def test_loader_rejects_non_train_split(self):
        with self.assertRaisesRegex(ValueError, "Unsupported split 'validation'"):
            get_image_answer_placeholder_questions("validation")

    def test_loader_rejects_missing_jsonl_file(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            data_utils, "THINKMORPH_DEFAULT_DATA_ROOT", temp_dir
        ):
            with self.assertRaisesRegex(FileNotFoundError, "ThinkMorph jsonl file\\(s\\) not found"):
                get_image_answer_placeholder_questions("train")

    def test_loader_rejects_invalid_fields(self):
        cases = [
            ({"question": "", "problem_image_0": "images/q1.png", "answer": "42"}, "invalid question"),
            ({"question": "Q", "problem_image_0": "", "answer": "42"}, "invalid problem_image_0"),
            ({"question": "Q", "problem_image_0": "images/q1.png", "answer": "   "}, "invalid answer"),
        ]
        for sample, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                with tempfile.TemporaryDirectory() as temp_dir:
                    data_path = self._write_jsonl(
                        temp_dir, data_utils.THINKMORPH_LOCAL_JSONL_FILES[0], [sample]
                    )
                    self._write_empty_jsonl_files(temp_dir, exclude={data_path})

                    with patch.object(data_utils, "THINKMORPH_DEFAULT_DATA_ROOT", temp_dir), patch.object(
                        data_utils, "THINKMORPH_DEFAULT_IMAGE_ROOT", "/mock-images"
                    ):
                        with self.assertRaisesRegex(ValueError, expected_error):
                            get_image_answer_placeholder_questions("train")

    @staticmethod
    def _write_jsonl(root_dir: str, file_name: str, rows: list[dict]) -> str:
        path = f"{root_dir}/{file_name}"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return path

    @staticmethod
    def _write_empty_jsonl_files(root_dir: str, exclude: set[str]) -> None:
        for file_name in data_utils.THINKMORPH_LOCAL_JSONL_FILES:
            path = f"{root_dir}/{file_name}"
            if path in exclude:
                continue
            with open(path, "w", encoding="utf-8") as f:
                f.write("")


class ThinkMorphAnswerTrainingRegistrationTest(unittest.TestCase):
    def test_validate_reward_dataset_compatibility_accepts_answer_gt(self):
        dataset = Dataset.from_list(
            [
                {
                    "task_type": "text",
                    "prompt": [{"role": "user", "content": "<image>\nQuestion"}],
                    "image": "/tmp/example.png",
                    "answer_gt": "42",
                }
            ]
        )

        diffu_grpo_train._validate_reward_dataset_compatibility(
            "thinkmorph_answer",
            dataset,
            [soft_format_reward_func, strict_format_reward_func, correctness_reward_func],
        )

    def test_main_registers_thinkmorph_answer_dataset_and_rewards(self):
        dataset = Dataset.from_list(
            [
                {
                    "task_type": "text",
                    "prompt": [{"role": "user", "content": "<image>\nQuestion"}],
                    "image": "/tmp/example.png",
                    "answer_gt": "42",
                }
            ]
        )
        grpo_config = SimpleNamespace(
            dataset="thinkmorph_answer",
            seed=123,
            mm_tunable_parts="",
            save_steps=1,
            num_iterations=1,
        )
        model_config = SimpleNamespace()

        trainer_instance = self._trainer_stub()

        with patch.object(
            diffu_grpo_train, "get_image_answer_placeholder_questions", return_value=dataset
        ) as mock_loader, patch.object(
            diffu_grpo_train, "init_lavida_model_and_tokenizer", return_value=("model", "tokenizer")
        ), patch.object(diffu_grpo_train, "_set_trainable_parameters"), patch.object(
            diffu_grpo_train, "_log_trainable_ratio"
        ), patch.object(diffu_grpo_train, "_build_optimizer", return_value="optimizer"), patch.object(
            diffu_grpo_train, "DiffuGRPOTrainer", return_value=trainer_instance
        ) as mock_trainer:
            diffu_grpo_train.main(grpo_config=grpo_config, model_config=model_config)

        mock_loader.assert_called_once_with("train")
        trainer_kwargs = mock_trainer.call_args.kwargs
        self.assertEqual(
            trainer_kwargs["reward_funcs"],
            [soft_format_reward_func, strict_format_reward_func, correctness_reward_func],
        )
        self.assertIs(trainer_kwargs["train_dataset"], dataset)
        self.assertEqual(trainer_instance.train_call_count, 1)

    @staticmethod
    def _trainer_stub():
        class _TrainerStub:
            def __init__(self):
                self.train_call_count = 0

            def train(self):
                self.train_call_count += 1

        return _TrainerStub()


if __name__ == "__main__":
    unittest.main()
