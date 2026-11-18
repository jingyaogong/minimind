"""CPU regressions with the real tokenizer, tools and tiny MiniMind models."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.agent_opd_utils import (
    AgentTrajectory, collect_trajectory, distillation_loss, pack_trajectories,
)
from trainer.train_agent import TOOLS
from trainer import train_agent_opd as trainer

ROOT = Path(__file__).resolve().parents[1]
MATH_TOOLS = [TOOLS[0]]
MESSAGES = [{"role": "user", "content": "请用工具计算 2+2。"}]
CALL = '<tool_call>\n{"name":"calculate_math","arguments":{"expression":"2+2"}}\n</tool_call>'


class ScriptedStudent(torch.nn.Module):
    """Deterministic generation at the model boundary, keeping real tool/token IO."""

    def __init__(self, tokenizer, responses):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.responses = responses
        self.tokenizer = tokenizer
        self.inputs, self.options = [], []

    def generate(self, input_ids, **kwargs):
        self.inputs.append(input_ids[0].tolist())
        self.options.append(kwargs)
        response = self.responses[len(self.inputs) - 1]
        ids = self.tokenizer(response, add_special_tokens=False)["input_ids"] if isinstance(response, str) else response
        ids = ids[:kwargs["max_new_tokens"]]
        return torch.cat((input_ids, input_ids.new_tensor([ids])), dim=1)


class TrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
        cls.eos = cls.tokenizer.eos_token

    def sample(self, responses, **kwargs):
        student = ScriptedStudent(self.tokenizer, responses)
        trajectory = collect_trajectory(student, self.tokenizer, MESSAGES, MATH_TOOLS, **kwargs)
        return student, trajectory

    def test_tool_result_conditions_next_turn_but_is_not_a_target(self):
        student, trajectory = self.sample([CALL + self.eos, "结果是 4。" + self.eos])
        self.assertEqual(trajectory.turns, 2)
        self.assertEqual(trajectory.tool_calls, 1)
        self.assertFalse(trajectory.truncated)
        self.assertIn('"result": "4"', self.tokenizer.decode(student.inputs[1]))
        assistant_ids = [t for t, keep in zip(trajectory.input_ids, trajectory.assistant_mask) if keep]
        expected = self.tokenizer(CALL + self.eos, add_special_tokens=False)["input_ids"]
        expected += self.tokenizer("结果是 4。" + self.eos, add_special_tokens=False)["input_ids"]
        self.assertEqual(assistant_ids, expected)
        self.assertEqual(assistant_ids.count(self.tokenizer.eos_token_id), 2)
        self.assertTrue(student.training)
        for options in student.options:
            self.assertEqual((options["top_p"], options["top_k"], options["temperature"]), (1.0, 0, 1.0))
        self.assertEqual(MESSAGES, [{"role": "user", "content": "请用工具计算 2+2。"}])

    def test_preserves_exact_sampled_ids_and_first_eos_only(self):
        sampled = self.tokenizer(CALL, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id] * 3
        student, trajectory = self.sample([sampled, [self.tokenizer.eos_token_id]])
        start = len(student.inputs[0])
        self.assertEqual(trajectory.input_ids[start:start + len(sampled) - 2], sampled[:-2])
        self.assertEqual(sum(trajectory.assistant_mask), len(sampled) - 1)

    def test_multiple_tool_calls_return_separate_observations(self):
        second = CALL.replace("2+2", "3*3")
        student, trajectory = self.sample([CALL + "\n" + second + self.eos, "4 和 9。" + self.eos])
        context = self.tokenizer.decode(student.inputs[1])
        self.assertIn('"result": "4"', context)
        self.assertIn('"result": "9"', context)
        self.assertEqual(trajectory.tool_calls, 2)

    def test_unknown_tool_and_malformed_arguments_become_observations(self):
        for call in [CALL.replace("calculate_math", "get_current_weather"),
                     '<tool_call>{"name":[],"arguments":null}</tool_call>',
                     '<tool_call>{"name":"calculate_math","arguments":"invalid"}</tool_call>']:
            with self.subTest(call=call):
                student, trajectory = self.sample([call + self.eos, "无法计算。" + self.eos])
                self.assertIn("unknown tool or invalid arguments", self.tokenizer.decode(student.inputs[1]))
                self.assertEqual(trajectory.turns, 2)

    def test_non_dictionary_tool_json_does_not_crash(self):
        _, trajectory = self.sample(['<tool_call>[]</tool_call>' + self.eos])
        self.assertEqual(trajectory.turns, 1)

    def test_turn_limit_keeps_generated_tokens_without_executing_more_tools(self):
        with patch("trainer.agent_opd_utils.execute_tool") as execute:
            _, trajectory = self.sample([CALL + self.eos], max_turns=1)
        self.assertTrue(trajectory.truncated)
        execute.assert_not_called()
        self.assertTrue(trajectory.assistant_mask[-1])

    def test_context_limit_never_slices_an_observation_into_a_training_target(self):
        probe = ScriptedStudent(self.tokenizer, [CALL + self.eos])
        prompt = self.tokenizer.apply_chat_template(MESSAGES, tools=MATH_TOOLS, tokenize=False,
                                                    add_generation_prompt=True, open_thinking=False)
        limit = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"]) + 8
        trajectory = collect_trajectory(probe, self.tokenizer, MESSAGES, MATH_TOOLS, max_total_len=limit)
        self.assertEqual(len(trajectory.input_ids), limit)
        self.assertTrue(trajectory.truncated)
        with self.assertRaisesRegex(ValueError, "Prompt leaves no generation space"):
            collect_trajectory(probe, self.tokenizer, MESSAGES, MATH_TOOLS, max_total_len=1)

    def test_empty_final_answer_still_trains_eos(self):
        _, trajectory = self.sample([[self.tokenizer.eos_token_id]])
        self.assertEqual(sum(trajectory.assistant_mask), 1)
        self.assertFalse(trajectory.truncated)

    def test_real_model_can_generate_an_on_policy_trajectory(self):
        torch.manual_seed(3)
        student = MiniMindForCausalLM(MiniMindConfig(hidden_size=32, num_hidden_layers=1))
        trajectory = collect_trajectory(student, self.tokenizer, MESSAGES, max_new_tokens=2)
        self.assertGreaterEqual(sum(trajectory.assistant_mask), 1)
        self.assertLessEqual(sum(trajectory.assistant_mask), 2)


class LossTests(unittest.TestCase):
    def test_shifted_masks_include_both_turns_and_exclude_observations_and_padding(self):
        trajectories = [AgentTrajectory([1, 10, 2, 30, 31, 11, 2], [0, 1, 1, 0, 0, 1, 1], 2, 1, False),
                        AgentTrajectory([1, 12, 2], [0, 1, 1], 1, 0, False)]
        ids, attention, targets = pack_trajectories(trajectories, 2, "cpu")
        self.assertEqual(targets.tolist(), [[True, True, False, False, True, True],
                                          [True, True, False, False, False, False]])
        self.assertEqual(attention[1].tolist(), [1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(ids[1, -1].item(), 2)

    def test_kl_directions_match_independent_probability_calculation(self):
        student = torch.tensor([[[0.2, -0.4, 1.0]]], requires_grad=True)
        teacher = torch.tensor([[[0.8, -0.2, 0.1]]], requires_grad=True)
        for direction in ("forward_kl", "reverse_kl"):
            with self.subTest(direction=direction):
                temperature = 1.7
                ps, pt = (student.detach() / temperature).softmax(-1), (teacher.detach() / temperature).softmax(-1)
                p, q = (pt, ps) if direction == "forward_kl" else (ps, pt)
                expected = (p * (p.log() - q.log())).sum() * temperature ** 2
                loss = distillation_loss(student, teacher, torch.ones(1, 1, dtype=torch.bool), temperature, direction)
                torch.testing.assert_close(loss, expected)
                loss.backward()
                self.assertIsNone(teacher.grad)
                self.assertGreater(student.grad.abs().sum().item(), 0)
                student.grad = None

    def test_masked_tokens_have_no_direct_gradient_even_with_invalid_logits(self):
        student = torch.randn(1, 5, 4, requires_grad=True)
        teacher = torch.randn(1, 5, 4, requires_grad=True)
        mask = torch.tensor([[True, False, False, True, False]])
        with torch.no_grad():
            student[:, 1] = float("nan")
            teacher[:, 1] = float("nan")
        loss = distillation_loss(student, teacher, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(student.grad[~mask].abs().sum().item(), 0)
        self.assertIsNone(teacher.grad)

    def test_empty_mask_is_finite_differentiable_zero(self):
        student = torch.randn(2, 3, 5, requires_grad=True)
        loss = distillation_loss(student, student.detach(), torch.zeros(2, 3, dtype=torch.bool))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(student.grad.abs().sum().item(), 0)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.student_path, self.teacher_path = self.directory / "student.pth", self.directory / "teacher.pth"
        config = MiniMindConfig(hidden_size=32, num_hidden_layers=1)
        torch.manual_seed(1)
        torch.save(MiniMindForCausalLM(config).state_dict(), self.student_path)
        torch.manual_seed(2)
        torch.save(MiniMindForCausalLM(config).state_dict(), self.teacher_path)
        self.data = self.directory / "prompts.jsonl"
        self.data.write_text("\n".join(json.dumps({"messages": MESSAGES}) for _ in range(3)) + "\n")

    def args(self, name="run", extra=()):
        return trainer.parse_args([
            "--student_checkpoint", str(self.student_path), "--teacher_checkpoint", str(self.teacher_path),
            "--data_path", str(self.data), "--output_dir", str(self.directory / name),
            "--student_hidden_size", "32", "--student_num_layers", "1",
            "--teacher_hidden_size", "32", "--teacher_num_layers", "1",
            "--device", "cpu", "--max_gen_len", "2", "--learning_rate", "0.001",
            *extra,
        ])

    def run_train(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return trainer.train(args)

    def checkpoint(self, name):
        return torch.load(self.directory / name / "agent_opd_32_resume.pth", weights_only=True)

    def test_actual_training_updates_student_only_and_saves_residual_window(self):
        loaded = []
        original = trainer.load_model

        def track(*args):
            model = original(*args)
            loaded.append(model)
            return model

        args = self.args(extra=["--accumulation_steps", "2"])
        with patch.object(trainer, "load_model", side_effect=track):
            metrics = self.run_train(args)
        self.assertEqual(len(metrics), 2)  # 3 microbatches, including the final partial window.
        student, teacher = loaded
        initial = torch.load(self.student_path, weights_only=True)
        self.assertTrue(any(not torch.equal(initial[k], v) for k, v in student.state_dict().items()))
        teacher_weights = torch.load(self.teacher_path, weights_only=True)
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, teacher_weights[key], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))
        saved = self.checkpoint("run")
        self.assertEqual((saved["epoch"], saved["batch"], saved["step"]), (1, 0, 2))
        for key, value in student.state_dict().items():
            torch.testing.assert_close(saved["model"][key], value, rtol=0, atol=0)

    def test_resumed_online_sampling_matches_uninterrupted_training(self):
        self.run_train(self.args("full", ["--max_steps", "2"]))
        self.run_train(self.args("resume", ["--max_steps", "1"]))
        path = self.directory / "resume/agent_opd_32_resume.pth"
        self.run_train(self.args("resume", ["--max_steps", "2", "--resume", str(path)]))
        full, resumed = self.checkpoint("full"), self.checkpoint("resume")
        self.assertEqual((full["epoch"], full["batch"], full["step"]),
                         (resumed["epoch"], resumed["batch"], resumed["step"]))
        for key in full["model"]:
            torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0, atol=0)

    def test_resume_rejects_a_different_teacher_or_sampling_setup(self):
        self.run_train(self.args(extra=["--max_steps", "1"]))
        resume = str(self.directory / "run/agent_opd_32_resume.pth")
        with self.assertRaisesRegex(ValueError, "same data, teacher, sampling"):
            self.run_train(self.args(extra=["--resume", resume, "--temperature", "0.8"]))

    def test_agent_dataset_needs_no_ground_truth_answer(self):
        self.data.write_text(json.dumps({"conversations": [
            {"role": "system", "content": "Use tools", "tools": json.dumps(MATH_TOOLS)},
            *MESSAGES, {"role": "assistant", "content": "ignored fixed answer"},
        ]}) + "\n")
        messages, tools = trainer.load_prompts(self.data)[0]
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(tools, MATH_TOOLS)

    def test_multi_turn_training_reduces_teacher_divergence(self):
        # Control generation only; both models, KL, backprop and optimizer are real.
        tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
        self.data.write_text(json.dumps({"messages": MESSAGES, "tools": MATH_TOOLS}) + "\n")

        def generate(model, input_ids, **kwargs):
            has_result = '"result": "4"' in tokenizer.decode(input_ids[0])
            text = ("结果是 4。" if has_result else CALL) + tokenizer.eos_token
            new_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            return torch.cat((input_ids, input_ids.new_tensor([new_ids])), dim=1)

        args = self.args(extra=["--epochs", "12", "--max_gen_len", "128"])
        with patch.object(MiniMindForCausalLM, "generate", new=generate):
            metrics = self.run_train(args)
        self.assertTrue(all(record["mean_turns"] == 2 and record["tool_calls"] == 1 for record in metrics))
        self.assertLess(metrics[-1]["loss"], metrics[0]["loss"] * 0.5)

    def test_invalid_training_parameters_fail_before_loading_models(self):
        for flag, value in [("--grad_clip", "0"), ("--learning_rate", "nan"), ("--max_steps", "-1")]:
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                self.run_train(self.args(extra=[flag, value]))


if __name__ == "__main__":
    unittest.main()
