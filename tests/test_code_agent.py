from code_agent import CodeTask, ExecutionFeedbackAgent, TestCase as CodeTestCase


TASK = CodeTask(
    task_id="maximum",
    prompt="Implement maximum(values), returning the largest integer.",
    entry_point="maximum",
    tests=(
        CodeTestCase(args=([1, 9, 3],), expected=9),
        CodeTestCase(args=([-5, -2, -8],), expected=-2),
    ),
)


class ScriptedGenerator:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return next(self.responses)


def test_agent_repairs_solution_from_hidden_test_feedback():
    generator = ScriptedGenerator(
        [
            "```python\ndef maximum(values):\n    return values[0]\n```",
            "```python\ndef maximum(values):\n    return max(values)\n```",
        ]
    )
    result = ExecutionFeedbackAgent(generator, max_attempts=3).run(TASK)

    assert result.success
    assert len(result.attempts) == 2
    assert result.attempts[0].verification.pass_rate == 0.0
    assert "hidden test failed" in generator.prompts[1]
    assert "expected 9" not in generator.prompts[1]
    assert result.final.reward.total == 1.0


def test_agent_stops_at_attempt_limit():
    generator = ScriptedGenerator(["def maximum(values):\n    return 0"] * 2)
    result = ExecutionFeedbackAgent(generator, max_attempts=2).run(TASK)

    assert not result.success
    assert len(result.attempts) == 2
