import pytest

from code_agent import CodeTask, TestCase as CodeTestCase, score_code_batch


TASK = CodeTask(
    task_id="identity",
    prompt="Implement identity(x).",
    entry_point="identity",
    tests=(CodeTestCase(args=(7,), expected=7),),
)


def test_scores_task_major_grouped_responses():
    rewards = score_code_batch(
        [TASK],
        ["def identity(x):\n    return x", "def identity(x):\n    return 0"],
        num_generations=2,
    )
    assert [reward.total for reward in rewards] == [1.0, 0.0]


def test_rejects_misaligned_group_size():
    with pytest.raises(ValueError, match="expected 2 responses"):
        score_code_batch([TASK], ["def identity(x): return x"], num_generations=2)
