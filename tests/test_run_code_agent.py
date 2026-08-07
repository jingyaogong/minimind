import json

import pytest

from scripts.run_code_agent import build_output, read_existing_results, write_output


def test_agent_output_round_trip(tmp_path):
    output_path = tmp_path / "agent.json"
    results = [
        {"task_id": "task-a", "success": True, "attempt_count": 1, "attempts": []},
        {"task_id": "task-b", "success": False, "attempt_count": 3, "attempts": []},
    ]

    output = build_output(results)
    write_output(output_path, output)

    assert json.loads(output_path.read_text(encoding="utf-8")) == output
    assert read_existing_results(output_path) == results
    assert output["task_count"] == 2
    assert output["success_count"] == 1


def test_agent_resume_rejects_duplicate_task_ids(tmp_path):
    output_path = tmp_path / "agent.json"
    output_path.write_text(
        json.dumps(
            {
                "results": [
                    {"task_id": "task-a", "success": False},
                    {"task_id": "task-a", "success": True},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate or missing task_id"):
        read_existing_results(output_path)
