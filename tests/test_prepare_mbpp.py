import pytest

from scripts.prepare_mbpp import build_sft_record, convert_record, convert_split, parse_assertion


def test_parse_mbpp_equality_and_boolean_assertions():
    assert parse_assertion("assert add(1, 2) == 3") == (
        "add",
        {"args": [1, 2], "kwargs": {}, "expected": 3},
    )
    assert parse_assertion("assert is_even(4)")[-1]["expected"] is True
    assert parse_assertion("assert not is_even(3)")[-1]["expected"] is False


def test_convert_record_validates_reference_solution():
    record = {
        "task_id": 1,
        "prompt": "Write a function that adds two integers.",
        "code": "def add(a, b):\n    return a + b",
        "test_list": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
    }

    converted = convert_record(record)

    assert converted["task_id"] == "mbpp-1"
    assert converted["entry_point"] == "add"
    assert len(converted["tests"]) == 2

    sft = build_sft_record(record, converted)
    assert "Required entry point: add" in sft["conversations"][0]["content"]
    assert "def add(a, b)" in sft["conversations"][1]["content"]


def test_convert_split_keeps_tasks_and_sft_records_aligned():
    record = {
        "task_id": 2,
        "prompt": "Return the input integer.",
        "code": "def identity(value):\n    return value",
        "test_list": ["assert identity(7) == 7"],
    }

    tasks, sft_records, skipped = convert_split([record])

    assert len(tasks) == len(sft_records) == 1
    assert not skipped


def test_parse_mbpp_rejects_nonliteral_expectation():
    with pytest.raises(ValueError):
        parse_assertion("assert add(1, 2) == expected")
