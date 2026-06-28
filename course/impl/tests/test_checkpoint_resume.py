"""Alignment tests for lesson 15 checkpoint/resume mechanics.

Run after implementing:

    python course/impl/tests/test_checkpoint_resume.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from course.impl.core.train_loop import (  # noqa: E402
    CourseSkipBatchSampler,
    checkpoint_paths,
    load_course_checkpoint,
    save_course_checkpoint,
)


def test_checkpoint_paths() -> None:
    ckp_path, resume_path = checkpoint_paths("checkpoints", "pretrain", 768, use_moe=False)
    print(f"ckp_path={ckp_path}")
    print(f"resume_path={resume_path}")
    assert ckp_path == Path("checkpoints/pretrain_768.pth")
    assert resume_path == Path("checkpoints/pretrain_768_resume.pth")

    moe_ckp_path, moe_resume_path = checkpoint_paths("checkpoints", "pretrain", 768, use_moe=True)
    assert moe_ckp_path == Path("checkpoints/pretrain_768_moe.pth")
    assert moe_resume_path == Path("checkpoints/pretrain_768_moe_resume.pth")


def test_skip_batch_sampler() -> None:
    sampler = CourseSkipBatchSampler(range(10), batch_size=3, skip_batches=2)
    batches = list(sampler)
    print(f"skipped_batches={batches}")
    assert batches == [[6, 7, 8], [9]]
    assert len(sampler) == 2


def test_save_and_load_checkpoint() -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckp_path, resume_path = save_course_checkpoint(
            model,
            optimizer,
            save_dir=tmpdir,
            weight="course_pretrain",
            hidden_size=64,
            use_moe=False,
            epoch=2,
            step=7,
            scaler=scaler,
            extra_state={"note": "unit-test"},
        )
        print(f"saved_ckp_exists={ckp_path.exists()}")
        print(f"saved_resume_exists={resume_path.exists()}")
        assert ckp_path.exists()
        assert resume_path.exists()

        loaded = load_course_checkpoint(tmpdir, "course_pretrain", 64, use_moe=False)
        assert loaded is not None
        assert loaded["epoch"] == 2
        assert loaded["step"] == 7
        assert loaded["note"] == "unit-test"
        assert "model" in loaded
        assert "optimizer" in loaded
        assert "scaler" in loaded


def main() -> None:
    try:
        test_checkpoint_paths()
        test_skip_batch_sampler()
        test_save_and_load_checkpoint()
    except NotImplementedError as exc:
        raise SystemExit(f"TODO not implemented yet: {exc}") from exc
    print("checkpoint_resume=passed")


if __name__ == "__main__":
    main()
