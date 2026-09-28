#!/usr/bin/env python
"""Reward-model load check: prove the 1.8B RM works *before* the RM arms.

The RM arms (``--reward_mode rm`` / ``mix``) are the only runs that need
``internlm2-1_8b-reward`` together with its ``trust_remote_code`` files.  A
missing dependency or a transformers-incompatible remote module would otherwise
surface minutes into a training run, buried in the training log.  This script
loads the model exactly the way ``trainer/train_grpo.py`` does (same wrapper
class, same default path) and answers three questions:

1. does it load, how long does it take, how much VRAM does it hold;
2. does a correct answer outscore a wrong one;
3. how long does one ``get_score`` call take -- that cost is paid 12x per step.

    python scripts/rm_load_check.py
    python scripts/rm_load_check.py --model-path /abs/path/to/internlm2-1_8b-reward
    python scripts/rm_load_check.py --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# trainer/train_grpo.py defaults to ``../../internlm2-1_8b-reward`` and is always
# started from ``trainer/``; resolve the same string from the same directory so a
# passing check implies the training command will find the model too.
TRAINER_RM_PATH = (REPO_ROOT / "trainer" / ".." / ".." / "internlm2-1_8b-reward").resolve()

PROMPT = "计算 12+7，最后一行以「答案是 数字」输出。"
CASES = (
    ("<think>\n12+7=19\n</think>\n\n答案是 19", "correct + well formatted"),
    ("答案是 999", "formatted but wrong"),
    ("我不知道", "neither format nor answer"),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-path",
        default=str(TRAINER_RM_PATH),
        help=f"默认等于 trainer/ 下解析出的路径：{TRAINER_RM_PATH}",
    )
    parser.add_argument("--device", default=None, help="默认 cuda:0（可用时），否则 cpu")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="训练脚本固定 float16",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    path = os.path.abspath(args.model_path)

    print(f"cwd          : {os.getcwd()}")
    print(f"trainer 解析 : {TRAINER_RM_PATH}")
    print(f"加载路径     : {path}")
    if not os.path.isdir(path):
        print(
            f"FAIL 目录不存在：{path}\n"
            "     确认下载位置，或用 --model-path 显式指定；训练脚本从 trainer/ 运行时"
            "会把 ../../internlm2-1_8b-reward 解析成上面那行。",
            file=sys.stderr,
        )
        return 2

    import torch
    import transformers

    try:
        import sentencepiece

        sp_version = sentencepiece.__version__
    except ImportError:
        sp_version = None
    print(
        f"环境         : torch {torch.__version__} | transformers {transformers.__version__}"
        f" | sentencepiece {sp_version or '缺失'}"
    )
    if sp_version is None:
        print("FAIL 缺 sentencepiece：pip install sentencepiece==0.2.0", file=sys.stderr)
        return 2

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    from trainer.trainer_utils import LMForRewardModel

    t0 = time.time()
    reward_model = LMForRewardModel(path, device=device, dtype=getattr(torch, args.dtype))
    load_seconds = time.time() - t0
    vram = torch.cuda.memory_allocated() / 2**30 if device.startswith("cuda") else 0.0
    print(f"加载完成     : {load_seconds:.1f}s | 该进程显存 {vram:.2f} GB | device={device}")

    messages = [{"role": "user", "content": PROMPT}]
    scores = []
    for response, label in CASES:
        t1 = time.time()
        score = float(reward_model.get_score(messages, response))
        scores.append(score)
        print(f"  score={score:+.3f}  ({time.time() - t1:.2f}s)  {label}")

    passed = scores[0] > scores[1] and scores[0] > scores[2]
    verdict = (
        "OK 正确回答得分最高，RM 可用"
        if passed
        else "FAIL 正确回答没有高于错误回答，RM 可能没被正确加载（或权重不匹配）"
    )
    print(f"{verdict}：correct={scores[0]:+.3f} wrong={scores[1]:+.3f} no_answer={scores[2]:+.3f}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
