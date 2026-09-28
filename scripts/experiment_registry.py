#!/usr/bin/env python
"""Read, validate and turn the pre-registered experiment table into commands.

``configs/experiment_registry.csv`` is written *before* any formal run
(SOP §8.4).  A registry that is only prose drifts away from what was actually
executed, so this script derives the training command from the registered row
and cross-checks every mapped column against the real ``train_grpo.py``
argparse flags.  A typo like ``alpha`` vs ``--mix_alpha`` therefore fails
loudly instead of silently producing a run that no longer matches the registry.

    python scripts/experiment_registry.py --check
    python scripts/experiment_registry.py --list
    python scripts/experiment_registry.py --print-command mini_rule_seed42
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "experiment_registry.csv"
DEFAULT_TRAINER = REPO_ROOT / "trainer" / "train_grpo.py"

# The SOP §8.4 header.  Order matters: it is the contract of the file.
COLUMNS = [
    "run_tag", "model", "reward_mode", "loss_type", "seed", "G", "max_gen_len",
    "beta", "dynamic_sampling", "len_penalty_coef", "alpha", "beta_mix", "hypothesis",
]

# Registry column -> train_grpo.py flag.  Everything not listed here is either
# informational (run_tag/model/hypothesis) or handled explicitly below.
COLUMN_TO_FLAG = {
    "reward_mode": "--reward_mode",
    "loss_type": "--loss_type",
    "seed": "--seed",
    "G": "--num_generations",
    "max_gen_len": "--max_gen_len",
    "beta": "--beta",
    "dynamic_sampling": "--dynamic_sampling",
    "len_penalty_coef": "--len_penalty_coef",
    "alpha": "--mix_alpha",
    "beta_mix": "--mix_beta",
}

# Held fixed across every arm: the SOP requires the arms to differ in exactly
# one factor, so anything not registered here must not drift between runs.
FIXED_ARGS: dict[str, Any] = {
    "batch_size": 2,
    "learning_rate": 3e-7,
    "max_seq_len": 512,
    "epochs": 1,
    "num_workers": 0,
    "log_interval": 5,
    "save_interval": 50,
    "temperature": 0.8,
    "thinking_ratio": 1.0,
    "verdict_tier": "binary",
    "reward_marker": "答案是",
    "device": "cuda:0",
}
DEFAULT_FROM_WEIGHT = "math_cold_cot_seed42"
DEFAULT_RLVR_DATA = "../dataset/math_rlvr_train.jsonl"
DEFAULT_RM_PATH = "../../internlm2-1_8b-reward"

REWARD_MODES = ("rule", "rm", "mix", "official")
LOSS_TYPES = ("grpo", "cispo")
RUN_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
FLAG_RE = re.compile(r"""add_argument\(\s*["'](--[A-Za-z0-9_]+)["']""")


# --------------------------------------------------------------------------- #
# loading / validation
# --------------------------------------------------------------------------- #
def load_registry(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return header, rows


def trainer_flags(path: Path) -> set[str]:
    return set(FLAG_RE.findall(path.read_text(encoding="utf-8")))


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _as_int(row: dict[str, str], column: str, problems: list[str]) -> int | None:
    try:
        return int(str(row.get(column, "")).strip())
    except ValueError:
        problems.append(f"{row.get('run_tag')}: {column}={row.get(column)!r} is not an integer")
        return None


def _as_float(row: dict[str, str], column: str, problems: list[str]) -> float | None:
    try:
        return float(str(row.get(column, "")).strip())
    except ValueError:
        problems.append(f"{row.get('run_tag')}: {column}={row.get(column)!r} is not a number")
        return None


def check_registry(
    header: list[str],
    rows: list[dict[str, str]],
    flags: set[str],
    warnings: list[str] | None = None,
) -> list[str]:
    """Return a list of problems; an empty list means the registry is usable."""

    problems: list[str] = []
    warnings = warnings if warnings is not None else []

    if header != COLUMNS:
        problems.append(f"header mismatch:\n  expected {COLUMNS}\n  found    {header}")
    if not rows:
        problems.append("registry has no rows")

    for column, flag in COLUMN_TO_FLAG.items():
        if flag not in flags:
            problems.append(f"column {column!r} maps to {flag}, which train_grpo.py does not define")

    seen: dict[str, int] = {}
    for index, row in enumerate(rows, start=2):  # row 1 is the header
        tag = (row.get("run_tag") or "").strip()
        where = tag or f"<row {index}>"
        if not tag:
            problems.append(f"row {index}: empty run_tag")
        elif not RUN_TAG_RE.match(tag):
            problems.append(f"{where}: run_tag must match {RUN_TAG_RE.pattern}")
        elif tag in seen:
            problems.append(
                f"{where}: duplicated run_tag (also on row {seen[tag]}); MiniMind reuses "
                "checkpoint files by prefix, so a duplicate silently continues the wrong run"
            )
        else:
            seen[tag] = index

        if not (row.get("model") or "").strip():
            problems.append(f"{where}: empty model")
        if not (row.get("hypothesis") or "").strip():
            problems.append(f"{where}: empty hypothesis (a run without a preregistered claim is not an experiment)")

        mode = (row.get("reward_mode") or "").strip()
        if mode not in REWARD_MODES:
            problems.append(f"{where}: reward_mode {mode!r} not in {list(REWARD_MODES)}")
        if (row.get("loss_type") or "").strip() not in LOSS_TYPES:
            problems.append(f"{where}: loss_type {row.get('loss_type')!r} not in {list(LOSS_TYPES)}")

        seed = _as_int(row, "seed", problems)
        generations = _as_int(row, "G", problems)
        max_gen_len = _as_int(row, "max_gen_len", problems)
        beta = _as_float(row, "beta", problems)
        penalty = _as_float(row, "len_penalty_coef", problems)
        alpha = _as_float(row, "alpha", problems)
        beta_mix = _as_float(row, "beta_mix", problems)

        dynamic = (row.get("dynamic_sampling") or "").strip()
        if dynamic not in ("0", "1"):
            problems.append(f"{where}: dynamic_sampling {dynamic!r} must be 0 or 1")
        if seed is not None and seed < 0:
            problems.append(f"{where}: seed must be >= 0")
        if generations is not None and generations < 2:
            problems.append(
                f"{where}: G={generations}; group normalisation needs G >= 2, otherwise every "
                "advantage is zero by construction"
            )
        if max_gen_len is not None and max_gen_len <= 0:
            problems.append(f"{where}: max_gen_len must be positive")
        if beta is not None and beta <= 0:
            problems.append(f"{where}: beta={beta}; the SOP forbids a run with the KL term disabled")
        if penalty is not None and penalty < 0:
            problems.append(f"{where}: len_penalty_coef must be >= 0")
        if alpha is not None and alpha < 0 or beta_mix is not None and beta_mix < 0:
            problems.append(f"{where}: mix weights must be >= 0")
        if dynamic == "1" and mode not in ("rule", "rm", "mix"):
            problems.append(f"{where}: dynamic_sampling needs per-sample correctness, not available in {mode!r}")
        if penalty and float(penalty) > 0 and dynamic == "1":
            warnings.append(
                f"{where}: dynamic_sampling and len_penalty_coef are both on; the SOP asks for one "
                "factor per run so the contribution stays attributable"
            )
    return problems


# --------------------------------------------------------------------------- #
# reference arm
# --------------------------------------------------------------------------- #
def is_baseline_row(row: dict[str, str]) -> bool:
    """The reference arm: GT-rule reward, no degenerate-group filter, no length term.

    configs/README.md declares this combination the baseline every other arm is
    compared against, so the predicate lives next to the registry contract
    rather than in the reporting scripts.
    """

    return (
        (row.get("reward_mode") or "").strip() == "rule"
        and (row.get("dynamic_sampling") or "").strip() == "0"
        and _to_float(row.get("len_penalty_coef")) == 0.0
    )


def baseline_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    """The first baseline-signature row in file order, or None."""

    for row in rows:
        if is_baseline_row(row):
            return row
    return None


# --------------------------------------------------------------------------- #
# command rendering
# --------------------------------------------------------------------------- #
def build_args(
    row: dict[str, str],
    *,
    from_weight: str = DEFAULT_FROM_WEIGHT,
    rlvr_data_path: str = DEFAULT_RLVR_DATA,
    reward_model_path: str = DEFAULT_RM_PATH,
) -> list[str]:
    """Derive the exact ``train_grpo.py`` argv for one registered row."""

    tag = row["run_tag"].strip()
    mode = row["reward_mode"].strip()
    args = [
        "--reward_mode", mode,
        "--rlvr_data_path", rlvr_data_path,
        "--from_weight", from_weight,
        "--save_weight", tag,
        "--run_tag", tag,
    ]
    if mode in ("rm", "mix", "official"):
        args += ["--reward_model_path", reward_model_path]
    args += [
        "--loss_type", row["loss_type"].strip(),
        "--seed", str(row["seed"]).strip(),
        "--temperature", str(FIXED_ARGS["temperature"]),
        "--batch_size", str(FIXED_ARGS["batch_size"]),
        "--num_generations", str(row["G"]).strip(),
        "--learning_rate", str(FIXED_ARGS["learning_rate"]),
        "--beta", str(row["beta"]).strip(),
        "--max_seq_len", str(FIXED_ARGS["max_seq_len"]),
        "--max_gen_len", str(row["max_gen_len"]).strip(),
        "--epochs", str(FIXED_ARGS["epochs"]),
        "--num_workers", str(FIXED_ARGS["num_workers"]),
        "--log_interval", str(FIXED_ARGS["log_interval"]),
        "--save_interval", str(FIXED_ARGS["save_interval"]),
        "--thinking_ratio", str(FIXED_ARGS["thinking_ratio"]),
        "--verdict_tier", str(FIXED_ARGS["verdict_tier"]),
        "--reward_marker", str(FIXED_ARGS["reward_marker"]),
        "--dynamic_sampling", str(row["dynamic_sampling"]).strip(),
        "--len_penalty_coef", str(row["len_penalty_coef"]).strip(),
        "--mix_alpha", str(row["alpha"]).strip(),
        "--mix_beta", str(row["beta_mix"]).strip(),
        "--device", str(FIXED_ARGS["device"]),
    ]
    return args


def shell_quote(value: str) -> str:
    return value if re.fullmatch(r"[A-Za-z0-9_./=:+-]*", value) else f"'{value}'"


def render_command(tag: str, args: list[str], *, archival: bool = True) -> str:
    """Render a copy-pasteable block; the archival form also captures the log."""

    # -u matters: the archival form pipes stdout into tee, and a buffered python
    # process leaves the log file empty for minutes at a time.
    lines = ["python -u train_grpo.py"]
    for index in range(0, len(args), 2):
        lines.append(f"  {args[index]} {shell_quote(args[index + 1])}")
    # Every line needs a trailing backslash, otherwise the shell runs each flag
    # as its own command.
    body = " \\\n".join(lines)
    if not archival:
        return body + "\n"
    # Path note: the block is meant to run with cwd = repo root.  The tee target is
    # written relative to ``trainer/`` (because we cd there), so mkdir must use the
    # repo-root-relative path -- ``../runs`` would create a directory one level above
    # the repo and leave tee writing into a directory that does not exist.
    return (
        "set -o pipefail\n"
        f"mkdir -p runs/{tag}\n"
        "cd trainer\n"
        + body
        + f" \\\n  2>&1 | tee ../runs/{tag}/train.log\n"
    )


def save_command(tag: str, args: list[str], *, run_dir: Path) -> Path:
    target = run_dir / tag
    target.mkdir(parents=True, exist_ok=True)
    path = target / "command.txt"
    path.write_text("cd trainer\n" + render_command(tag, args, archival=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the preregistered experiment table and render its commands.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--trainer", type=Path, default=DEFAULT_TRAINER)
    parser.add_argument("--check", action="store_true", help="validate the table and exit")
    parser.add_argument("--list", action="store_true", help="print one line per registered run")
    parser.add_argument("--print-command", metavar="RUN_TAG", action="append", default=None,
                        help="render the training command for this run (repeatable)")
    parser.add_argument("--save-command", metavar="RUN_TAG", action="append", default=None,
                        help="write runs/<RUN_TAG>/command.txt")
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--from-weight", type=str, default=DEFAULT_FROM_WEIGHT)
    parser.add_argument("--rlvr-data-path", type=str, default=DEFAULT_RLVR_DATA)
    parser.add_argument("--reward-model-path", type=str, default=DEFAULT_RM_PATH)
    parser.add_argument("--plain", action="store_true", help="print only the python command")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.registry.is_file():
        print(f"error: registry not found: {args.registry}", file=sys.stderr)
        return 2

    header, rows = load_registry(args.registry)
    flags = trainer_flags(args.trainer)
    warnings: list[str] = []
    problems = check_registry(header, rows, flags, warnings)
    for warning in warnings:
        print(f"WARN {warning}")
    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        return 1

    by_tag = {row["run_tag"].strip(): row for row in rows}
    did_something = False

    if args.list or not (args.check or args.print_command or args.save_command):
        for row in rows:
            print(
                f"{row['run_tag']:<28} {row['reward_mode']:<5} seed={row['seed']:<3} "
                f"G={row['G']:<2} gen={row['max_gen_len']:<4} dyn={row['dynamic_sampling']} "
                f"len={row['len_penalty_coef']:<5} {row['hypothesis'][:60]}"
            )
        did_something = True

    for tag in args.print_command or []:
        if tag not in by_tag:
            print(f"error: unknown run_tag {tag!r}; known: {sorted(by_tag)}", file=sys.stderr)
            return 2
        argv_out = build_args(
            by_tag[tag],
            from_weight=args.from_weight,
            rlvr_data_path=args.rlvr_data_path,
            reward_model_path=args.reward_model_path,
        )
        print(render_command(tag, argv_out, archival=not args.plain))
        did_something = True

    for tag in args.save_command or []:
        if tag not in by_tag:
            print(f"error: unknown run_tag {tag!r}; known: {sorted(by_tag)}", file=sys.stderr)
            return 2
        argv_out = build_args(
            by_tag[tag],
            from_weight=args.from_weight,
            rlvr_data_path=args.rlvr_data_path,
            reward_model_path=args.reward_model_path,
        )
        path = save_command(tag, argv_out, run_dir=args.run_dir)
        print(f"wrote {path}")
        did_something = True

    if args.check and not did_something:
        print(f"PASS {len(rows)} registered runs are consistent with {args.trainer.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
