#!/usr/bin/env python
"""Assemble the SOP section 13 archive for one pre-registered run.

The registry decides what *should* have run, the trainer writes the raw numbers,
and this script copies both into ``runs/{run_tag}/`` together with the
environment and hash evidence.  Two rules keep the archive honest:

* raw evidence (``metrics.csv``, ``samples.jsonl``, ``eval.json``) is copied
  byte for byte and never rewritten, so an archived number is always the number
  the run produced;
* anything the archive cannot find is written into ``manifest.json`` as a
  missing item instead of being silently omitted, so a half-finished run is
  visibly half-finished rather than looking complete.

    python scripts/archive_run.py --run-tag mini_rule_seed42
    python scripts/archive_run.py --all
    python scripts/archive_run.py --run-tag mini_rule_seed42 --check
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment_registry import (  # noqa: E402
    DEFAULT_REGISTRY,
    baseline_row,
    build_args,
    check_registry,
    load_registry,
    render_command,
    trainer_flags,
)

REQUIRED_FILES = [
    "command.txt",
    "config.yaml",
    "environment.txt",
    "data_manifest.json",
    "metrics.csv",
    "eval.json",
    "samples.jsonl",
    "analysis.md",
    "checkpoints.txt",
]

# A run that is not in the registry is a *prerequisite* run (e.g. the SFT cold start
# that produces the weights the registered arms start from).  MiniMind's upstream
# scripts do not emit a metrics CSV, a samples JSONL or a run config, so those three
# files cannot exist and must not be reported as evidence gaps -- otherwise
# ``--check --strict`` fails a run that is behaving exactly as documented.
# ``status``/``missing`` still use the full list, so the archive keeps saying
# "[partial]" and never overstates the evidence.
PREREQUISITE_REQUIRED_FILES = [
    name for name in REQUIRED_FILES if name not in {"config.yaml", "metrics.csv", "samples.jsonl"}
]

# Copied verbatim from the run; never regenerated.
RAW_EVIDENCE = ("metrics.csv", "samples.jsonl", "eval.json", "eval.jsonl")

# Strings the trainer interpolates into the archive before the first real
# metric exists.  A checkpoint tag that still carries one of these was never
# trained by a human-visible command.
PLACEHOLDER_TAGS = {"unnamed"}

EVAL_REQUIRED_KEYS = [
    "model",
    "seed",
    "temperature",
    "greedy_accuracy",
    "sampled_pass_at_1",
    "format_rate",
    "by_difficulty",
]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_metrics(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


# --------------------------------------------------------------------------- #
# registry context
# --------------------------------------------------------------------------- #
def registry_row(registry_path: Path, run_tag: str) -> dict[str, str] | None:
    _header, rows = load_registry(registry_path)
    for row in rows:
        if (row.get("run_tag") or "").strip() == run_tag:
            return row
    return None


def changed_columns(row: dict[str, str], reference: dict[str, str] | None) -> list[tuple[str, str, str]]:
    """Columns where this arm differs from the reference arm."""

    if reference is None:
        return []
    skip = {"run_tag", "hypothesis"}
    out: list[tuple[str, str, str]] = []
    for column in reference:
        if column in skip:
            continue
        left = (reference.get(column) or "").strip()
        right = (row.get(column) or "").strip()
        if left != right:
            out.append((column, left, right))
    return out


def reference_row(rows: list[dict[str, str]], run_tag: str) -> dict[str, str] | None:
    """The arm every other arm is diffed against.

    The reference is the *first* baseline-signature row in file order (the
    registry lists the plain GT-rule arm first, and configs/README.md declares
    it the baseline).  Returning ``None`` therefore means "the row being
    archived *is* that baseline", which is what makes ``mini_rule_seed7`` diff
    against ``mini_rule_seed42`` on the seed alone.
    """

    reference = baseline_row(rows)
    if reference is None:
        return None
    return None if (reference.get("run_tag") or "").strip() == run_tag else reference


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
PACKAGE_NAMES = [
    "torch",
    "transformers",
    "datasets",
    "numpy",
    "trl",
    "peft",
    "accelerate",
    "vllm",
    "verl",
    "flash_attn",
]


def package_versions(names: list[str] | None = None) -> dict[str, str]:
    import importlib

    versions: dict[str, str] = {}
    for name in names or PACKAGE_NAMES:
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def gpu_description() -> tuple[bool, str]:
    try:
        import torch
    except Exception:
        return False, "torch is not importable"
    if not torch.cuda.is_available():
        return False, f"no CUDA device (torch.cuda.device_count()={torch.cuda.device_count()})"
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return True, ", ".join(names)


def git_info(repo_root: Path) -> dict[str, str]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=20
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    if run("rev-parse", "--git-dir") is None:
        return {"commit": "n/a (no git repository in this checkout)", "status": "n/a"}
    return {
        "commit": run("rev-parse", "HEAD") or "unknown",
        "status": run("status", "--short") or "(clean)",
    }


def collect_environment(run_tag: str, repo_root: Path = REPO_ROOT, names: list[str] | None = None) -> str:
    has_cuda, gpu = gpu_description()
    git = git_info(repo_root)
    versions = package_versions(names)
    lines = [
        f"# environment for {run_tag}",
        f"captured_at: {utcnow()}",
        f"host: {platform.node()}",
        f"platform: {platform.platform()}",
        f"machine: {platform.machine()}",
        f"python: {platform.python_version()} ({platform.python_implementation()})",
        f"executable: {sys.executable}",
        f"cwd: {os.getcwd()}",
        f"cuda_available: {has_cuda}",
        f"gpu: {gpu}",
        f"cpu_count: {os.cpu_count()}",
        f"git_commit: {git['commit']}",
        "git_status:",
        git["status"] if git["status"] == "(clean)" else git["status"] or "n/a",
        "packages:",
    ]
    for name in sorted(versions):
        lines.append(f"  {name}=={versions[name]}")
    lines.append("nvidia-smi:")
    lines.append(run_nvidia_smi())
    return "\n".join(lines) + "\n"


def run_nvidia_smi() -> str:
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20)
    except Exception:
        return "  n/a (nvidia-smi not available)"
    if result.returncode != 0:
        return "  n/a (nvidia-smi failed)"
    return "\n".join("  " + line for line in result.stdout.strip().splitlines())


# --------------------------------------------------------------------------- #
# derived artifacts
# --------------------------------------------------------------------------- #
def checkpoints_index(prefix: str, directories: list[Path]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or not path.name.startswith(prefix):
                continue
            found.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "modified": dt.datetime.fromtimestamp(
                        path.stat().st_mtime, dt.timezone.utc
                    ).replace(microsecond=0).isoformat(),
                }
            )
    return found


def render_checkpoints(prefix: str, entries: list[dict[str, Any]], extra_notes: list[str]) -> str:
    lines = [f"# checkpoints matching prefix {prefix!r} (binaries are deliberately not archived)"]
    if not entries:
        lines.append("(none found; check --save_dir and --checkpoints-dir)")
    for entry in entries:
        lines.append(f"{entry['sha256']}  {entry['bytes']:>12}  {entry['modified']}  {entry['path']}")
    for note in extra_notes:
        lines.append(f"# note: {note}")
    return "\n".join(lines) + "\n"


def render_config_yaml(config: dict[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True, default_flow_style=False)


def metrics_digest(header: list[str], rows: list[dict[str, str]]) -> dict[str, Any]:
    """First/last/step-count summary of the training dynamics columns."""

    watched = [
        "avg_reward",
        "format_rate",
        "correct_rate",
        "no_answer_rate",
        "avg_completion_tokens",
        "median_completion_tokens",
        "kl_ref",
        "entropy",
        "adv_std",
        "policy_loss",
        "valid_groups",
        "all_correct_groups",
        "all_wrong_groups",
        "filtered_fraction",
    ]
    present = [column for column in watched if column in header]
    digest: dict[str, Any] = {"rows": len(rows), "columns": header, "first": {}, "last": {}, "min": {}, "max": {}}
    if not rows:
        return digest
    first, last = rows[0], rows[-1]
    for column in present:
        digest["first"][column] = _to_float(first.get(column))
        digest["last"][column] = _to_float(last.get(column))
        values = [v for v in (_to_float(row.get(column)) for row in rows) if v is not None]
        if values:
            digest["min"][column] = min(values)
            digest["max"][column] = max(values)
    digest["first_step"] = _to_float(first.get("step"))
    digest["last_step"] = _to_float(last.get("step"))
    return digest


def sample_status_counts(samples_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not samples_path.is_file():
        return counts
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["<unparseable>"] = counts.get("<unparseable>", 0) + 1
                continue
            completions = record.get("completions") or record.get("gens") or []
            for item in completions:
                status = (item or {}).get("status", "unknown") if isinstance(item, dict) else "unknown"
                counts[status] = counts.get(status, 0) + 1
    return counts


def eval_digest(eval_json: dict[str, Any] | None) -> str:
    if not eval_json:
        return "- pending: run `scripts/eval_heldout.py` and pass --eval-json (SOP section 10.1)\n"
    keys = [
        "model",
        "seed",
        "temperature",
        "num_samples",
        "max_new_tokens",
        "greedy_accuracy",
        "sampled_pass_at_1",
        "pass_at_4",
        "pass_at_8",
        "pass_at_16",
        "format_rate",
        "no_answer_rate",
        "marker_hit_rate",
        "mean_completion_tokens",
        "median_completion_tokens",
        "data_matches_manifest",
    ]
    lines = []
    for key in keys:
        if key in eval_json:
            lines.append(f"- {key}: {_fmt(eval_json[key])}")
    by_difficulty = eval_json.get("by_difficulty")
    if isinstance(by_difficulty, dict):
        lines.append("- by_difficulty:")
        for name, entry in by_difficulty.items():
            if isinstance(entry, dict):
                lines.append(
                    f"  - {name}: greedy={_fmt(entry.get('greedy_accuracy'))}, "
                    f"pass@1={_fmt(entry.get('sampled_pass_at_1'))}, n={_fmt(entry.get('rows_evaluated', entry.get('count')))}"
                )
            else:
                lines.append(f"  - {name}: {_fmt(entry)}")
    if eval_json.get("rows_available") is not None:
        lines.append(f"- rows_available: {_fmt(eval_json['rows_available'])}")
    return "\n".join(lines) + "\n"


def command_lead_line(command: str | None) -> str:
    """The line worth quoting in analysis.md.

    ``command.txt`` opens with the shell preamble (``set -o pipefail``,
    ``mkdir``, ``cd``), so the first plain ``python ...`` line is the one that
    names the script being run.
    """

    for line in (command or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("python "):
            return stripped
    for line in (command or "").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "set ", "mkdir", "cd ")):
            return stripped
    return "n/a"


def render_analysis(
    *,
    run_tag: str,
    row: dict[str, str] | None,
    changes: list[tuple[str, str, str]],
    is_reference_arm: bool,
    environment: str,
    digest: dict[str, Any],
    statuses: dict[str, int],
    eval_json: dict[str, Any] | None,
    samples_path: Path | None,
    command: str | None,
    notes: list[str],
) -> str:
    hypothesis = (row or {}).get("hypothesis", "n/a (run is not in the registry)")
    mode = (row or {}).get("reward_mode", "n/a")
    model = (row or {}).get("model", "n/a")
    env_lines = [line for line in environment.splitlines() if line.startswith(("python:", "platform:", "gpu:", "git_commit:", "host:"))]
    command_excerpt = command_lead_line(command)

    if changes:
        change_lines = [f"- {column}: reference={before} -> this run={after}" for column, before, after in changes]
    elif is_reference_arm:
        change_lines = ["- this is the reference arm itself (the other arms are diffed against it)"]
    elif row is None:
        change_lines = ["- unknown: run_tag is missing from configs/experiment_registry.csv"]
    else:
        change_lines = ["- n/a: no reference arm found in the registry"]

    lines = [
        f"# Run: {run_tag}",
        "",
        "## 1. 目的",
        "",
    ]
    lines += change_lines
    lines += [
        f"- reward_mode: {mode}, model: {model}",
        f"- 预注册假设: {hypothesis}",
        "",
        "## 2. 环境",
        "",
        f"- command: `{command_excerpt}` (full command in command.txt)",
    ]
    lines += [f"- {line}" for line in env_lines]
    lines += [
        "",
        "## 3. 训练动态",
        "",
        f"- steps logged: {digest.get('rows', 0)} (step {_fmt(digest.get('first_step'))} -> {_fmt(digest.get('last_step'))})",
    ]
    for column in ("avg_reward", "format_rate", "correct_rate", "avg_completion_tokens", "kl_ref", "entropy"):
        if column in digest.get("first", {}):
            lines += [
                f"- {column}: {_fmt(digest['first'][column])} -> {_fmt(digest['last'][column])} "
                f"(min {_fmt(digest.get('min', {}).get(column))}, max {_fmt(digest.get('max', {}).get(column))})"
            ]
    if "filtered_fraction" in digest.get("first", {}):
        lines.append(
            f"- dynamic sampling: filtered_fraction {_fmt(digest['first']['filtered_fraction'])} -> "
            f"{_fmt(digest['last']['filtered_fraction'])}, valid_groups last {_fmt(digest['last'].get('valid_groups'))}"
        )
    lines += [
        "",
        "## 4. Held-out",
        "",
        eval_digest(eval_json).rstrip("\n"),
        "",
        "## 5. Error analysis",
        "",
        f"- raw completions: {samples_path if samples_path else 'n/a'}",
        f"- status counts (training samples): {statuses if statuses else 'n/a'}",
        "- TODO: 人工标注 20 条 held-out 采样（正确/格式错/数值错/无答案/超长），把类别统计与典型路径写在这里。",
        "",
        "## 6. 结论边界",
        "",
        "- TODO: 支持/不支持哪条预注册假设（对照 §4 的数值，不要引用训练期 reward）。",
        "- TODO: 哪些结论不能从本 run 推出（单 seed / 单规模 / 单难度分布 / 未做多轮评估）。",
        "- TODO: 下一步最小实验是什么。",
        "",
    ]
    if notes:
        lines += ["## 7. 归档备注", ""]
        lines += [f"- {note}" for note in notes]
        lines.append("")
    return "\n".join(lines)


def data_manifest_payload(
    *,
    run_tag: str,
    row: dict[str, str] | None,
    manifest_path: Path,
    data_paths: list[Path],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "captured_at": utcnow(),
        "run_tag": run_tag,
        "registry_row": row,
        "dataset_manifest_path": str(manifest_path),
        "files": {},
    }
    recorded: dict[str, Any] = {}
    if manifest_path.is_file():
        payload["dataset_manifest_sha256"] = sha256_file(manifest_path)
        manifest = read_json(manifest_path)
        payload["dataset_manifest"] = manifest
        recorded = manifest.get("outputs") or {}
    else:
        payload["dataset_manifest_sha256"] = None
    for path in data_paths:
        if not path.is_file():
            payload["files"][str(path)] = {"exists": False}
            continue
        digest = sha256_file(path)
        name = path.name
        payload["files"][str(path)] = {
            "exists": True,
            "bytes": path.stat().st_size,
            "sha256": digest,
            "recorded_sha256": recorded.get(name),
            "matches_manifest": (recorded.get(name) == digest) if name in recorded else None,
        }
    return payload


# --------------------------------------------------------------------------- #
# archiving
# --------------------------------------------------------------------------- #
def archive_run(
    run_tag: str,
    *,
    log_dir: Path = Path("logs"),
    runs_dir: Path = Path("runs"),
    out_dir: Path = Path("out"),
    checkpoints_dir: Path = Path("checkpoints"),
    dataset_dir: Path = Path("dataset"),
    registry_path: Path = DEFAULT_REGISTRY,
    eval_json: Path | None = None,
    repo_root: Path = REPO_ROOT,
    force: bool = False,
    environment: str | None = None,
) -> dict[str, Any]:
    """Write ``runs/{run_tag}/`` and return the archive manifest."""

    if run_tag in PLACEHOLDER_TAGS:
        raise ValueError(f"refusing to archive {run_tag!r}: the trainer default tag is not a run")

    run_dir = runs_dir / run_tag
    if run_dir.exists() and any(run_dir.iterdir()) and not force:
        raise FileExistsError(f"{run_dir} already exists; pass --force to refresh the derived files")

    run_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    written: dict[str, Any] = {}

    # ---- registry / command ----
    row = registry_row(registry_path, run_tag)
    header, rows = load_registry(registry_path)
    problems = check_registry(header, rows, trainer_flags(REPO_ROOT / "trainer" / "train_grpo.py"))
    if problems:
        raise ValueError("run registry is not valid, archiving it would freeze a broken config:\n  " + "\n  ".join(problems))
    if row is None:
        notes.append(f"{run_tag} is not present in {registry_path}; command.txt holds no command, only a warning")
        command = (
            f"# {run_tag} is not registered in {registry_path}\n"
            "# an unregistered run cannot be reproduced from the registry;\n"
            "# paste the executed command here before treating this run as evidence\n"
        )
    else:
        command = render_command(run_tag, build_args(row))
    (run_dir / "command.txt").write_text(command, encoding="utf-8")
    written["command.txt"] = {"sha256": sha256_file(run_dir / "command.txt"), "kind": "derived"}

    reference = reference_row(rows, run_tag) if row else None
    changes = changed_columns(row, reference) if row else []
    is_reference_arm = row is not None and reference is None

    # ---- raw evidence ----
    metrics_src = log_dir / f"{run_tag}_metrics.csv"
    header_cols: list[str] = []
    metric_rows: list[dict[str, str]] = []
    if metrics_src.is_file():
        header_cols, metric_rows = read_metrics(metrics_src)
        shutil.copy2(metrics_src, run_dir / "metrics.csv")
        written["metrics.csv"] = {"sha256": sha256_file(run_dir / "metrics.csv"), "kind": "raw"}
    else:
        notes.append(f"metrics missing: {metrics_src} not found")

    samples_src = log_dir / f"{run_tag}_samples.jsonl"
    if samples_src.is_file():
        shutil.copy2(samples_src, run_dir / "samples.jsonl")
        written["samples.jsonl"] = {"sha256": sha256_file(run_dir / "samples.jsonl"), "kind": "raw"}
    else:
        notes.append(f"samples missing: {samples_src} not found")

    archive_eval = run_dir / "eval.json"
    candidates = [path for path in (eval_json, log_dir / f"{run_tag}_eval.json", log_dir / "eval_heldout.json") if path]
    chosen = next((path for path in candidates if path.is_file() and path.resolve() != archive_eval.resolve()), None)
    eval_payload: dict[str, Any] | None = None
    if chosen is not None:
        shutil.copy2(chosen, archive_eval)
        written["eval.json"] = {"sha256": sha256_file(archive_eval), "kind": "raw", "source": str(chosen)}
        if eval_json is None and chosen == log_dir / "eval_heldout.json":
            notes.append(
                "eval.json was taken from the default logs/eval_heldout.json path; confirm it belongs to this run"
            )
        # eval_heldout.py writes the per-sample JSONL next to the JSON file.
        sidecar = Path(str(chosen) + "l")
        if sidecar.is_file():
            shutil.copy2(sidecar, run_dir / "eval.jsonl")
            written["eval.jsonl"] = {"sha256": sha256_file(run_dir / "eval.jsonl"), "kind": "raw"}
    elif archive_eval.is_file():
        written["eval.json"] = {"sha256": sha256_file(archive_eval), "kind": "raw"}
    else:
        notes.append("eval missing: no held-out evaluation archived yet (see SOP section 10)")
    if archive_eval.is_file():
        eval_payload = read_json(archive_eval)

    # ---- config.yaml ----
    config_src = log_dir / f"{run_tag}_config.json"
    if config_src.is_file():
        config = read_json(config_src)
        (run_dir / "config.yaml").write_text(render_config_yaml(config), encoding="utf-8")
        written["config.yaml"] = {"sha256": sha256_file(run_dir / "config.yaml"), "kind": "derived", "source": str(config_src)}
    else:
        notes.append(f"config missing: {config_src} not found")

    # ---- environment.txt ----
    env_text = environment if environment is not None else collect_environment(run_tag, repo_root=repo_root)
    (run_dir / "environment.txt").write_text(env_text, encoding="utf-8")
    written["environment.txt"] = {"sha256": sha256_file(run_dir / "environment.txt"), "kind": "derived"}

    # ---- data_manifest.json ----
    data_paths = [dataset_dir / "math_rlvr_train.jsonl", dataset_dir / "math_heldout.jsonl"]
    payload = data_manifest_payload(
        run_tag=run_tag,
        row=row,
        manifest_path=dataset_dir / "manifests" / "math_manifest.json",
        data_paths=data_paths,
    )
    (run_dir / "data_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    written["data_manifest.json"] = {"sha256": sha256_file(run_dir / "data_manifest.json"), "kind": "derived"}
    for path, info in payload["files"].items():
        if info.get("exists") and info.get("matches_manifest") is False:
            notes.append(f"{path} no longer matches the recorded sha256 in the dataset manifest")
        if info.get("exists") and info.get("matches_manifest") is None:
            notes.append(f"{path} is not listed in the dataset manifest")

    # ---- checkpoints.txt ----
    entries = checkpoints_index(run_tag, [out_dir, checkpoints_dir])
    checkpoint_notes: list[str] = []
    if not entries:
        checkpoint_notes.append(f"no checkpoint file found with prefix {run_tag!r} (check --save_dir)")
        notes.extend(checkpoint_notes)
    (run_dir / "checkpoints.txt").write_text(
        render_checkpoints(run_tag, entries, checkpoint_notes), encoding="utf-8"
    )
    written["checkpoints.txt"] = {"sha256": sha256_file(run_dir / "checkpoints.txt"), "kind": "derived"}

    # ---- analysis.md ----
    digest = metrics_digest(header_cols, metric_rows)
    statuses = sample_status_counts(run_dir / "samples.jsonl")
    analysis = render_analysis(
        run_tag=run_tag,
        row=row,
        changes=changes,
        is_reference_arm=is_reference_arm,
        environment=env_text,
        digest=digest,
        statuses=statuses,
        eval_json=eval_payload,
        samples_path=run_dir / "samples.jsonl" if (run_dir / "samples.jsonl").is_file() else None,
        command=command,
        notes=notes,
    )
    (run_dir / "analysis.md").write_text(analysis, encoding="utf-8")
    written["analysis.md"] = {"sha256": sha256_file(run_dir / "analysis.md"), "kind": "derived"}

    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    required = REQUIRED_FILES if row is not None else PREREQUISITE_REQUIRED_FILES
    manifest = {
        "created_at": utcnow(),
        "run_tag": run_tag,
        "registry_row": row,
        "prerequisite": row is None,
        "status": "complete" if not missing else "partial",
        "required_files": required,
        "missing": missing,
        "raw_evidence": [name for name in RAW_EVIDENCE if (run_dir / name).is_file()],
        "written_files": written,
        "notes": notes,
        "metrics_rows": len(metric_rows),
        "train_log": (log_dir / f"{run_tag}_train.log").is_file() or (run_dir / "train.log").is_file(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def check_archive(run_dir: Path, *, strict: bool = False) -> list[str]:
    """Return problems found in an existing archive directory."""

    problems: list[str] = []
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        problems.append(f"{run_dir}: manifest.json missing (was it archived by scripts/archive_run.py?)")
        manifest: dict[str, Any] = {}
        files = {name: run_dir / name for name in REQUIRED_FILES}
    else:
        manifest = read_json(manifest_path)
        files = {name: run_dir / name for name in manifest.get("required_files", REQUIRED_FILES)}

    # eval.json and train.log only become hard requirements with --strict: a run
    # is routinely archived right after training and evaluated later.
    deferred = set() if strict else {"eval.json"}
    for name, path in files.items():
        if name == "eval.json" and strict:
            continue  # reported by the --strict check below, once and by name
        if not path.is_file():
            if name not in deferred:
                problems.append(f"{name} missing")
            continue
        expected = (manifest.get("written_files") or {}).get(name, {}).get("sha256")
        if expected:
            actual = sha256_file(path)
            if actual != expected:
                problems.append(f"{name} sha256 changed after archiving (expected {expected[:12]}, found {actual[:12]})")

    metrics = run_dir / "metrics.csv"
    if metrics.is_file():
        header, rows = read_metrics(metrics)
        if not header:
            problems.append("metrics.csv has no header")
        elif not rows:
            problems.append("metrics.csv has a header but no data rows")

    for name in RAW_EVIDENCE:
        path = run_dir / name
        if path.is_file() and path.stat().st_size == 0:
            problems.append(f"{name} is empty")

    eval_path = run_dir / "eval.json"
    if strict and not eval_path.is_file():
        problems.append("eval.json missing (required by --strict)")
    if eval_path.is_file():
        try:
            payload = read_json(eval_path)
        except json.JSONDecodeError as exc:
            problems.append(f"eval.json is not valid JSON: {exc}")
        else:
            for key in EVAL_REQUIRED_KEYS:
                if key not in payload:
                    problems.append(f"eval.json lacks the required field {key!r} (SOP section 10.1)")

    if strict and not (run_dir / "train.log").is_file():
        problems.append("train.log missing (required by --strict: the SOP captures it with tee)")
    return problems


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble or check the SOP section 13 archive for a registered run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-tag", default=None, help="run_tag from configs/experiment_registry.csv")
    parser.add_argument("--all", action="store_true", help="archive every run_tag that has logs in --log-dir")
    parser.add_argument("--check", action="store_true", help="verify an existing archive instead of writing it")
    parser.add_argument("--strict", action="store_true", help="with --check: require eval.json and train.log")
    parser.add_argument("--force", action="store_true", help="refresh derived files in an existing archive")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--eval-json", type=Path, default=None, help="held-out evaluation JSON to attach")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def discover_tags(log_dir: Path, restrict: list[str] | None = None) -> list[str]:
    """Run tags with metrics logs, restricted to the registry when set."""

    if not log_dir.is_dir():
        return []
    found = sorted({path.name[: -len("_metrics.csv")] for path in log_dir.glob("*_metrics.csv")})
    if restrict is not None:
        found = [tag for tag in found if tag in set(restrict)]
    return found


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.all:
        if args.run_tag:
            print("--all and --run-tag are mutually exclusive", file=sys.stderr)
            return 2
        _header, rows = load_registry(args.registry)
        tags = discover_tags(args.log_dir, [row.get("run_tag", "").strip() for row in rows])
        if not tags:
            print(f"no metrics logs found in {args.log_dir}", file=sys.stderr)
            return 1
    elif args.run_tag:
        tags = [args.run_tag]
    else:
        print("pass --run-tag TAG or --all", file=sys.stderr)
        return 2

    exit_code = 0
    for tag in tags:
        if args.check:
            problems = check_archive(args.runs_dir / tag, strict=args.strict)
            if problems:
                exit_code = 1
                print(f"[FAIL] {tag}")
                for problem in problems:
                    print(f"  - {problem}")
            elif not args.quiet:
                print(f"[ OK ] {tag}")
            continue
        try:
            manifest = archive_run(
                tag,
                log_dir=args.log_dir,
                runs_dir=args.runs_dir,
                out_dir=args.out_dir,
                checkpoints_dir=args.checkpoints_dir,
                dataset_dir=args.dataset_dir,
                registry_path=args.registry,
                eval_json=args.eval_json,
                force=args.force,
            )
        except (FileExistsError, ValueError) as exc:
            exit_code = 1
            print(f"[FAIL] {tag}: {exc}", file=sys.stderr)
            continue
        if not args.quiet:
            print(f"[{manifest['status']}] runs/{tag}: {manifest['metrics_rows']} metric rows"
                  + (f", missing {', '.join(manifest['missing'])}" if manifest["missing"] else ""))
            for note in manifest["notes"]:
                print(f"  - {note}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
