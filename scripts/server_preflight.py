#!/usr/bin/env python
"""Server-side preflight: fail before burning GPU hours, not after.

Run this as the first command of every remote session (SOP sections 2.1-2.3,
7.2, 12).  It checks the four things that actually waste a rented GPU slot:

1. the runtime (python / torch / CUDA / dtype support) can run this repo;
2. the data triple is present *and* still matches ``data.sha256``;
3. the starting checkpoint the next command needs actually exists;
4. there is disk and write permission for ``out/``, ``logs/``, ``runs/``,
   ``reports/`` and the HuggingFace cache.

Every check is informational except the ones marked FAIL: FAIL means "do not
start training", WARN means "know what you are accepting".

    python scripts/server_preflight.py
    python scripts/server_preflight.py --expect-gpu --min-disk-gb 60
    python scripts/server_preflight.py --json /tmp/preflight.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OK = "ok"
WARN = "warn"
FAIL = "fail"

DATASET_FILES = ["math_sft_train.jsonl", "math_rlvr_train.jsonl", "math_heldout.jsonl"]
REQUIRED_PACKAGES = ["torch", "transformers", "datasets", "numpy"]
OPTIONAL_PACKAGES = ["matplotlib", "yaml"]
MIN_PYTHON = (3, 10)


def check(status: str, name: str, detail: str) -> dict[str, str]:
    return {"status": status, "name": name, "detail": detail}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def check_python(version: tuple[int, int] | None = None) -> dict[str, str]:
    version = version or sys.version_info[:2]
    text = f"{version[0]}.{version[1]}"
    if version < MIN_PYTHON:
        return check(FAIL, "python", f"{text} is older than the required {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
    return check(OK, "python", text)


def check_packages(versions: dict[str, str] | None = None) -> list[dict[str, str]]:
    """Which of the required/optional imports resolved, and at what version."""

    if versions is None:
        import importlib

        versions = {}
        for name in REQUIRED_PACKAGES + OPTIONAL_PACKAGES:
            try:
                module = importlib.import_module(name)
            except Exception:
                continue
            versions[name] = str(getattr(module, "__version__", "unknown"))

    checks = []
    for name in REQUIRED_PACKAGES:
        if name in versions:
            checks.append(check(OK, f"package:{name}", versions[name]))
        else:
            checks.append(check(FAIL, f"package:{name}", "not importable; `pip install -r requirements.txt`"))
    for name in OPTIONAL_PACKAGES:
        if name in versions:
            checks.append(check(OK, f"package:{name}", versions[name]))
        else:
            checks.append(check(WARN, f"package:{name}",
                                "missing; summary.md still works, yaml config and figures do not"))
    return checks


def check_torch(cuda: dict[str, Any] | None = None, *, expect_gpu: bool) -> list[dict[str, str]]:
    """CUDA availability, device name and bf16 support."""

    if cuda is None:
        try:
            import torch
        except Exception as exc:
            return [check(FAIL, "torch", f"import failed: {exc}")]
        info: dict[str, Any] = {
            "version": torch.__version__,
            "available": torch.cuda.is_available(),
            "count": torch.cuda.device_count(),
        }
        if info["available"]:
            info["name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["memory_gb"] = round(props.total_memory / 1024**3, 1)
            info["capability"] = torch.cuda.get_device_capability(0)
            try:
                info["bf16"] = bool(torch.cuda.is_bf16_supported())
            except Exception:
                info["bf16"] = False
        cuda = info

    status = OK if cuda.get("available") else (FAIL if expect_gpu else WARN)
    detail = f"torch {cuda.get('version', '?')}, cuda_available={cuda.get('available')}"
    checks = [check(status, "torch/cuda", detail)]
    if cuda.get("available"):
        checks.append(check(
            OK, "gpu",
            f"{cuda.get('name')} x{cuda.get('count')}, {cuda.get('memory_gb')} GB, "
            f"capability {cuda.get('capability')}, bf16={cuda.get('bf16')}",
        ))
        if cuda.get("bf16") is False:
            checks.append(check(WARN, "dtype",
                                "bf16 unsupported on this GPU: run with --dtype float16 or expect a crash"))
        memory = cuda.get("memory_gb") or 0
        if memory and memory < 20:
            checks.append(check(WARN, "vram",
                                f"{memory} GB is below the 24 GB the SOP assumes; "
                                "reduce --batch_size/--num_generations or use --max_gen_len 256"))
    else:
        checks.append(check(WARN if not expect_gpu else FAIL, "gpu",
                            "no CUDA device: only code-path smokes are meaningful here"))
    return checks


def check_layout(repo_root: Path) -> list[dict[str, str]]:
    """The files the first two training commands read."""

    wanted = [
        ("model/tokenizer.json", FAIL),
        ("model/tokenizer_config.json", FAIL),
        ("trainer/train_full_sft.py", FAIL),
        ("trainer/train_grpo.py", FAIL),
        ("dataset/math_rl_dataset.py", FAIL),
        ("trainer/math_verifier.py", FAIL),
        ("scripts/clean_math_data.py", FAIL),
        ("scripts/eval_heldout.py", FAIL),
        ("scripts/archive_run.py", FAIL),
        ("scripts/summarize_runs.py", FAIL),
        ("configs/experiment_registry.csv", FAIL),
        ("dataset/manifests/math_manifest.json", WARN),
        ("dataset/manifests/data.sha256", WARN),
    ]
    checks = []
    for relative, missing_status in wanted:
        path = repo_root / relative
        checks.append(check(OK, relative, f"{path.stat().st_size} bytes") if path.is_file()
                      else check(missing_status, relative, "missing"))
    return checks


def check_dataset(dataset_dir: Path, recorded: dict[str, str] | None = None) -> list[dict[str, str]]:
    """Row counts plus the recorded sha256 of every file we train or evaluate on."""

    checks = []
    for name in DATASET_FILES:
        path = dataset_dir / name
        if not path.is_file():
            checks.append(check(FAIL, name, "missing; regenerate with scripts/clean_math_data.py "
                                            "--synthetic-only (or copy the dataset/ directory)"))
            continue
        rows = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        digest = sha256_file(path)
        expected = (recorded or {}).get(name)
        if expected is None:
            checks.append(check(OK, name, f"{rows} rows, sha256={digest[:12]} (not in data.sha256)"))
        elif expected == digest:
            checks.append(check(OK, name, f"{rows} rows, sha256={digest[:12]} matches data.sha256"))
        else:
            checks.append(check(FAIL, name, f"sha256={digest[:12]} but data.sha256 records {expected[:12]}; "
                                            "the data changed after the registry was written"))
    return checks


def read_recorded_hashes(sha_path: Path) -> dict[str, str]:
    recorded: dict[str, str] = {}
    if not sha_path.is_file():
        return recorded
    for line in sha_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            recorded[parts[1].lstrip("*")] = parts[0]
    return recorded


def check_weights(out_dir: Path, *, from_weight: str = "full_sft", hidden_size: int = 768,
                  use_moe: bool = False) -> dict[str, str]:
    """``init_model`` loads ``{save_dir}/{from_weight}_{hidden}{_moe}.pth`` verbatim."""

    suffix = "_moe" if use_moe else ""
    path = out_dir / f"{from_weight}_{hidden_size}{suffix}.pth"
    if not path.is_file():
        return check(FAIL, f"weights:{path.name}",
                     f"missing in {out_dir}; download the official MiniMind weights and copy this file there "
                     "(init_model loads it with strict=False, so a missing file is fatal but a wrong one is silent)")
    size_gb = path.stat().st_size / 1024**3
    return check(OK, f"weights:{path.name}", f"{size_gb:.2f} GB, sha256={sha256_file(path)[:12]}")


def check_disk(paths: dict[str, Path], *, min_gb: float, cache_dir: Path | None = None) -> list[dict[str, str]]:
    checks = []
    seen: set[str] = set()
    for label, path in list(paths.items()) + ([("hf_cache", cache_dir)] if cache_dir else []):
        if path is None:
            continue
        probe = path
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        usage = shutil.disk_usage(probe)
        free_gb = usage.free / 1024**3
        if str(probe) in seen:
            continue
        seen.add(str(probe))
        status = OK if free_gb >= min_gb else FAIL
        checks.append(check(status, f"disk:{label}", f"{free_gb:.1f} GB free on {probe} (need >= {min_gb} GB)"))
    return checks


def check_writable(dirs: list[Path]) -> list[dict[str, str]]:
    checks = []
    for directory in dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".preflight_write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            checks.append(check(FAIL, f"writable:{directory}", str(exc)))
        else:
            checks.append(check(OK, f"writable:{directory}", "created and removed a probe file"))
    return checks


def hf_cache_dir() -> Path:
    import os

    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "datasets"
    return Path.home() / ".cache" / "huggingface" / "datasets"


def collect(
    *,
    repo_root: Path = REPO_ROOT,
    dataset_dir: Path | None = None,
    out_dir: Path | None = None,
    from_weight: str = "full_sft",
    hidden_size: int = 768,
    expect_gpu: bool = False,
    min_disk_gb: float = 20.0,
) -> list[dict[str, str]]:
    dataset_dir = dataset_dir or repo_root / "dataset"
    out_dir = out_dir or repo_root / "out"
    checks = [check_python()]
    checks += check_packages()
    checks += check_torch(expect_gpu=expect_gpu)
    checks += check_layout(repo_root)
    recorded = read_recorded_hashes(dataset_dir / "manifests" / "data.sha256")
    checks += check_dataset(dataset_dir, recorded)
    checks.append(check_weights(out_dir, from_weight=from_weight, hidden_size=hidden_size))
    checks += check_disk(
        {"repo": repo_root, "out": out_dir, "hf_datasets": hf_cache_dir()},
        min_gb=min_disk_gb,
    )
    checks += check_writable([repo_root / "out", repo_root / "logs", repo_root / "runs", repo_root / "reports"])
    return checks


def summarize(checks: list[dict[str, str]]) -> dict[str, int]:
    counts = {OK: 0, WARN: 0, FAIL: 0}
    for item in checks:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return counts


def render(checks: list[dict[str, str]]) -> str:
    icons = {OK: " OK ", WARN: "WARN", FAIL: "FAIL"}
    width = max((len(item["name"]) for item in checks), default=8)
    lines = [f"{icons[item['status']]}  {item['name'].ljust(width)}  {item['detail']}" for item in checks]
    counts = summarize(checks)
    lines.append("")
    lines.append(f"{counts[OK]} ok, {counts[WARN]} warnings, {counts[FAIL]} blockers")
    if counts[FAIL]:
        lines.append("Do not start training until every FAIL is resolved (SOP section 12).")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a (remote) machine before starting the RLVR runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--from-weight", type=str, default="full_sft",
                        help="checkpoint prefix the next command will load")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--expect-gpu", action="store_true",
                        help="treat 'no CUDA device' as a blocker instead of a warning")
    parser.add_argument("--min-disk-gb", type=float, default=20.0)
    parser.add_argument("--json", type=Path, default=None, help="also write the raw checks here")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checks = collect(
        repo_root=args.repo_root,
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        from_weight=args.from_weight,
        hidden_size=args.hidden_size,
        expect_gpu=args.expect_gpu,
        min_disk_gb=args.min_disk_gb,
    )
    if not args.quiet:
        print(render(checks))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"checks": checks, "summary": summarize(checks)}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 1 if summarize(checks)[FAIL] else 0


if __name__ == "__main__":
    raise SystemExit(main())
