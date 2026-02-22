import os
import sys
import json
import time
import shlex
import shutil
import argparse
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = sys.executable


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path):
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def ensure_dir(path):
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def now_ts():
    return time.strftime("%Y%m%d_%H%M%S")


def dict_to_args(d, flags=None):
    flags = set(flags or [])
    args = []
    for k, v in d.items():
        key = f"--{k}"
        if isinstance(v, bool):
            if k in flags:
                if v:
                    args.append(key)
            else:
                args.extend([key, str(int(v))])
        elif v is None:
            continue
        else:
            args.extend([key, str(v)])
    return args


def build_method_args(method, growth_cfg):
    if method == "baseline":
        return {"neuron_growth": 0}
    if method == "random":
        return {"neuron_growth": 1, "grow_method": "random", **growth_cfg}
    if method == "grad":
        return {
            "neuron_growth": 1,
            "grow_method": "act_grad",
            "grow_score_alpha": 0.0,
            "grow_score_beta": 1.0,
            **growth_cfg
        }
    if method == "act":
        return {
            "neuron_growth": 1,
            "grow_method": "act_grad",
            "grow_score_alpha": 1.0,
            "grow_score_beta": 0.0,
            **growth_cfg
        }
    if method == "actgrad":
        return {
            "neuron_growth": 1,
            "grow_method": "act_grad",
            "grow_score_alpha": 1.0,
            "grow_score_beta": 1.0,
            **growth_cfg
        }
    raise ValueError(f"Unknown method: {method}")


def run_cmd(cmd, log_path, env=None, cwd=None, retries=0, dry_run=False):
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    print("[CMD]", cmd_str)
    if dry_run:
        return True
    for attempt in range(retries + 1):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n[{stamp}] $ {cmd_str}\n")
            proc = subprocess.run(cmd, cwd=cwd or ROOT, env=env, stdout=log, stderr=log, text=True)
        if proc.returncode == 0:
            return True
        if attempt < retries:
            time.sleep(5)
    return False


def run_capture(cmd):
    try:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return proc.returncode == 0, proc.stdout.strip()
    except FileNotFoundError:
        return False, ""
    except Exception as e:
        return False, str(e)


def disk_free_gb(path):
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def safe_remove(path):
    abs_path = resolve_path(path)
    if not abs_path:
        return
    if not abs_path.startswith(ROOT + os.sep):
        print(f"[WARN] Skip remove outside repo: {abs_path}")
        return
    if os.path.isdir(abs_path):
        shutil.rmtree(abs_path, ignore_errors=True)
    elif os.path.isfile(abs_path):
        try:
            os.remove(abs_path)
        except OSError:
            pass


def cleanup_paths(paths):
    for p in paths:
        if p == "__pycache__":
            for root, dirs, _ in os.walk(ROOT):
                for d in dirs:
                    if d == "__pycache__":
                        safe_remove(os.path.join(root, d))
        else:
            safe_remove(p)


def is_lfs_pointer(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.readline().strip()
        return head.startswith("version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


def preflight_checks(runtime, log_path):
    checks = []
    ok_git, out_git = run_capture(["git", "--version"])
    checks.append(("git", ok_git, out_git or "not found"))
    ok_lfs, out_lfs = run_capture(["git", "lfs", "version"])
    checks.append(("git_lfs", ok_lfs, out_lfs or "not found"))

    ok_smi, out_smi = run_capture([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"
    ])
    checks.append(("nvidia_smi", ok_smi, out_smi or "not found"))

    min_free = runtime.get("min_free_gb", 0)
    free_gb = disk_free_gb(ROOT)
    disk_ok = free_gb >= min_free if min_free else True
    checks.append(("disk_free_gb", disk_ok, f"{free_gb:.1f}GB (min {min_free}GB)"))

    # 写入日志
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n[{stamp}] [PREFLIGHT]\n")
        for name, ok, info in checks:
            log.write(f"{name}: {'OK' if ok else 'FAIL'} | {info}\n")
    return checks


def ensure_dataset_repo(dataset_cfg, log_path, env, dry_run, retries=0):
    target = resolve_path(dataset_cfg.get("target_dir", "minimind_dataset"))
    if os.path.isdir(target):
        return True
    repo = dataset_cfg.get("repo", "")
    if not repo:
        print("[WARN] dataset.repo not set")
        return False
    ok = run_cmd(["git", "lfs", "install"], log_path, env=env, dry_run=dry_run, retries=retries)
    ok = run_cmd(["git", "clone", repo, target], log_path, env=env, dry_run=dry_run, retries=retries) and ok
    return ok


def lfs_pull(dataset_cfg, files, log_path, env, dry_run, retries=0):
    target = resolve_path(dataset_cfg.get("target_dir", "minimind_dataset"))
    if not os.path.isdir(target):
        return False
    mode = dataset_cfg.get("download_mode", "selective")
    if mode == "full":
        return run_cmd(["git", "lfs", "pull"], log_path, env=env, cwd=target, dry_run=dry_run, retries=retries)
    files = [f for f in files if f]
    if not files:
        return True
    include = ",".join(files)
    return run_cmd(["git", "lfs", "pull", "--include", include], log_path, env=env, cwd=target, dry_run=dry_run, retries=retries)


def ensure_dataset_files(dataset_cfg, stage, log_path, env, dry_run, retries=0):
    if not ensure_dataset_repo(dataset_cfg, log_path, env, dry_run, retries=retries):
        return False
    stage_files = dataset_cfg.get("stage_files", {}).get(stage, [])
    if not stage_files:
        return True
    # 缺失就尝试拉取
    missing = []
    for name in stage_files:
        file_path = resolve_path(os.path.join(dataset_cfg.get("target_dir", "minimind_dataset"), name))
        if not os.path.exists(file_path) or is_lfs_pointer(file_path):
            missing.append(name)
    if not missing:
        return True
    ok = lfs_pull(dataset_cfg, missing, log_path, env, dry_run, retries=retries)
    if not ok:
        return False
    # 再次检查
    for name in missing:
        file_path = resolve_path(os.path.join(dataset_cfg.get("target_dir", "minimind_dataset"), name))
        if not os.path.exists(file_path) or is_lfs_pointer(file_path):
            return False
    return True


def weight_path(prefix, hidden_size, use_moe, save_dir):
    moe_suffix = "_moe" if use_moe else ""
    return os.path.join(resolve_path(save_dir), f"{prefix}_{hidden_size}{moe_suffix}.pth")


def main():
    parser = argparse.ArgumentParser(description="MiniMind 过夜一键训练/评测/汇总/绘图脚本（强纠错）")
    parser.add_argument("--config", default="eval/pipeline_config_overnight.json", type=str, help="配置文件路径")
    parser.add_argument("--stages", default="", type=str, help="仅运行的阶段（逗号分隔）")
    parser.add_argument("--methods", default="", type=str, help="方法列表（逗号分隔）")
    parser.add_argument("--seeds", default="", type=str, help="随机种子列表（逗号分隔）")
    parser.add_argument("--dry_run", action="store_true", help="只打印命令，不执行")
    parser.add_argument("--stop_on_error", action="store_true", help="遇到错误就停止")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    runtime = cfg.get("runtime", {})
    retry_download = int(runtime.get("retry_download", 0))
    dataset_cfg = cfg.get("dataset", {})
    paths = cfg.get("paths", {})
    growth_cfg = cfg.get("growth", {})
    cache_cfg = cfg.get("cache", {})

    stages = cfg.get("stages", [])
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    methods = cfg.get("methods", ["baseline"])
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    seeds = cfg.get("seeds", [42])
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    log_dir = resolve_path(paths.get("log_dir", "logs"))
    ensure_dir(log_dir)
    log_path = os.path.join(log_dir, f"pipeline_{now_ts()}.log")

    # 统一缓存目录，方便清理
    env = os.environ.copy()
    hf_home = resolve_path(cache_cfg.get("hf_home", ".cache/hf_home"))
    hf_datasets = resolve_path(cache_cfg.get("hf_datasets", ".cache/hf_datasets"))
    hf_transformers = resolve_path(cache_cfg.get("transformers", ".cache/hf_transformers"))
    env["HF_HOME"] = hf_home
    env["HF_DATASETS_CACHE"] = hf_datasets
    env["TRANSFORMERS_CACHE"] = hf_transformers
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    failures = []

    def mark_fail(stage, msg):
        failures.append(f"{stage}: {msg}")
        print(f"[FAIL] {stage}: {msg}")
        if args.stop_on_error or not runtime.get("continue_on_error", True):
            raise RuntimeError(msg)

    def maybe_cleanup(stage_name):
        cleanup_after = runtime.get("cleanup_after_stage", [])
        if stage_name in cleanup_after:
            cleanup_paths(runtime.get("cleanup_paths", []))

    def disk_guard(stage_name):
        min_free = runtime.get("min_free_gb", 0)
        if not min_free:
            return True
        free_gb = disk_free_gb(ROOT)
        if free_gb >= min_free:
            return True
        print(f"[WARN] Free disk {free_gb:.1f}GB < {min_free}GB, try cleanup")
        cleanup_paths(runtime.get("cleanup_paths", []))
        free_gb = disk_free_gb(ROOT)
        if free_gb < min_free:
            mark_fail(stage_name, f"disk too low: {free_gb:.1f}GB")
            return False
        return True

    # ========== Stage: preflight ==========
    if "preflight" in stages:
        checks = preflight_checks(runtime, log_path)
        for name, ok, info in checks:
            status = "OK" if ok else "FAIL"
            print(f"[PREFLIGHT] {name}: {status} | {info}")
            if not ok:
                mark_fail("preflight", f"{name} check failed")
        maybe_cleanup("preflight")

    # ========== Stage: prepare ==========
    if "prepare" in stages:
        ensure_dir(resolve_path(paths.get("out_dir", "out")))
        ensure_dir(resolve_path(paths.get("eval_runs", "eval_runs")))
        ensure_dir(resolve_path(paths.get("eval_runs_ppl", "eval_runs_ppl")))
        ensure_dir(hf_home)
        ensure_dir(hf_datasets)
        ensure_dir(hf_transformers)
        maybe_cleanup("prepare")

    # ========== Stage: download ==========
    if "download" in stages:
        if disk_guard("download"):
            if not ensure_dataset_repo(dataset_cfg, log_path, env, args.dry_run, retries=retry_download):
                mark_fail("download", "dataset repo init failed")
            else:
                stage_files = dataset_cfg.get("stage_files", {})
                union_files = []
                for s in stages:
                    union_files.extend(stage_files.get(s, []))
                ok = lfs_pull(dataset_cfg, sorted(set(union_files)), log_path, env, args.dry_run, retries=retry_download)
                if not ok:
                    mark_fail("download", "git lfs pull failed")
        maybe_cleanup("download")

    # ========== Stage: make_val ==========
    if "make_val" in stages:
        if disk_guard("make_val"):
            if not ensure_dataset_files(dataset_cfg, "make_val", log_path, env, args.dry_run, retries=retry_download):
                mark_fail("make_val", "missing pretrain data")
            else:
                cmd = [
                    PYTHON, "scripts/make_val_split.py",
                    "--data_path", resolve_path(paths.get("pretrain_data", "")),
                    "--out_path", resolve_path(paths.get("val_data", "eval/val_pretrain.jsonl")),
                    "--val_size", "2000",
                    "--seed", "42"
                ]
                if not run_cmd(cmd, log_path, env=env, dry_run=args.dry_run):
                    mark_fail("make_val", "script failed")
        maybe_cleanup("make_val")

    # ========== Stage: train_pretrain ==========
    if "train_pretrain" in stages:
        if disk_guard("train_pretrain"):
            if not ensure_dataset_files(dataset_cfg, "train_pretrain", log_path, env, args.dry_run, retries=retry_download):
                mark_fail("train_pretrain", "missing pretrain data")
            else:
                pre_cfg = cfg.get("pretrain", {})
                for seed in seeds:
                    for method in methods:
                        save_weight = f"{pre_cfg.get('save_prefix','pretrain')}_{method}_s{seed}"
                        base_args = dict_to_args(pre_cfg.get("args", {}), pre_cfg.get("flags", []))
                        method_args = dict_to_args(build_method_args(method, growth_cfg))
                        cmd = [
                            PYTHON, pre_cfg.get("script", "trainer/train_pretrain.py"),
                            "--save_dir", resolve_path(paths.get("out_dir", "out")),
                            "--save_weight", save_weight,
                            "--data_path", resolve_path(paths.get("pretrain_data", "")),
                            "--seed", str(seed)
                        ] + base_args + method_args
                        if not run_cmd(cmd, log_path, env=env, retries=0, dry_run=args.dry_run):
                            mark_fail("train_pretrain", f"{save_weight} failed")
        maybe_cleanup("train_pretrain")

    # ========== Stage: train_sft ==========
    if "train_sft" in stages and cfg.get("sft", {}).get("enabled", False):
        if disk_guard("train_sft"):
            if not ensure_dataset_files(dataset_cfg, "train_sft", log_path, env, args.dry_run, retries=retry_download):
                mark_fail("train_sft", "missing sft data")
            else:
                sft_cfg = cfg.get("sft", {})
                for seed in seeds:
                    for method in methods:
                        save_weight = f"{sft_cfg.get('save_prefix','sft')}_{method}_s{seed}"
                        base_args = dict_to_args(sft_cfg.get("args", {}), sft_cfg.get("flags", []))
                        method_args = dict_to_args(build_method_args(method, growth_cfg))
                        from_weight_mode = sft_cfg.get("from_weight_mode", "fixed")
                        if from_weight_mode == "match_pretrain":
                            from_weight = f"{cfg.get('pretrain', {}).get('save_prefix','pretrain')}_{method}_s{seed}"
                        else:
                            from_weight = sft_cfg.get("from_weight", "pretrain")
                        cmd = [
                            PYTHON, sft_cfg.get("script", "trainer/train_full_sft.py"),
                            "--save_dir", resolve_path(paths.get("out_dir", "out")),
                            "--save_weight", save_weight,
                            "--data_path", resolve_path(paths.get("sft_data", "")),
                            "--from_weight", from_weight,
                            "--seed", str(seed)
                        ] + base_args + method_args
                        if not run_cmd(cmd, log_path, env=env, retries=0, dry_run=args.dry_run):
                            mark_fail("train_sft", f"{save_weight} failed")
        maybe_cleanup("train_sft")

    # ========== Stage: eval_ppl ==========
    if "eval_ppl" in stages:
        if disk_guard("eval_ppl"):
            if not ensure_dataset_files(dataset_cfg, "eval_ppl", log_path, env, args.dry_run, retries=retry_download):
                mark_fail("eval_ppl", "missing val/pretrain data")
            else:
                eval_cfg = cfg.get("eval", {})
                targets = eval_cfg.get("targets") or [eval_cfg.get("target", "pretrain")]
                for target in targets:
                    target_cfg = cfg.get(target, {})
                    prefix = target_cfg.get("save_prefix", target)
                    use_moe = int(target_cfg.get("args", {}).get("use_moe", 0))
                    hidden_size = int(target_cfg.get("args", {}).get("hidden_size", 512))
                    for seed in seeds:
                        for method in methods:
                            weight = f"{prefix}_{method}_s{seed}"
                            if not os.path.exists(weight_path(weight, hidden_size, use_moe, paths.get("out_dir", "out"))):
                                mark_fail("eval_ppl", f"missing weight {weight}")
                                continue
                            out_dir = resolve_path(paths.get("eval_runs_ppl", "eval_runs_ppl"))
                            ensure_dir(out_dir)
                            out_path = os.path.join(out_dir, f"{weight}.json")
                            cmd = [
                                PYTHON, "scripts/eval_ppl.py",
                                "--weight", weight,
                                "--save_dir", resolve_path(paths.get("out_dir", "out")),
                                "--data_path", resolve_path(paths.get("val_data", "")),
                                "--hidden_size", str(hidden_size),
                                "--max_seq_len", str(eval_cfg.get("max_seq_len", 340)),
                                "--batch_size", str(eval_cfg.get("batch_size", 8)),
                                "--max_samples", str(eval_cfg.get("max_samples", 0)),
                                "--method", method,
                                "--out_path", out_path
                            ]
                            if not run_cmd(cmd, log_path, env=env, dry_run=args.dry_run):
                                mark_fail("eval_ppl", f"{weight} failed")
        maybe_cleanup("eval_ppl")

    # ========== Stage: eval_prompts ==========
    if "eval_prompts" in stages:
        if disk_guard("eval_prompts"):
            prompts = resolve_path(paths.get("prompts", ""))
            if not os.path.exists(prompts):
                mark_fail("eval_prompts", "missing prompts file")
            else:
                eval_cfg = cfg.get("eval", {})
                targets = eval_cfg.get("targets") or [eval_cfg.get("target", "pretrain")]
                for target in targets:
                    target_cfg = cfg.get(target, {})
                    prefix = target_cfg.get("save_prefix", target)
                    use_moe = int(target_cfg.get("args", {}).get("use_moe", 0))
                    hidden_size = int(target_cfg.get("args", {}).get("hidden_size", 512))
                    for seed in seeds:
                        for method in methods:
                            weight = f"{prefix}_{method}_s{seed}"
                            if not os.path.exists(weight_path(weight, hidden_size, use_moe, paths.get("out_dir", "out"))):
                                mark_fail("eval_prompts", f"missing weight {weight}")
                                continue
                            cmd = [
                                PYTHON, "scripts/eval_fixed_prompts.py",
                                "--weight", weight,
                                "--save_dir", resolve_path(paths.get("out_dir", "out")),
                                "--hidden_size", str(hidden_size),
                                "--prompts_file", prompts,
                                "--out_dir", resolve_path(paths.get("eval_runs", "eval_runs")),
                                "--config", resolve_path(paths.get("eval_config", "eval/eval_config.json")),
                                "--run_name", weight
                            ]
                            if not run_cmd(cmd, log_path, env=env, dry_run=args.dry_run):
                                mark_fail("eval_prompts", f"{weight} failed")
        maybe_cleanup("eval_prompts")

    # ========== Stage: aggregate ==========
    if "aggregate" in stages:
        if disk_guard("aggregate"):
            summary_csv = resolve_path(paths.get("summary_csv", "eval/summary_ppl.csv"))
            cmd = [PYTHON, "scripts/aggregate_eval.py", "--inputs", resolve_path(paths.get("eval_runs_ppl", "eval_runs_ppl")), "--out", summary_csv]
            if not run_cmd(cmd, log_path, env=env, dry_run=args.dry_run):
                mark_fail("aggregate", "aggregate failed")
        maybe_cleanup("aggregate")

    # ========== Stage: plot ==========
    if "plot" in stages:
        if disk_guard("plot"):
            summary_csv = resolve_path(paths.get("summary_csv", "eval/summary_ppl.csv"))
            plot_png = resolve_path(paths.get("plot_png", "eval/plot_ppl.png"))
            cmd = [PYTHON, "scripts/plot_growth.py", "--csv", summary_csv, "--x", "weight", "--y", "ppl", "--out", plot_png]
            if not run_cmd(cmd, log_path, env=env, dry_run=args.dry_run):
                mark_fail("plot", "plot failed")
        maybe_cleanup("plot")

    # ========== Stage: cleanup ==========
    if "cleanup" in stages:
        cleanup_paths(runtime.get("cleanup_paths", []))

    summary = "\n".join(failures) if failures else "OK"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n[SUMMARY]\n")
        f.write(summary + "\n")
    print("[SUMMARY]", summary)


if __name__ == "__main__":
    main()
