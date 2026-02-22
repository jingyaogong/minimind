import os
import sys
import json
import shlex
import argparse
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = sys.executable


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def run_cmd(cmd, run=False):
    print("[CMD]", " ".join(cmd))
    if run:
        subprocess.run(cmd, check=True, cwd=ROOT)


def ensure_exists(path, strict=False):
    if os.path.exists(os.path.join(ROOT, path)):
        return True
    msg = f"[WARN] Missing path: {path}"
    if strict:
        raise FileNotFoundError(msg)
    print(msg)
    return False


def build_method_args(method, growth_cfg):
    if method == "baseline":
        return {"neuron_growth": 0}
    if method == "random":
        return {
            "neuron_growth": 1,
            "grow_method": "random",
            **growth_cfg
        }
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


def main():
    parser = argparse.ArgumentParser(description="MiniMind 论文实验一键流水线")
    parser.add_argument("--config", default="eval/pipeline_config.json", type=str, help="配置文件路径")
    parser.add_argument("--stages", default="", type=str, help="仅运行的阶段（逗号分隔）")
    parser.add_argument("--methods", default="", type=str, help="方法列表（逗号分隔）")
    parser.add_argument("--seeds", default="", type=str, help="随机种子列表（逗号分隔）")
    parser.add_argument("--run", action="store_true", help="实际执行（不加则只打印命令）")
    parser.add_argument("--strict", action="store_true", help="缺少数据路径时直接报错")
    args = parser.parse_args()

    cfg = load_config(os.path.join(ROOT, args.config))

    stages = cfg.get("stages", [])
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    methods = cfg.get("methods", ["baseline"])
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    seeds = cfg.get("seeds", [42])
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    paths = cfg.get("paths", {})
    growth_cfg = cfg.get("growth", {})

    # ========== Stage: make_val ==========
    if "make_val" in stages:
        if ensure_exists(paths.get("pretrain_data", ""), strict=args.strict):
            cmd = [PYTHON, "scripts/make_val_split.py",
                   "--data_path", paths["pretrain_data"],
                   "--out_path", paths["val_data"],
                   "--val_size", "2000",
                   "--seed", "42"]
            run_cmd(cmd, run=args.run)
        else:
            print("[SKIP] make_val (missing pretrain_data)")

    # ========== Stage: train_pretrain ==========
    if "train_pretrain" in stages:
        if ensure_exists(paths.get("pretrain_data", ""), strict=args.strict):
            pre_cfg = cfg.get("pretrain", {})
            for seed in seeds:
                for method in methods:
                    save_weight = f"{pre_cfg.get('save_prefix','pretrain')}_{method}_s{seed}"
                    base_args = dict_to_args(pre_cfg.get("args", {}), pre_cfg.get("flags", []))
                    method_args = dict_to_args(build_method_args(method, growth_cfg))
                    cmd = [PYTHON, pre_cfg.get("script", "trainer/train_pretrain.py"),
                           "--save_dir", paths.get("out_dir", "out"),
                           "--save_weight", save_weight,
                           "--data_path", paths.get("pretrain_data"),
                           "--seed", str(seed)] + base_args + method_args
                    run_cmd(cmd, run=args.run)
        else:
            print("[SKIP] train_pretrain (missing pretrain_data)")

    # ========== Stage: train_sft ==========
    if "train_sft" in stages and cfg.get("sft", {}).get("enabled", False):
        if ensure_exists(paths.get("sft_data", ""), strict=args.strict):
            sft_cfg = cfg.get("sft", {})
            for seed in seeds:
                for method in methods:
                    save_weight = f"{sft_cfg.get('save_prefix','sft')}_{method}_s{seed}"
                    base_args = dict_to_args(sft_cfg.get("args", {}), sft_cfg.get("flags", []))
                    method_args = dict_to_args(build_method_args(method, growth_cfg))
                    # from_weight 选择
                    from_weight_mode = sft_cfg.get("from_weight_mode", "fixed")
                    if from_weight_mode == "match_pretrain":
                        from_weight = f"{cfg.get('pretrain', {}).get('save_prefix','pretrain')}_{method}_s{seed}"
                    else:
                        from_weight = sft_cfg.get("from_weight", "pretrain")
                    cmd = [PYTHON, sft_cfg.get("script", "trainer/train_full_sft.py"),
                           "--save_dir", paths.get("out_dir", "out"),
                           "--save_weight", save_weight,
                           "--data_path", paths.get("sft_data"),
                           "--from_weight", from_weight,
                           "--seed", str(seed)] + base_args + method_args
                    run_cmd(cmd, run=args.run)
        else:
            print("[SKIP] train_sft (missing sft_data)")

    # ========== Stage: eval_ppl ==========
    if "eval_ppl" in stages:
        val_path = paths.get("val_data", "")
        if ensure_exists(val_path, strict=args.strict):
            eval_cfg = cfg.get("eval", {})
            target = eval_cfg.get("target", "pretrain")
            prefix = cfg.get(target, {}).get("save_prefix", target)
            eval_runs_ppl = paths.get("eval_runs_ppl", "eval_runs_ppl")
            for seed in seeds:
                for method in methods:
                    weight = f"{prefix}_{method}_s{seed}"
                    out_path = os.path.join(eval_runs_ppl, f"{weight}.json")
                    cmd = [PYTHON, "scripts/eval_ppl.py",
                           "--weight", weight,
                           "--save_dir", paths.get("out_dir", "out"),
                           "--data_path", val_path,
                           "--max_seq_len", str(eval_cfg.get("max_seq_len", 340)),
                           "--batch_size", str(eval_cfg.get("batch_size", 8)),
                           "--max_samples", str(eval_cfg.get("max_samples", 0)),
                           "--method", method,
                           "--out_path", out_path]
                    run_cmd(cmd, run=args.run)
        else:
            print("[SKIP] eval_ppl (missing val_data)")

    # ========== Stage: eval_prompts ==========
    if "eval_prompts" in stages:
        prompts = paths.get("prompts", "")
        if ensure_exists(prompts, strict=args.strict):
            eval_cfg = cfg.get("eval", {})
            target = eval_cfg.get("target", "pretrain")
            prefix = cfg.get(target, {}).get("save_prefix", target)
            for seed in seeds:
                for method in methods:
                    weight = f"{prefix}_{method}_s{seed}"
                    cmd = [PYTHON, "scripts/eval_fixed_prompts.py",
                           "--weight", weight,
                           "--save_dir", paths.get("out_dir", "out"),
                           "--prompts_file", prompts,
                           "--out_dir", paths.get("eval_runs", "eval_runs"),
                           "--config", paths.get("eval_config", "eval/eval_config.json"),
                           "--run_name", weight]
                    run_cmd(cmd, run=args.run)
        else:
            print("[SKIP] eval_prompts (missing prompts)")

    # ========== Stage: aggregate ==========
    if "aggregate" in stages:
        eval_runs_ppl = paths.get("eval_runs_ppl", "eval_runs_ppl")
        summary_csv = paths.get("summary_csv", "eval/summary_ppl.csv")
        cmd = [PYTHON, "scripts/aggregate_eval.py", "--inputs", eval_runs_ppl, "--out", summary_csv]
        run_cmd(cmd, run=args.run)

    # ========== Stage: plot ==========
    if "plot" in stages:
        summary_csv = paths.get("summary_csv", "eval/summary_ppl.csv")
        plot_png = paths.get("plot_png", "eval/plot_ppl.png")
        cmd = [PYTHON, "scripts/plot_growth.py", "--csv", summary_csv, "--x", "weight", "--y", "ppl", "--out", plot_png]
        run_cmd(cmd, run=args.run)


if __name__ == "__main__":
    main()
