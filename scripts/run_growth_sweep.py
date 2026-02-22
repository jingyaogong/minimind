import argparse
import shlex
import subprocess


def build_commands(script, prefix, base_args, seeds):
    runs = [
        ("baseline", "--neuron_growth 0"),
        ("random", "--neuron_growth 1 --grow_method random"),
        ("grad", "--neuron_growth 1 --grow_method act_grad --grow_score_alpha 0.0 --grow_score_beta 1.0"),
        ("act", "--neuron_growth 1 --grow_method act_grad --grow_score_alpha 1.0 --grow_score_beta 0.0"),
        ("actgrad", "--neuron_growth 1 --grow_method act_grad --grow_score_alpha 1.0 --grow_score_beta 1.0"),
    ]
    cmds = []
    for seed in seeds:
        for name, extra in runs:
            save_weight = f"{prefix}_{name}_s{seed}"
            cmd = f"python {script} --save_weight {save_weight} --seed {seed} {base_args} {extra}".strip()
            cmds.append(cmd)
    return cmds


def main():
    parser = argparse.ArgumentParser(description="批量运行动态神经元生长对照实验")
    parser.add_argument("--script", default="trainer/train_pretrain.py", type=str, help="训练脚本路径")
    parser.add_argument("--prefix", default="exp", type=str, help="save_weight 前缀")
    parser.add_argument("--base_args", default="", type=str, help="统一附加参数")
    parser.add_argument("--seeds", default="42", type=str, help="随机种子列表（逗号分隔）")
    parser.add_argument("--run", action="store_true", help="实际执行（不加则只打印命令）")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cmds = build_commands(args.script, args.prefix, args.base_args, seeds)
    for c in cmds:
        print(c)
        if args.run:
            subprocess.run(shlex.split(c), check=True)


if __name__ == "__main__":
    main()
