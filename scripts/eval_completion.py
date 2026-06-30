"""
Qualitative eval of a MiniMind pretrain (or post-pretrain) checkpoint by sampling
text continuations from a fixed prompt set spanning multiple domains.

For pretrain (base) models — does NOT use chat template. Use eval_llm.py for SFT models.

Usage:
    # Default: 64M pretrain ckpt
    python eval_completion.py

    # 1B ckpt
    python eval_completion.py --hidden_size 2048 --num_hidden_layers 20 --weight pretrain

    # Custom prompt
    python eval_completion.py --prompt "今天天气不错，我打算"

    # Save outputs to markdown for design doc
    python eval_completion.py --output_md eval_results.md

What to look for:
    ✓ Grammatically coherent Chinese, no repetition loops
    ✓ Stays roughly on topic of the prefix
    ✓ Variety across runs (different temperature should give different outputs)
    ✗ Repetitive: "的的的的" / 同一句重复 → 训练或解码有问题
    ✗ 全部生成都一模一样 → 模型坍缩 / temperature 太低
    ✗ 乱码 / 非中文片段（如果是纯中文 prefix）→ tokenizer 问题
"""
import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import warnings
import time
import torch
from transformers import AutoTokenizer, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

warnings.filterwarnings('ignore')


DEFAULT_PROMPTS = [
    ("叙事",     "在很久很久以前，有一个小村庄，"),
    ("日常",     "今天天气不错，我打算"),
    ("技术",     "Python 是一种编程语言，它的特点是"),
    ("事实",     "中华人民共和国成立于"),
    ("代码",     "def fibonacci(n):\n    if n < 2:"),
    ("科学",     "牛顿第一定律说的是"),
    ("文学",     "李白是唐代著名的"),
    ("数学",     "二的十次方等于"),
    ("常识",     "水的沸点在标准大气压下是"),
    ("地理",     "中国的首都是"),
]


def load_model(args):
    cfg = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model = MiniMindForCausalLM(cfg)
    moe_suffix = '_moe' if args.use_moe else ''
    ckp = f'{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.half().eval().to(args.device)
    return model, cfg


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="MiniMind Pretrain Completion Eval")
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--use_moe', type=int, default=0, choices=[0, 1])
    parser.add_argument('--save_dir', type=str, default='../out')
    parser.add_argument('--weight', type=str, default='pretrain')
    parser.add_argument('--tokenizer_path', type=str, default='../model')
    parser.add_argument('--prompt', type=str, default=None,
                        help='单条自定义 prompt；不指定则跑 DEFAULT_PROMPTS 全集')
    parser.add_argument('--max_new_tokens', type=int, default=80)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--repetition_penalty', type=float, default=1.1)
    parser.add_argument('--num_samples_per_prompt', type=int, default=1,
                        help='每个 prompt 采样几次（验证多样性）')
    parser.add_argument('--output_md', type=str, default=None,
                        help='输出 markdown 报告路径（可选）')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f'[load] ckpt={args.save_dir}/{args.weight}_{args.hidden_size}.pth')
    model, cfg = load_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    prompts = [('custom', args.prompt)] if args.prompt else DEFAULT_PROMPTS

    md_lines = []
    md_lines.append(f'# Completion Eval: `{args.weight}_{args.hidden_size}.pth`')
    md_lines.append(f'temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, '
                    f'rep_penalty={args.repetition_penalty}, max_new={args.max_new_tokens}\n')

    for label, prompt in prompts:
        print(f'\n{"=" * 60}')
        print(f'[{label}] Prompt: {prompt!r}')
        md_lines.append(f'## [{label}] `{prompt}`\n')
        for sample_i in range(args.num_samples_per_prompt):
            print(f'\n--- Sample {sample_i + 1} ---')
            # 必须加 BOS：训练时所有样本以 [BOS] 开头，eval 时不加 BOS 模型会产生 degenerate
            # 输出（"的的的"/"是是是" 重复），因为它从没见过"无 BOS 开头"的序列
            body = tokenizer(prompt, add_special_tokens=False).input_ids
            ids = torch.tensor([[tokenizer.bos_token_id] + body], device=args.device)
            t0 = time.time()
            output_ids = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
            )
            elapsed = time.time() - t0
            text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
            continuation = text[len(prompt):]
            print(continuation)
            new_tokens = output_ids.shape[1] - ids.shape[1]
            print(f'  [{new_tokens} new tokens in {elapsed:.2f}s = {new_tokens/elapsed:.1f} tok/s]')
            md_lines.append(f'**Sample {sample_i + 1}**: {continuation.strip()}\n')
        md_lines.append('')

    if args.output_md:
        with open(args.output_md, 'w') as f:
            f.write('\n'.join(md_lines))
        print(f'\n[saved] {args.output_md}')


if __name__ == "__main__":
    main()
