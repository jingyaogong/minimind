"""
Compute perplexity of a MiniMind checkpoint on a holdout text dataset.

Usage:
    # Default: 64M pretrain ckpt, hash-sampled 1% of pretrain_t2t.jsonl
    python eval_perplexity.py

    # 1B ckpt with explicit holdout
    python eval_perplexity.py --hidden_size 2048 --num_hidden_layers 20 \\
        --weight pretrain --holdout_file ../dataset/holdout.jsonl

    # Compare two ckpts (e.g. before/after QAT)
    python eval_perplexity.py --weight pretrain
    python eval_perplexity.py --weight qat

Reference PPL ranges (rough):
    64M from-scratch on Chinese pretrain holdout:  8-15
    200M:  6-10
    1B:    5-8
    Anything > 30 → training is broken (data leak, lr too high, etc.)
    Anything < 5 → suspect data leak (test set seen during training)
"""
import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import math
import warnings
import time
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

warnings.filterwarnings('ignore')


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
    # strict=False — tied embeddings have duplicate keys; load handles via shared storage
    model.load_state_dict(state_dict, strict=False)
    model = model.half().eval().to(args.device)
    return model, cfg


def iter_holdout(args):
    """Yield text strings from holdout. Either a separate file or hash-sampled from pretrain."""
    if args.holdout_file:
        with open(args.holdout_file) as f:
            for line in f:
                d = json.loads(line)
                yield d.get('text') or d.get('content') or ''
        return
    # Hash-based holdout: take 1/sampling_mod of training file (deterministic, repeatable)
    # NOTE: only meaningful as proxy — model HAS seen these during training.
    # True holdout requires reserving samples BEFORE training (Week 1 trainer fix).
    with open(args.train_file) as f:
        for line in f:
            if hash(line) % args.sampling_mod == 0:
                d = json.loads(line)
                yield d.get('text') or d.get('content') or ''


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="MiniMind Perplexity Eval")
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--use_moe', type=int, default=0, choices=[0, 1])
    parser.add_argument('--save_dir', type=str, default='../out')
    parser.add_argument('--weight', type=str, default='pretrain',
                        help='ckpt 前缀，如 pretrain / full_sft / qat')
    parser.add_argument('--tokenizer_path', type=str, default='../model')
    parser.add_argument('--holdout_file', type=str, default=None,
                        help='独立 holdout jsonl 文件；若为 None 则从 train_file 按 hash 抽样')
    parser.add_argument('--train_file', type=str, default='../dataset/pretrain_t2t.jsonl',
                        help='没 holdout 时从这里 hash 抽样作 proxy holdout')
    parser.add_argument('--sampling_mod', type=int, default=1000,
                        help='hash 抽样模数：1000=取 0.1%, 100=取 1%')
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f'[load] ckpt={args.save_dir}/{args.weight}_{args.hidden_size}.pth')
    model, cfg = load_model(args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    total_loss, total_tokens, n_samples = 0.0, 0, 0
    t0 = time.time()
    # ⚠️ 必须加 BOS/EOS：minimind PretrainDataset 训练时格式是 [BOS] + tokens + [EOS] + pad。
    # eval 时不加 BOS，第一个 token 的 loss 会异常高（模型从来没见过"无 BOS 开头"），
    # 整体 PPL 会虚高 ~4x（实测从 4.7 → 20.2）。
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    for text in iter_holdout(args):
        if not text or len(text) < 10:
            continue
        body = tokenizer(text, max_length=args.max_seq_len - 2,
                         truncation=True, add_special_tokens=False).input_ids
        tokens = [bos_id] + body + [eos_id]
        if len(tokens) < 2:
            continue
        ids = torch.tensor([tokens], device=args.device)
        labels = ids.clone()
        labels[labels == pad_id] = -100
        out = model(input_ids=ids, labels=labels)
        n_tokens = (labels != -100).sum().item() - 1  # 减 1：last token 不算 loss
        if n_tokens <= 0:
            continue
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens
        n_samples += 1
        if n_samples >= args.max_samples:
            break

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    elapsed = time.time() - t0
    print(f'\n{"=" * 60}')
    print(f'Samples evaluated: {n_samples}')
    print(f'Total tokens:      {total_tokens:,}')
    print(f'Avg loss:          {avg_loss:.4f}')
    print(f'Perplexity (PPL):  {ppl:.2f}')
    print(f'Elapsed:           {elapsed:.1f}s ({total_tokens/elapsed:.0f} tokens/s)')
    print(f'{"=" * 60}')

    if args.holdout_file is None:
        print(f'⚠️  Used hash-sampled "holdout" from train_file (model HAS seen these).')
        print(f'   For real eval, reserve a holdout BEFORE training.')


if __name__ == "__main__":
    main()
