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


def _is_qat_state_dict(sd):
    """QAT ckpt 会带 act_fq.running_max / act_fq.step buffer——用它判定要不要先 wrap。"""
    return any('act_fq.running_max' in k for k in sd.keys())


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

    is_qat_ckpt = _is_qat_state_dict(state_dict)
    apply_qat = bool(args.apply_qat) or is_qat_ckpt

    if apply_qat:
        from model.quant import prepare_qat, bake_quantized_weights
        skip_patterns = tuple(s.strip() for s in args.quant_skip.split(',') if s.strip())
        if is_qat_ckpt:
            # QAT ckpt 已含 buffer——先 wrap 出对应 module，再整体 load
            replaced = prepare_qat(model, w_bits=args.w_bits, a_bits=args.a_bits,
                                   skip_patterns=skip_patterns)
            model.load_state_dict(state_dict, strict=False)
            print(f'[QAT] loaded QAT ckpt, wrapped {replaced} Linear modules')
        else:
            # fp ckpt——先 load 权重，再 wrap（QATLinear.from_float 保留原 weight tensor）
            model.load_state_dict(state_dict, strict=False)
            replaced = prepare_qat(model, w_bits=args.w_bits, a_bits=args.a_bits,
                                   skip_patterns=skip_patterns)
            print(f'[QAT] wrapped fp ckpt with {replaced} QATLinear (observer un-calibrated)')

        model = model.half().to(args.device)

        if not is_qat_ckpt and args.qat_calibrate > 0:
            # 对未训练过的 QAT 模型：observer running_max 全 0，直接 eval 会 scale ~= 0 全饱和
            # 跑一段 train() 模式的 forward 让 EMA 收敛，再切 eval
            _calibrate_observer(model, args)

        if args.qat_bake:
            print('[QAT] baking fake-quantized weights in-place (simulates deployed int8)')
            bake_quantized_weights(model)
    else:
        # strict=False — tied embeddings have duplicate keys; load handles via shared storage
        model.load_state_dict(state_dict, strict=False)
        model = model.half().to(args.device)

    return model.eval(), cfg


@torch.no_grad()
def _calibrate_observer(model, args):
    """在 train() 模式下跑 N 个 forward 把 ActFakeQuant.running_max EMA 跑起来。

    ActFakeQuant.forward 里 self.training 分支才会更新 EMA + 记 step；observer_steps
    warmup 期内也不做 fake-quant，只统计。这里跑够 observer_steps 步保证之后 eval
    切进真正 fake-quant 分支。
    """
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    model.train()
    fed = 0
    for text in iter_holdout(args):
        if not text or len(text) < 10:
            continue
        body = tokenizer(text, max_length=args.max_seq_len - 2,
                         truncation=True, add_special_tokens=False).input_ids
        tokens = [bos_id] + body + [eos_id]
        if len(tokens) < 2:
            continue
        ids = torch.tensor([tokens], device=args.device)
        model(input_ids=ids)
        fed += 1
        if fed >= args.qat_calibrate:
            break
    model.eval()
    print(f'[QAT] calibration done ({fed} forwards; observer_steps warmup = {args.qat_calibrate})')


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

    # ---- QAT eval options ----
    parser.add_argument('--apply_qat', type=int, default=0, choices=[0, 1],
                        help='对 fp ckpt 也用 QATLinear 包一层，测纯量化噪声；QAT ckpt 会自动识别')
    parser.add_argument('--w_bits', type=int, default=8)
    parser.add_argument('--a_bits', type=int, default=8)
    parser.add_argument('--qat_calibrate', type=int, default=200,
                        help='fp ckpt + apply_qat 时的 observer 预热步数；QAT ckpt 时忽略')
    parser.add_argument('--qat_bake', type=int, default=0, choices=[0, 1],
                        help='把 fake-quantized 权重 in-place 写回 weight（模拟真实 int8 部署）')
    parser.add_argument('--quant_skip', type=str, default='lm_head,mlp.gate',
                        help='要跳过量化的 Linear 名（逗号分隔），需和训练一致')
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
