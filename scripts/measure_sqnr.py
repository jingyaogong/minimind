"""
Measure SQNR (Signal-to-Quantization-Noise Ratio) of MiniMind checkpoints.

三个维度：
1. Weight SQNR: 对每层 Linear.weight 应用 W8/W4 fake-quant，算 SQNR
   - Pretrain vs QAT-trained: QAT 训练应把 W SQNR 推高（权重朝量化友好区漂）
2. Activation SQNR: 用 forward hook 抓每层输入 activation, 用 running_max 量化后算 SQNR
   - 需要真实数据（否则激活分布不代表推理时）
3. Bit-width sweep: 对 pretrain 权重扫 W2/W4/W8/W16 看 SQNR ~ bits 曲线（应该 +6dB/bit）

Usage:
    cd scripts && python measure_sqnr.py                     # 默认：W SQNR (pretrain vs qat) + bit sweep
    python measure_sqnr.py --act 1                           # 加上 activation SQNR (需要 GPU + 数据)
    python measure_sqnr.py --act 1 --n_samples 32            # activation 用 N samples 校准
"""
import os
import sys
__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import warnings
import json
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.quant import WeightFakeQuant, prepare_qat, DEFAULT_SKIP_PATTERNS

warnings.filterwarnings('ignore')


def sqnr_db(x: torch.Tensor, x_q: torch.Tensor) -> float:
    """SQNR = 10 log10(||x||² / ||x - x_q||²)，用 fp64 算防止 fp16 数值下溢。"""
    x, x_q = x.double(), x_q.double()
    signal_pow = (x ** 2).mean()
    noise_pow = ((x - x_q) ** 2).mean().clamp_min(1e-24)
    return (10.0 * torch.log10(signal_pow / noise_pow)).item()


def is_quantizable_linear_key(k: str, skip_patterns=DEFAULT_SKIP_PATTERNS) -> bool:
    """判断 state_dict 的 key 是否对应我们会量化的 Linear.weight。"""
    if not k.endswith('.weight'):
        return False
    # 排除 embedding / norm / lm_head / MoE 路由（和 prepare_qat 的 skip 一致）
    if 'embed_tokens' in k or 'lm_head' in k or 'norm' in k:
        return False
    parent = k[:-len('.weight')]
    for pat in skip_patterns:
        if parent == pat or parent.endswith('.' + pat):
            return False
    return True


def measure_weight_sqnr(pretrain_sd, qat_sd, bits=8):
    """对每层 Linear.weight 算 W SQNR，返回 dict {layer_name: (sqnr_pretrain, sqnr_qat, delta)}。"""
    wfq = WeightFakeQuant(bits=bits)
    results = {}
    for k in pretrain_sd.keys():
        if not is_quantizable_linear_key(k):
            continue
        w_pt = pretrain_sd[k].float()
        w_pt_q = wfq(w_pt)
        sqnr_pt = sqnr_db(w_pt, w_pt_q)
        rec = {'pretrain': sqnr_pt}
        if k in qat_sd:
            w_q = qat_sd[k].float()
            w_q_q = wfq(w_q)
            sqnr_q = sqnr_db(w_q, w_q_q)
            rec['qat'] = sqnr_q
            rec['delta'] = sqnr_q - sqnr_pt
        results[k] = rec
    return results


def measure_bit_sweep(pretrain_sd, bit_list=(2, 3, 4, 6, 8, 12)):
    """对每层权重扫多个 bit，验证 ~6 dB/bit 规律。返回 dict {bits: mean_sqnr_across_layers}。"""
    out = {}
    for bits in bit_list:
        wfq = WeightFakeQuant(bits=bits)
        sqnrs = []
        for k in pretrain_sd.keys():
            if not is_quantizable_linear_key(k):
                continue
            w = pretrain_sd[k].float()
            sqnrs.append(sqnr_db(w, wfq(w)))
        out[bits] = sum(sqnrs) / len(sqnrs)
    return out


@torch.no_grad()
def measure_activation_sqnr(cfg, qat_ckpt_path, tokenizer_path, data_path, n_samples, device='cuda'):
    """
    跑真实数据，用 forward hook 抓每个 QATLinear 的输入激活，
    对比 fp 激活 vs act_fq(fp激活) 的 SQNR。使用 QAT ckpt 里训好的 running_max。
    """
    model = MiniMindForCausalLM(cfg)
    prepare_qat(model, w_bits=8, a_bits=8)
    sd = torch.load(qat_ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(sd, strict=False)
    model = model.half().eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id

    # 收集每个 QATLinear 的激活 SQNR（累积 sum + count 便于跨样本求平均）
    from model.quant import QATLinear
    stats = {}  # name -> {sum_sig_pow, sum_noise_pow, count}
    hooks = []

    def make_hook(name, act_fq):
        def hook(mod, inp, out):
            x = inp[0].detach()
            x_q = act_fq(x.clone())  # 现用 QAT 训好的 running_max 量化
            xd, xqd = x.double(), x_q.double()
            s = (xd ** 2).sum().item()
            n = ((xd - xqd) ** 2).sum().item()
            if name not in stats:
                stats[name] = {'sig': 0.0, 'noise': 0.0, 'nel': 0}
            stats[name]['sig'] += s
            stats[name]['noise'] += n
            stats[name]['nel'] += x.numel()
        return hook

    for name, m in model.named_modules():
        if isinstance(m, QATLinear):
            hooks.append(m.register_forward_hook(make_hook(name, m.act_fq)))

    # 走 N 条 pretrain 数据
    fed = 0
    with open(data_path) as f:
        for line in f:
            d = json.loads(line)
            text = d.get('text') or d.get('content') or ''
            if not text or len(text) < 10:
                continue
            body = tokenizer(text, max_length=510, truncation=True,
                             add_special_tokens=False).input_ids
            tokens = [bos_id] + body + [eos_id]
            ids = torch.tensor([tokens], device=device)
            model(input_ids=ids)
            fed += 1
            if fed >= n_samples:
                break

    for h in hooks:
        h.remove()

    # 转成 SQNR
    result = {}
    for name, s in stats.items():
        sqnr = 10.0 * torch.log10(torch.tensor(s['sig'] / max(s['noise'], 1e-24))).item()
        result[name] = sqnr
    return result, fed


def _fmt_layer_stats(results, val_key='pretrain'):
    """紧凑打印每层 SQNR（截 layer name 到 self_attn.q_proj 之类）。"""
    values = [(k.replace('model.layers.', 'L').replace('.weight', ''), v)
              for k, v in results.items()]
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--use_moe', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='../out')
    parser.add_argument('--pretrain', type=str, default='pretrain')
    parser.add_argument('--qat', type=str, default='qat')
    parser.add_argument('--tokenizer_path', type=str, default='../model')
    parser.add_argument('--data_path', type=str, default='../dataset/pretrain_t2t.jsonl')
    parser.add_argument('--n_samples', type=int, default=32)
    parser.add_argument('--act', type=int, default=1, choices=[0, 1],
                        help='是否算激活 SQNR（需要 GPU + 数据）')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    moe_suffix = '_moe' if args.use_moe else ''
    pt_path = f'{args.save_dir}/{args.pretrain}_{args.hidden_size}{moe_suffix}.pth'
    qat_path = f'{args.save_dir}/{args.qat}_{args.hidden_size}{moe_suffix}.pth'

    print(f'[load] pretrain: {pt_path}')
    print(f'[load] qat:      {qat_path}')
    pretrain_sd = torch.load(pt_path, map_location='cpu', weights_only=True)
    qat_sd = torch.load(qat_path, map_location='cpu', weights_only=True)

    # =========================================================
    # 1. Weight SQNR: pretrain vs QAT-trained, W8
    # =========================================================
    print(f'\n{"=" * 70}')
    print(f'  Weight SQNR (W8 per-out-channel symmetric)')
    print(f'  higher = weights sit closer to int8 grid points')
    print(f'{"=" * 70}')
    w8_results = measure_weight_sqnr(pretrain_sd, qat_sd, bits=8)

    # 分组：attn (q/k/v/o) vs ffn (gate/up/down) 分别汇总
    def group(k):
        if 'self_attn' in k: return 'attn'
        if 'mlp' in k: return 'ffn'
        return 'other'

    groups = {'attn': [], 'ffn': [], 'other': []}
    for k, r in w8_results.items():
        groups[group(k)].append(r)

    print(f"{'Group':10s} {'#layers':>8s} {'PT SQNR mean':>14s} {'QAT SQNR mean':>14s} {'Δ mean':>10s}")
    for g, rs in groups.items():
        if not rs: continue
        pt_mean = sum(r['pretrain'] for r in rs) / len(rs)
        qat_mean = sum(r.get('qat', 0) for r in rs) / len(rs)
        delta_mean = sum(r.get('delta', 0) for r in rs) / len(rs)
        print(f"{g:10s} {len(rs):>8d} {pt_mean:>13.2f}dB {qat_mean:>13.2f}dB {delta_mean:>+9.2f}dB")

    print(f'\n  Per-layer detail (first 8 layers, all types):')
    print(f'  {"layer":50s} {"PT dB":>8s} {"QAT dB":>8s} {"Δ":>8s}')
    shown = 0
    for k, r in w8_results.items():
        if 'layers.0.' not in k: continue  # 只看第 0 层
        short = k.replace('model.layers.0.', '').replace('.weight', '')
        print(f'  {short:50s} {r["pretrain"]:>7.2f}  {r.get("qat", 0):>7.2f}  {r.get("delta", 0):>+7.2f}')
        shown += 1

    # 排序找最难 / 最容易量化的 3 层
    ranked_pt = sorted(w8_results.items(), key=lambda kv: kv[1]['pretrain'])
    print(f'\n  Hardest to quantize (lowest PT SQNR):')
    for k, r in ranked_pt[:3]:
        print(f'    {k:60s}  {r["pretrain"]:.2f} dB')
    print(f'  Easiest to quantize (highest PT SQNR):')
    for k, r in ranked_pt[-3:]:
        print(f'    {k:60s}  {r["pretrain"]:.2f} dB')

    # =========================================================
    # 2. Bit-width sweep（验证 ~6 dB/bit）
    # =========================================================
    print(f'\n{"=" * 70}')
    print(f'  Bit-width sweep on pretrain weights (avg across {len(w8_results)} Linears)')
    print(f'  theory: SQNR ≈ 6.02 × bits + 1.76 dB (uniform quantizer, full-scale signal)')
    print(f'{"=" * 70}')
    sweep = measure_bit_sweep(pretrain_sd, bit_list=(2, 3, 4, 6, 8, 12))
    prev = None
    print(f"  {'bits':>5s} {'measured':>12s} {'theory (6b+1.76)':>18s} {'Δ vs prev':>12s}")
    for bits, s in sweep.items():
        theory = 6.02 * bits + 1.76
        delta = f"{s - prev:+.2f} dB" if prev is not None else "—"
        print(f"  {bits:>5d} {s:>10.2f}dB {theory:>15.2f}dB {delta:>12s}")
        prev = s

    # =========================================================
    # 3. Activation SQNR（可选，需要 GPU + 数据）
    # =========================================================
    if args.act:
        print(f'\n{"=" * 70}')
        print(f'  Activation SQNR (per-tensor symmetric int8, running_max from QAT training)')
        print(f'  用 {args.n_samples} 条 pretrain 样本走前向，hook 到每个 QATLinear 输入')
        print(f'{"=" * 70}')
        cfg = MiniMindConfig(hidden_size=args.hidden_size,
                             num_hidden_layers=args.num_hidden_layers,
                             use_moe=bool(args.use_moe))
        act_sqnr, n_run = measure_activation_sqnr(
            cfg, qat_path, args.tokenizer_path,
            args.data_path, args.n_samples, args.device,
        )
        print(f'  (evaluated on {n_run} samples)')
        # 按 attn/ffn 分组汇总
        act_groups = {'attn': [], 'ffn': []}
        for name, v in act_sqnr.items():
            if 'self_attn' in name: act_groups['attn'].append(v)
            elif 'mlp' in name: act_groups['ffn'].append(v)
        print(f"  {'Group':10s} {'#Linear':>10s} {'A8 SQNR mean':>15s} {'min':>10s} {'max':>10s}")
        for g, vs in act_groups.items():
            if not vs: continue
            print(f"  {g:10s} {len(vs):>10d} {sum(vs)/len(vs):>13.2f}dB {min(vs):>8.2f}dB {max(vs):>8.2f}dB")
        # 层 0 明细
        print(f'\n  Layer 0 per-Linear activation SQNR:')
        for name, v in act_sqnr.items():
            if 'layers.0.' not in name: continue
            short = name.replace('model.layers.0.', '')
            print(f'    {short:40s}  {v:.2f} dB')
        # 全局最差 3 个
        ranked_act = sorted(act_sqnr.items(), key=lambda kv: kv[1])
        print(f'\n  Worst 3 activation SQNR (量化最难扛的层):')
        for k, v in ranked_act[:3]:
            print(f'    {k:60s}  {v:.2f} dB')

    print(f'\n{"=" * 70}')
    print(f'  参照：W8 通常 40-50 dB / W4 15-25 dB / <20 dB 开始明显掉 PPL')
    print(f'{"=" * 70}')


if __name__ == "__main__":
    main()
