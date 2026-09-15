"""MoE 分发的性能基准 / MoE dispatch benchmark. 自包含，不引入任何新依赖。

  --mode sync-probe   测本机一次 device→host 同步的往返开销
  --mode layer        单个 MoE 层：当前实现 vs 改动前的掩码写法

为什么要先测同步开销：掩码写法每层要做 3E 次主机往返，一次往返的代价由主机 CPU/PCIe
延迟决定，而不是 GPU。我实测过从裸机主机的几微秒到 WSL2 笔记本的约 120 微秒，
所以同一份基准在同型号 GPU、不同主机上可以相差 2-3 倍。先看自己机器的往返开销，
再看下面的加速比，才知道该期待哪一端。

计时协议：≥10 次预热，计时区间前后 torch.cuda.synchronize()，取中位数并报告 min/max。
"""
import argparse
import os
import statistics
import sys
import time

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from model.model_minimind import MiniMindConfig, MOEFeedForward
from scripts.test_moe_sorted import reference_masked_forward


def timed(fn, warmup=10, reps=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return {'median': statistics.median(samples), 'min': min(samples), 'max': max(samples)}


def mode_sync_probe(args):
    flag = torch.zeros(1, dtype=torch.bool, device='cuda')
    x = torch.randn(1024, 1024, device='cuda')

    def probe(fn, n):
        for _ in range(20): fn()
        torch.cuda.synchronize()
        s = []
        for _ in range(args.reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n): fn()
            torch.cuda.synchronize()
            s.append((time.perf_counter() - t0) / n)
        return statistics.median(s)

    launch = probe(lambda: x.add_(0.0), 200)
    rt = probe(lambda: flag.any().item(), 100)
    print(f'  kernel launch (无同步) : {launch * 1e6:8.2f} us')
    print(f'  device->host 往返      : {rt * 1e6:8.2f} us   <- 掩码写法每层要做 3E 次')
    e, l = args.experts[0], args.num_hidden_layers
    print(f'\n  按 3 x {e} experts x {l} layers 估算，仅路由簿记的主机开销下限约 '
          f'{3 * e * l * rt * 1e3:.2f} ms/前向')


def mode_layer(args):
    print(f'{"E":>4}  {"masked (改动前)":>18}  {"sorted (当前)":>18}  {"加速":>7}  {"峰值显存":>16}')
    for e in args.experts:
        torch.manual_seed(0)
        mod = MOEFeedForward(MiniMindConfig(
            hidden_size=args.hidden_size, use_moe=True, num_experts=e,
            num_experts_per_tok=args.top_k, moe_intermediate_size=args.hidden_size * 2)).cuda().train()
        x = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device='cuda')

        def run(fn):
            def step():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    fn(x).sum().backward()
                mod.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            r = timed(step, args.warmup, args.reps)
            r['peak'] = torch.cuda.max_memory_allocated() / 1e6
            return r

        try:
            a = run(lambda t: reference_masked_forward(mod, t))
            b = run(mod)
        except torch.cuda.OutOfMemoryError:
            print(f'{e:>4}  OOM'); continue
        print(f'{e:>4}  {a["median"] * 1e3:8.2f} ms [{a["min"] * 1e3:6.2f},{a["max"] * 1e3:6.2f}]  '
              f'{b["median"] * 1e3:8.2f} ms [{b["min"] * 1e3:6.2f},{b["max"] * 1e3:6.2f}]  '
              f'{a["median"] / b["median"]:6.2f}x  {a["peak"]:7.0f} -> {b["peak"]:5.0f} MB')
        del mod; torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='sync-probe', choices=['sync-probe', 'layer'])
    p.add_argument('--hidden_size', default=768, type=int)
    p.add_argument('--num_hidden_layers', default=8, type=int)
    p.add_argument('--batch_size', default=8, type=int)
    p.add_argument('--seq_len', default=512, type=int)
    p.add_argument('--top_k', default=1, type=int)
    p.add_argument('--experts', default='2,4,8,16', type=lambda s: [int(x) for x in s.split(',')])
    p.add_argument('--warmup', default=10, type=int)
    p.add_argument('--reps', default=20, type=int)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print('本脚本需要 CUDA。/ This benchmark requires CUDA.'); sys.exit(1)
    print(f'torch {torch.__version__}  {torch.cuda.get_device_name(0)}  '
          f'sm_{"".join(map(str, torch.cuda.get_device_capability(0)))}\n')
    {'sync-probe': mode_sync_probe, 'layer': mode_layer}[args.mode](args)


if __name__ == '__main__':
    main()
