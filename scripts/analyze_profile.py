"""
Parse torch.profiler chrome trace, summarize top ops + backend detection.

Usage:
    # trace_out/ 是 train_pretrain.py --profile_out 指定的目录
    python analyze_profile.py --trace_dir ../profile_out

    # 或指定具体 trace 文件（一个 rank 一个）
    python analyze_profile.py --trace_file ../profile_out/DEVvm54331_XXX.pt.trace.json
"""
import argparse
import glob
import gzip
import json
import os
from collections import defaultdict


# op 名 → bucket 分类规则（按前缀 / substring 匹配）
BUCKET_RULES = [
    ('matmul (GEMM)',        ['aten::mm', 'aten::bmm', 'aten::addmm', 'aten::linear']),
    ('attention (SDPA/FA)',  ['flash', 'scaled_dot_product', 'sdpa', 'attention',
                              'aten::_scaled_dot_product', 'FlashAttn', 'mem_efficient',
                              'triton_flash', 'ampere_fp16_s16816gemm_bf16']),
    ('elementwise (norm/act)', ['rms_norm', 'RMSNorm', 'aten::silu', 'aten::mul', 'aten::add',
                                 'aten::div', 'aten::rsqrt', 'aten::sub', 'element_wise']),
    ('softmax',              ['softmax', 'log_softmax', 'cross_entropy']),
    ('rope',                 ['rope', 'rotary', 'cos', 'sin']),
    ('embedding',            ['embedding', 'aten::embedding']),
    ('dropout',              ['dropout']),
    ('reduce (all-reduce/gather)', ['nccl', 'ncclAllReduce', 'ncclAllGather', 'ncclReduceScatter',
                                     'allreduce', 'AllReduce', 'ProcessGroupNCCL']),
    ('memory (copy/permute/view)', ['aten::to', 'aten::_to_copy', 'aten::contiguous',
                                     'aten::permute', 'aten::view', 'aten::reshape',
                                     'aten::transpose', 'aten::clone', 'Memcpy']),
    ('optimizer',            ['adam', 'AdamW', 'optim', 'clip_grad']),
]

# attention backend 识别（chrome trace event name）
SDPA_BACKENDS = {
    'flash':      ['FlashAttn', 'flash_attn', 'triton_flash', 'ampere_fp16_s16816gemm'],
    'mem_efficient': ['mem_efficient', 'MemEfficient', 'cutlassF'],
    'math':       ['aten::baddbmm', 'aten::softmax'],  # naive fallback
}


def bucket_op(name: str) -> str:
    ln = name.lower()
    for bucket, patterns in BUCKET_RULES:
        for p in patterns:
            if p.lower() in ln:
                return bucket
    return 'other'


def load_trace(path):
    """支持 .json 和 .json.gz。"""
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        data = json.load(f)
    return data.get('traceEvents', data)


def analyze(events):
    # 只统计 CUDA kernel 时间（cat == 'kernel' 或 'cuda_runtime'）
    # 但 torch.profiler 输出的 chrome trace 里 CUDA op 是 name+dur 的 complete event (ph=='X')
    stats = defaultdict(lambda: {'time_us': 0, 'count': 0})
    total_cuda_us = 0
    sdpa_backend_hits = defaultdict(int)

    for e in events:
        if e.get('ph') != 'X':
            continue
        cat = e.get('cat', '')
        name = e.get('name', '')
        dur = e.get('dur', 0)
        # 只关心 GPU kernel + collective + cudaLaunch，跳过 python-level
        if cat not in ('kernel', 'cuda_runtime', 'user_annotation', 'gpu_memcpy', 'gpu_memset', 'cpu_op'):
            continue
        if cat == 'cpu_op' and 'ncclAllReduce' not in name and 'sync' not in name.lower():
            # CPU op 大部分是 dispatch overhead，除了 NCCL / sync 都跳
            continue
        stats[name]['time_us'] += dur
        stats[name]['count'] += 1
        if cat in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            total_cuda_us += dur

        # 探测 attention backend
        for backend, patterns in SDPA_BACKENDS.items():
            for p in patterns:
                if p in name:
                    sdpa_backend_hits[backend] += dur

    return stats, total_cuda_us, sdpa_backend_hits


def format_us(us):
    if us > 1e6: return f'{us/1e6:.2f} s'
    if us > 1e3: return f'{us/1e3:.2f} ms'
    return f'{us:.0f} us'


def report(stats, total_cuda_us, sdpa_backend_hits, top_n=25):
    print(f'\n{"=" * 80}')
    print(f'  Total CUDA time in trace: {format_us(total_cuda_us)}')
    print(f'{"=" * 80}')

    # bucket 汇总
    bucket_time = defaultdict(int)
    bucket_count = defaultdict(int)
    for name, s in stats.items():
        b = bucket_op(name)
        bucket_time[b] += s['time_us']
        bucket_count[b] += s['count']

    print(f'\n  Time by bucket (% of total CUDA):')
    print(f'  {"bucket":40s} {"time":>12s} {"% total":>8s} {"# calls":>8s}')
    for bucket, t in sorted(bucket_time.items(), key=lambda kv: -kv[1]):
        pct = t / max(total_cuda_us, 1) * 100
        print(f'  {bucket:40s} {format_us(t):>12s} {pct:>7.1f}% {bucket_count[bucket]:>8d}')

    # attention backend 探测
    print(f'\n  Attention backend detection:')
    if sdpa_backend_hits:
        winner = max(sdpa_backend_hits.items(), key=lambda kv: kv[1])
        for backend, t in sorted(sdpa_backend_hits.items(), key=lambda kv: -kv[1]):
            marker = ' ← DOMINANT' if backend == winner[0] else ''
            print(f'    {backend:20s}: {format_us(t):>12s}{marker}')
        if winner[0] != 'flash':
            print(f'  ⚠️  Not using FlashAttention. Expected FA2 on A100 for hidden 128/heads 16 → suboptimal.')
            print(f'     Fix: 训练里用 `torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)` context')
    else:
        print(f'    (未识别到 attention kernel——可能 kernel 名 encoded 或未走 SDPA)')

    # top ops 明细
    print(f'\n  Top {top_n} CUDA ops:')
    print(f'  {"rank":>4s} {"time":>12s} {"% total":>8s} {"# calls":>8s}  {"name":s}')
    sorted_ops = sorted(stats.items(), key=lambda kv: -kv[1]['time_us'])
    for i, (name, s) in enumerate(sorted_ops[:top_n], 1):
        pct = s['time_us'] / max(total_cuda_us, 1) * 100
        # 截断长 kernel 名
        display_name = name if len(name) < 80 else name[:77] + '...'
        print(f'  {i:>4d} {format_us(s["time_us"]):>12s} {pct:>7.1f}% {s["count"]:>8d}  {display_name}')

    # 分类建议
    print(f'\n{"=" * 80}')
    print(f'  优化提示（按 bucket 比例）:')
    print(f'{"=" * 80}')
    total = total_cuda_us
    reduce_pct = bucket_time.get('reduce (all-reduce/gather)', 0) / max(total, 1) * 100
    if reduce_pct > 15:
        print(f'  ⚠️  reduce/collective 占 {reduce_pct:.1f}%（>15% 算高）→ 考虑：')
        print(f'      - `no_sync()` 包 gradient accumulation 减少 3× 冗余 all-reduce')
        print(f'      - FSDP: 用 ZeRO-2/3 拆 optimizer state 减少每次 sync 数据量')
    elem_pct = bucket_time.get('elementwise (norm/act)', 0) / max(total, 1) * 100
    if elem_pct > 15:
        print(f'  ⚠️  elementwise 占 {elem_pct:.1f}%（>15% 算高）→ memory bandwidth bound')
        print(f'      - Fused RMSNorm + SiLU + Add kernel（apex 或 triton）省 50%+')
    mem_pct = bucket_time.get('memory (copy/permute/view)', 0) / max(total, 1) * 100
    if mem_pct > 8:
        print(f'  ⚠️  memory movement 占 {mem_pct:.1f}%（>8% 算高）→ 布局问题')
        print(f'      - 检查 permute/transpose/contiguous 频次，可能重复布局转换')
    attn_pct = bucket_time.get('attention (SDPA/FA)', 0) / max(total, 1) * 100
    matmul_pct = bucket_time.get('matmul (GEMM)', 0) / max(total, 1) * 100
    print(f'\n  Matmul + Attention (核心 compute): {matmul_pct + attn_pct:.1f}%')
    print(f'  如果这个数 <50%，多数时间在做非核心 op → 有巨大优化空间')
    print(f'  Llama 3 训练里这个数一般 60-75%')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace_file', type=str, default=None)
    parser.add_argument('--trace_dir', type=str, default='../profile_out')
    parser.add_argument('--top_n', type=int, default=25)
    args = parser.parse_args()

    if args.trace_file:
        files = [args.trace_file]
    else:
        # torch.profiler 每 rank 一个文件；分析 rank 0 即可
        files = sorted(glob.glob(os.path.join(args.trace_dir, '*.pt.trace.json*')))
        if not files:
            print(f'no trace file found in {args.trace_dir}')
            return
        # 只取第一个（一般 rank 0）
        files = files[:1]

    for f in files:
        print(f'\n[loading] {f} ({os.path.getsize(f)/1e6:.1f} MB)')
        events = load_trace(f)
        print(f'[loaded] {len(events):,} events')
        stats, total, sdpa_hits = analyze(events)
        report(stats, total, sdpa_hits, top_n=args.top_n)


if __name__ == "__main__":
    main()
