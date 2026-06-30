"""
Streaming 下载 SkyPile-150B 子采样，转成 minimind PretrainDataset 兼容格式（{"text": ...}）。

特性：
- Streaming：不全下，边读边写，磁盘只占目标量
- Token 计数：精确控制下载量（按 budget 而非样本数停）
- 断点续传：进程被杀也能从最后写到的行接着下
- 进度日志：每 N 个样本打印进度
- 简易过滤：太短 / 重复字符多的样本跳过

Usage:
    # 默认下 25B tokens（1B 模型 Chinchilla 最优）
    python download_skypile.py

    # 改下载量
    python download_skypile.py --target_tokens 10_000_000_000

    # 改输出路径
    python download_skypile.py --output ../dataset/skypile_25b.jsonl

需要 fwdproxy（HuggingFace 在 allowlist 但 OD 无直接 DNS）：
    export https_proxy=http://fwdproxy:8080
    export http_proxy=http://fwdproxy:8080
"""
import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import time
import warnings
from datasets import load_dataset
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')


def count_tokens_fast(text: str) -> int:
    """快速估算 token 数：中文按字符 ≈ token，英文按词 × 1.3。
    用于在线计数（精确 tokenize 太慢）。"""
    n_chars = len(text)
    # 中文 token ≈ 1 char；英文 1 token ≈ 4 char
    # 简单估算：取个折中
    return int(n_chars / 1.8)


def main():
    parser = argparse.ArgumentParser(description="Stream-download SkyPile-150B subset")
    parser.add_argument('--repo', default='Skywork/SkyPile-150B', type=str)
    parser.add_argument('--split', default='train', type=str)
    parser.add_argument('--output', default='../dataset/skypile_25b.jsonl', type=str)
    parser.add_argument('--target_tokens', type=int, default=25_000_000_000,
                        help='目标 token 数（estimate），到达后停止')
    parser.add_argument('--min_length', type=int, default=200,
                        help='样本最小字符数，太短跳过')
    parser.add_argument('--log_every', type=int, default=5000)
    parser.add_argument('--tokenizer_path', type=str, default='../model',
                        help='用 minimind tokenizer 精确计算 token 数（每 N 个样本核对一次）')
    parser.add_argument('--exact_count_every', type=int, default=50000,
                        help='每 N 个样本精确 tokenize 一次校准估算误差')
    args = parser.parse_args()

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 断点续传：count 已有文件的行数
    n_written, est_tokens_written = 0, 0
    if os.path.exists(output_path):
        print(f'[resume] found existing {output_path}, counting...')
        with open(output_path) as f:
            for line in f:
                n_written += 1
                try:
                    est_tokens_written += count_tokens_fast(json.loads(line)['text'])
                except Exception:
                    pass
        print(f'[resume] {n_written:,} samples already written, ~{est_tokens_written/1e9:.2f}B tokens')
        if est_tokens_written >= args.target_tokens:
            print('[resume] target already reached, exit')
            return

    # 校准 tokenizer（可选；下载完后跑 eval_perplexity 时会用到）
    tokenizer = None
    if args.tokenizer_path:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
            print(f'[tokenizer] loaded for calibration: vocab={tokenizer.vocab_size}')
        except Exception as e:
            print(f'[tokenizer] failed to load ({e}), will skip calibration')

    print(f'[start] streaming {args.repo} split={args.split} → {output_path}')
    print(f'[target] {args.target_tokens/1e9:.1f}B tokens (estimated)')

    ds = load_dataset(args.repo, split=args.split, streaming=True)

    skipped_short = 0
    n_seen = 0
    t0 = time.time()
    calibration_ratio = 1.0  # actual_tokens / estimated_tokens，每次校准更新

    with open(output_path, 'a') as fout:
        for sample in ds:
            n_seen += 1
            # 跳过已写过的（断点续传）
            if n_seen <= n_written:
                continue

            text = sample.get('text', '')
            if len(text) < args.min_length:
                skipped_short += 1
                continue

            est_tok = count_tokens_fast(text)
            actual_est = int(est_tok * calibration_ratio)

            # 写入 minimind 格式
            fout.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
            n_written += 1
            est_tokens_written += actual_est

            # 进度日志
            if n_written % args.log_every == 0:
                elapsed = time.time() - t0
                rate = (n_written / elapsed) if elapsed > 0 else 0
                progress = est_tokens_written / args.target_tokens * 100
                eta_h = (args.target_tokens - est_tokens_written) / max(est_tokens_written / elapsed, 1) / 3600
                print(f'[{n_written:>9,} samples] {est_tokens_written/1e9:>5.2f}B tok '
                      f'({progress:>5.1f}%) | {rate:.0f} samples/s | ETA {eta_h:.1f}h | '
                      f'skipped_short={skipped_short:,}')
                fout.flush()

            # 周期性精确 tokenize 几条校准估算
            if tokenizer and n_written % args.exact_count_every == 0:
                actual = sum(len(tokenizer(t, add_special_tokens=False).input_ids)
                             for t in [text])
                estimate = est_tok
                if estimate > 0:
                    new_ratio = actual / estimate
                    calibration_ratio = 0.9 * calibration_ratio + 0.1 * new_ratio
                    print(f'[calibration] est_ratio updated → {calibration_ratio:.3f} '
                          f'(this sample: est={estimate}, actual={actual})')

            if est_tokens_written >= args.target_tokens:
                print(f'\n[done] target reached: {est_tokens_written/1e9:.2f}B est tokens')
                break

    elapsed = time.time() - t0
    print(f'\n{"=" * 60}')
    print(f'Total: {n_written:,} samples, ~{est_tokens_written/1e9:.2f}B est tokens')
    print(f'Elapsed: {elapsed/60:.1f} min ({elapsed/3600:.1f}h)')
    print(f'Skipped (too short): {skipped_short:,}')
    print(f'Output: {output_path} ({os.path.getsize(output_path)/1e9:.2f} GB)')
    print(f'{"=" * 60}')
    if tokenizer:
        print(f'⚠️  估算 token 数有 ~{abs(1-calibration_ratio)*100:.1f}% 误差。')
        print(f'   精确数：跑 eval_perplexity.py 时会有 actual count。')


if __name__ == "__main__":
    main()
