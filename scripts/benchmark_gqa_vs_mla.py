"""
GQA vs MLA Benchmark 脚本
对比 MiniMind GQA 与 MLA (Multi-head Latent Attention) 的性能差异：
  1. 模型参数量 & KV-Cache 理论大小
  2. 推理速度 (Prefill / Decode)
  3. KV-Cache 显存占用
  4. 训练吞吐量
  5. 输出质量 (PPL)

Usage:
    # 性能测试（随机权重）
    python scripts/benchmark_gqa_vs_mla.py [--device cuda:0] [--hidden_size 768]

    # 加载 checkpoint 测 PPL（必须提供 checkpoint 路径）
    python scripts/benchmark_gqa_vs_mla.py --device cuda:0 \
        --gqa_ckpt out/pretrain_768.pth \
        --mla_ckpt out/pretrain_768_mla.pth \
        --data_path dataset/pretrain_data.jsonl \
        --tokenizer_path model
"""

import argparse
import gc
import math
import os
import sys
import time

import torch

sys.path.append(".")
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_minimind_mla import MiniMindMLAConfig, MiniMindMLAForCausalLM


# ──────────────────────────── helpers ────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def attention_params_per_layer(model):
    """统计第一个 transformer block 中 self_attn 的参数量"""
    block = model.model.layers[0]
    return sum(p.numel() for p in block.self_attn.parameters())


def fmt_num(n):
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def fmt_mem(bytes_val):
    if bytes_val >= 1 << 30:
        return f"{bytes_val / (1 << 30):.2f} GB"
    if bytes_val >= 1 << 20:
        return f"{bytes_val / (1 << 20):.2f} MB"
    return f"{bytes_val:.0f} B"


def warmup_forward(model, input_ids, device, n=3):
    """warmup: forward n 次，不计时"""
    model.train(False)
    for _ in range(n):
        with torch.no_grad():
            model(input_ids.to(device))
    torch.cuda.synchronize(device)


def cleanup_gpu(device):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


# ────────────────────── model factories ──────────────────────────

def build_dense_attention(hidden_size, attention_type):
    config = MiniMindConfig(hidden_size=hidden_size, attention_type=attention_type)
    return MiniMindForCausalLM(config)


def build_mla(hidden_size, kv_lora_rank):
    config = MiniMindMLAConfig(hidden_size=hidden_size, kv_lora_rank=kv_lora_rank)
    return MiniMindMLAForCausalLM(config)


# ────────────────────── Benchmark 1: 模型信息 ────────────────────

def benchmark_model_info(hidden_size):
    print("\n" + "=" * 70)
    print("  Benchmark 1: 模型信息对比")
    print("=" * 70)

    models = {
        "MHA": build_dense_attention(hidden_size, "mha"),
        "GQA": build_dense_attention(hidden_size, "gqa"),
        "MQA": build_dense_attention(hidden_size, "mqa"),
        "MLA(128)": build_mla(hidden_size, 128),
        "MLA(64)": build_mla(hidden_size, 64),
        "MLA(256)": build_mla(hidden_size, 256),
    }

    # Dense attention: KV-Cache floats/token/layer = 2 * num_kv_heads * head_dim
    gqa_config = MiniMindConfig(hidden_size=hidden_size, attention_type="gqa")
    kv_cache_gqa = 2 * gqa_config.num_key_value_heads * gqa_config.head_dim

    header = f"{'指标':<28}" + "".join(f"{name:>12}" for name in models)
    print(header)
    print("-" * len(header))

    total_params = {}
    attn_params = {}
    kv_cache = {}

    for name, m in models.items():
        total_params[name] = count_params(m)
        attn_params[name] = attention_params_per_layer(m)
        if "MLA" in name:
            rank = int(name.split("(")[1].rstrip(")"))
            kv_cache[name] = rank + m.config.rope_dim  # kv_latent + k_rope
        else:
            kv_cache[name] = 2 * m.config.num_key_value_heads * m.config.head_dim

    row = f"{'总参数量':<28}" + "".join(f"{fmt_num(total_params[n]):>12}" for n in models)
    print(row)
    row = f"{'注意力参数量/层':<28}" + "".join(f"{fmt_num(attn_params[n]):>12}" for n in models)
    print(row)
    row = f"{'KV-Cache (floats/tok/layer)':<28}" + "".join(f"{kv_cache[n]:>12}" for n in models)
    print(row)
    row = f"{'KV-Cache 压缩比':<28}" + "".join(
        f"{kv_cache_gqa / kv_cache[n]:>11.1f}x" for n in models
    )
    print(row)

    return models, kv_cache, kv_cache_gqa


# ────────────────────── Benchmark 2: 推理速度 ────────────────────

def benchmark_inference_speed(models, device):
    print("\n" + "=" * 70)
    print("  Benchmark 2: 推理速度 (Prefill / Decode)")
    print("=" * 70)

    prompt_lens = [128, 512, 1024]
    batch_sizes = [1, 4]
    max_new_tokens = 128

    for bs in batch_sizes:
        print(f"\n--- batch_size = {bs} ---")
        header = (f"{'模型':<16}{'prompt_len':>12}")
        for pl in prompt_lens:
            header += f"{'Prefill(ms)':>14}"
        for pl in prompt_lens:
            header += f"{'Decode(tok/s)':>16}"
        print(header)
        print("-" * len(header))

        for name, model in models.items():
            model = model.half().to(device).eval()
            row = f"{name:<16}"
            prefill_times = []

            for pl in prompt_lens:
                input_ids = torch.randint(0, 6400, (bs, pl), device=device)
                warmup_forward(model, input_ids, device, n=2)

                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    model(input_ids)
                torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                ms = (t1 - t0) * 1000
                prefill_times.append(ms)
                row += f"{ms:>14.2f}"

            for i, pl in enumerate(prompt_lens):
                input_ids = torch.randint(0, 6400, (bs, pl), device=device)
                attn_mask = torch.ones(bs, pl, device=device)
                warmup_forward(model, input_ids, device, n=2)

                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    model.generate(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                    )
                torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                tokens_per_sec = max_new_tokens * bs / (t1 - t0)
                row += f"{tokens_per_sec:>16.1f}"

            print(row)
            del model
            cleanup_gpu(device)


# ────────────────────── Benchmark 3: KV-Cache 显存 ───────────────

def benchmark_kv_cache_memory(models, device):
    print("\n" + "=" * 70)
    print("  Benchmark 3: KV-Cache 显存占用")
    print("=" * 70)

    context_lens = [256, 512, 1024, 2048, 4096]
    max_new_tokens = 64
    bs = 1

    header = f"{'模型':<16}" + "".join(f"{'ctx='+str(c):>12}" for c in context_lens)
    print(header)
    print("-" * len(header))

    for name, model in models.items():
        model = model.half().to(device).eval()
        row = f"{name:<16}"
        mem_results = []

        for ctx_len in context_lens:
            cleanup_gpu(device)
            input_ids = torch.randint(0, 6400, (bs, ctx_len), device=device)
            attn_mask = torch.ones(bs, ctx_len, device=device)

            # 记录 generate 前的显存
            torch.cuda.synchronize(device)
            mem_before = torch.cuda.max_memory_allocated(device)

            with torch.no_grad():
                model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )

            torch.cuda.synchronize(device)
            mem_after = torch.cuda.max_memory_allocated(device)
            delta = mem_after - mem_before
            mem_results.append(delta)
            row += f"{fmt_mem(delta):>12}"

        print(row)

        # 打印显存增长趋势
        print(f"  {'增长趋势':<14}", end="")
        for i in range(1, len(mem_results)):
            ratio = mem_results[i] / mem_results[i - 1] if mem_results[i - 1] > 0 else 0
            print(f"{'×' + f'{ratio:.2f}':>12}", end="")
        print()

        del model
        cleanup_gpu(device)


# ────────────────────── Benchmark 4: 训练吞吐量 ─────────────────

def benchmark_training_throughput(models, device):
    print("\n" + "=" * 70)
    print("  Benchmark 4: 训练吞吐量 (forward + backward)")
    print("=" * 70)

    batch_size = 4
    seq_len = 340
    n_steps = 10

    header = f"{'模型':<16}{'step时间(ms)':>16}{'steps/s':>12}{'tokens/s':>14}"
    print(header)
    print("-" * len(header))

    for name, model in models.items():
        model = model.float().to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # warmup
        for _ in range(3):
            input_ids = torch.randint(0, 6400, (batch_size, seq_len), device=device)
            labels = torch.randint(0, 6400, (batch_size, seq_len), device=device)
            outputs = model(input_ids, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        for _ in range(n_steps):
            input_ids = torch.randint(0, 6400, (batch_size, seq_len), device=device)
            labels = torch.randint(0, 6400, (batch_size, seq_len), device=device)
            outputs = model(input_ids, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        total_time = t1 - t0
        step_ms = total_time / n_steps * 1000
        steps_per_sec = n_steps / total_time
        tokens_per_sec = batch_size * seq_len * n_steps / total_time

        print(f"{name:<16}{step_ms:>16.2f}{steps_per_sec:>12.2f}{tokens_per_sec:>14.1f}")

        del model, optimizer
        cleanup_gpu(device)


# ────────────────────── Benchmark 5: 输出质量 (PPL) ──────────────

def benchmark_perplexity(args, device, hidden_size):
    print("\n" + "=" * 70)
    print("  Benchmark 5: 输出质量 (Perplexity)")
    print("=" * 70)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    samples = load_dataset("json", data_files=args.data_path, split="train")
    # 取前 eval_samples 条，截取文本
    max_seq = 512
    all_tokens = []
    for i in range(min(args.eval_samples, len(samples))):
        text = str(samples[i].get("text", ""))
        ids = tokenizer(text, add_special_tokens=False, max_length=max_seq, truncation=True).input_ids
        ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
        all_tokens.append(ids)

    results = {}

    for ckpt_name, ckpt_path, config_cls, config_kwargs in [
        ("GQA", args.gqa_ckpt, MiniMindConfig, {}),
        ("MLA(128)", args.mla_ckpt, MiniMindMLAConfig, {"kv_lora_rank": 128}),
        ("MLA(64)", args.mla_ckpt_r64, MiniMindMLAConfig, {"kv_lora_rank": 64}),
        ("MLA(256)", args.mla_ckpt_r256, MiniMindMLAConfig, {"kv_lora_rank": 256}),
    ]:
        if ckpt_path is None:
            continue
        if not os.path.isfile(ckpt_path):
            print(f"  [SKIP] {ckpt_name}: checkpoint not found at {ckpt_path}")
            continue

        config_kw = {"hidden_size": hidden_size}
        config_kw.update(config_kwargs)
        config = config_cls(**config_kw)
        if isinstance(config, MiniMindMLAConfig):
            model = MiniMindMLAForCausalLM(config)
        else:
            model = MiniMindForCausalLM(config)

        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model = model.half().to(device).eval()

        total_loss = 0.0
        total_tokens = 0
        batch_input_ids = []
        batch_labels = []

        for ids in all_tokens:
            pad_len = max_seq - len(ids)
            padded = ids + [tokenizer.pad_token_id] * pad_len
            labels = ids[1:] + [tokenizer.pad_token_id] * (pad_len + 1)
            # 标记 padding 为 -100
            labels = [t if t != tokenizer.pad_token_id else -100 for t in labels]
            batch_input_ids.append(padded)
            batch_labels.append(labels)

            if len(batch_input_ids) == args.eval_batch_size:
                input_t = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
                label_t = torch.tensor(batch_labels, dtype=torch.long, device=device)
                with torch.no_grad():
                    out = model(input_t, labels=label_t)
                # loss 是 cross_entropy 的 mean，乘以有效 token 数得到总 loss
                n_valid = (label_t != -100).sum().item()
                total_loss += out.loss.item() * n_valid
                total_tokens += n_valid
                batch_input_ids, batch_labels = [], []

        # 处理剩余样本
        if batch_input_ids:
            input_t = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
            label_t = torch.tensor(batch_labels, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_t, labels=label_t)
            n_valid = (label_t != -100).sum().item()
            total_loss += out.loss.item() * n_valid
            total_tokens += n_valid

        avg_nll = total_loss / total_tokens
        ppl = math.exp(avg_nll)
        results[ckpt_name] = ppl
        print(f"  {ckpt_name:<16} Loss: {avg_nll:.4f}  PPL: {ppl:.2f}  (eval on {total_tokens} tokens)")

        del model, state_dict
        cleanup_gpu(device)

    if not results:
        print("  没有可用的 checkpoint，跳过 PPL 测试")
        print("  提示：使用 --gqa_ckpt / --mla_ckpt 参数指定权重路径")
        return {}

    # Markdown PPL 表格
    print()
    cols = " | ".join(results.keys())
    print(f"| Metric | {cols} |")
    sep = "|".join(["------"] * len(results))
    print(f"|--------|{sep}|")
    print(f"| PPL | " + " | ".join(f"{results[k]:.2f}" for k in results) + " |")
    print()
    return results


# ────────────────────── Markdown 汇总 ────────────────────────────

def print_markdown_summary(models_info, hidden_size):
    print("\n" + "=" * 70)
    print("  Markdown 汇总（可复制到简历 / README）")
    print("=" * 70)
    print()

    gqa_config = MiniMindConfig(hidden_size=hidden_size, attention_type="gqa")
    kv_cache_gqa = 2 * gqa_config.num_key_value_heads * gqa_config.head_dim

    print("### Model Info")
    print()
    print("| Metric | " + " | ".join(models_info.keys()) + " |")
    print("|--------|" + "|".join(["------"] * len(models_info)) + "|")

    total_params = {}
    attn_params = {}
    kv_cache = {}
    for name, m in models_info.items():
        total_params[name] = count_params(m)
        attn_params[name] = attention_params_per_layer(m)
        if "MLA" in name:
            rank = int(name.split("(")[1].rstrip(")"))
            kv_cache[name] = rank + m.config.rope_dim  # kv_latent + k_rope
        else:
            kv_cache[name] = 2 * m.config.num_key_value_heads * m.config.head_dim

    row = "| Total Params |"
    for n in models_info:
        row += f" {fmt_num(total_params[n])} |"
    print(row)

    row = "| Attn Params/Layer |"
    for n in models_info:
        row += f" {fmt_num(attn_params[n])} |"
    print(row)

    row = "| KV-Cache (floats/tok/layer) |"
    for n in models_info:
        row += f" {kv_cache[n]} |"
    print(row)

    row = "| KV-Cache Compression Ratio |"
    for n in models_info:
        row += f" {kv_cache_gqa / kv_cache[n]:.1f}x |"
    print(row)
    print()

    print("> hidden_size={}, num_layers={}, num_attention_heads={}, num_kv_heads={}, head_dim={}".format(
        gqa_config.hidden_size, gqa_config.num_hidden_layers,
        gqa_config.num_attention_heads, gqa_config.num_key_value_heads, gqa_config.head_dim
    ))
    print()


# ────────────────────── main ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GQA vs MLA Benchmark")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden_size", type=int, default=768)
    # Benchmark 5 (PPL) 参数
    parser.add_argument("--gqa_ckpt", type=str, default=None, help="GQA checkpoint 路径 (e.g. out/pretrain_768.pth)")
    parser.add_argument("--mla_ckpt", type=str, default=None, help="MLA checkpoint 路径 (e.g. out/pretrain_768_mla.pth)")
    parser.add_argument("--mla_ckpt_r64", type=str, default=None, help="MLA(64) checkpoint 路径")
    parser.add_argument("--mla_ckpt_r256", type=str, default=None, help="MLA(256) checkpoint 路径")
    parser.add_argument("--data_path", type=str, default="dataset/pretrain_hq.jsonl", help="评估数据集路径")
    parser.add_argument("--tokenizer_path", type=str, default="model", help="tokenizer 路径")
    parser.add_argument("--eval_samples", type=int, default=500, help="评估样本数")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="评估 batch size")
    args = parser.parse_args()

    device = torch.device(args.device)
    hidden_size = args.hidden_size

    has_ckpt = any([args.gqa_ckpt, args.mla_ckpt, args.mla_ckpt_r64, args.mla_ckpt_r256])

    print("=" * 70)
    print(f"  MiniMind GQA vs MLA Benchmark")
    print(f"  hidden_size={hidden_size}  device={device}")
    if has_ckpt:
        print(f"  PPL evaluation: ENABLED (data={args.data_path})")
    print("=" * 70)

    # Benchmark 1: 模型信息（CPU 即可）
    models, kv_cache, kv_cache_gqa = benchmark_model_info(hidden_size)

    if device.type == "cuda":
        # Benchmark 2: 推理速度
        models_gpu = {
            "MHA": build_dense_attention(hidden_size, "mha"),
            "GQA": build_dense_attention(hidden_size, "gqa"),
            "MQA": build_dense_attention(hidden_size, "mqa"),
            "MLA(128)": build_mla(hidden_size, 128),
            "MLA(64)": build_mla(hidden_size, 64),
            "MLA(256)": build_mla(hidden_size, 256),
        }
        benchmark_inference_speed(models_gpu, device)

        # Benchmark 3: KV-Cache 显存
        models_gpu = {
            "MHA": build_dense_attention(hidden_size, "mha"),
            "GQA": build_dense_attention(hidden_size, "gqa"),
            "MQA": build_dense_attention(hidden_size, "mqa"),
            "MLA(128)": build_mla(hidden_size, 128),
            "MLA(64)": build_mla(hidden_size, 64),
            "MLA(256)": build_mla(hidden_size, 256),
        }
        benchmark_kv_cache_memory(models_gpu, device)

        # Benchmark 4: 训练吞吐量
        models_gpu = {
            "MHA": build_dense_attention(hidden_size, "mha"),
            "GQA": build_dense_attention(hidden_size, "gqa"),
            "MQA": build_dense_attention(hidden_size, "mqa"),
            "MLA(128)": build_mla(hidden_size, 128),
            "MLA(64)": build_mla(hidden_size, 64),
            "MLA(256)": build_mla(hidden_size, 256),
        }
        benchmark_training_throughput(models_gpu, device)

        # Benchmark 5: PPL（需要提供 checkpoint 路径）
        if has_ckpt:
            benchmark_perplexity(args, device, hidden_size)
        else:
            print("\n" + "=" * 70)
            print("  Benchmark 5: 输出质量 (PPL) — SKIPPED")
            print("=" * 70)
            print("  未指定 checkpoint，跳过 PPL 测试。")
            print("  使用方法: python scripts/benchmark_gqa_vs_mla.py \\")
            print("      --gqa_ckpt out/pretrain_768.pth \\")
            print("      --mla_ckpt out/pretrain_768_mla.pth \\")
            print("      --data_path dataset/pretrain_hq.jsonl")
    else:
        print("\n[SKIP] Benchmark 2-5 需要 CUDA 设备，当前 device=cpu")

    # Markdown 汇总（使用 Benchmark 1 的 CPU 模型）
    print_markdown_summary(models, hidden_size)

    print("Done.")


if __name__ == "__main__":
    main()
