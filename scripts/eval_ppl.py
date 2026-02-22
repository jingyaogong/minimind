import os
import math
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_model_params


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if "model" in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        state = torch.load(ckp, map_location=args.device)
        model.load_state_dict(state, strict=False)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer


def main():
    parser = argparse.ArgumentParser(description="MiniMind PPL 评测")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='pretrain', type=str, help="权重名称前缀")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--data_path', default='dataset/pretrain_hq.jsonl', type=str, help="评测数据路径")
    parser.add_argument('--max_seq_len', default=340, type=int, help="最大序列长度")
    parser.add_argument('--batch_size', default=8, type=int, help="batch size")
    parser.add_argument('--num_workers', default=4, type=int, help="dataloader workers")
    parser.add_argument('--max_samples', default=0, type=int, help="最多评测样本数（0=全量）")
    parser.add_argument('--out_path', default='', type=str, help="保存评测结果的 JSON 文件")
    parser.add_argument('--method', default='', type=str, help="方法名称（用于汇总统计）")
    args = parser.parse_args()

    model, tokenizer = init_model(args)
    ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    if args.max_samples and args.max_samples > 0:
        ds.samples = ds.samples.select(range(min(args.max_samples, len(ds.samples))))

    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)

    total_loss = 0.0
    total_tokens = 0
    total_steps = 0
    start = time.time()

    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)

            outputs = model(input_ids)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='sum'
            )
            token_count = (shift_labels != -100).sum().item()
            total_loss += loss.item()
            total_tokens += token_count
            total_steps += 1

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    elapsed = time.time() - start
    tokens_per_sec = total_tokens / max(elapsed, 1e-6)

    print(f"[PPL] loss={avg_loss:.4f} ppl={ppl:.4f}")
    print(f"[Info] tokens={total_tokens} steps={total_steps} time={elapsed:.2f}s tokens/s={tokens_per_sec:.2f}")

    if args.out_path:
        os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
        result = {
            "loss": avg_loss,
            "ppl": ppl,
            "tokens": total_tokens,
            "steps": total_steps,
            "time_sec": elapsed,
            "tokens_per_sec": tokens_per_sec,
            "data_path": args.data_path,
            "weight": args.weight,
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_hidden_layers,
            "use_moe": args.use_moe,
            "max_seq_len": args.max_seq_len,
            "batch_size": args.batch_size
        }
        if args.method:
            result["method"] = args.method
        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
