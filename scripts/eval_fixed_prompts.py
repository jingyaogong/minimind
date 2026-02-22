import os
import json
import time
import argparse
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params


def load_prompts(path):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item.get("prompt") or item.get("text")
            if prompt:
                prompts.append(prompt)
    return prompts


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
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] Missing keys: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys: {unexpected}")
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)

    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer


def main():
    parser = argparse.ArgumentParser(description="MiniMind 固定 Prompt 评测")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.7, type=float, help="生成温度")
    parser.add_argument('--top_p', default=0.9, type=float, help="top_p 采样")
    parser.add_argument('--do_sample', default=0, type=int, choices=[0, 1], help="是否采样（0=贪婪，1=采样）")
    parser.add_argument('--seed', default=2026, type=int, help="随机种子")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--prompts_file', default='eval/prompts_minimal.jsonl', type=str, help="prompt 文件路径")
    parser.add_argument('--out_dir', default='eval_runs', type=str, help="结果保存目录")
    parser.add_argument('--run_name', default='', type=str, help="本次评测名称")
    parser.add_argument('--use_chat', default=-1, type=int, choices=[-1, 0, 1], help="是否使用chat模板（-1=自动）")
    parser.add_argument('--config', default='', type=str, help="评测配置文件（JSON，可覆盖生成参数）")
    args = parser.parse_args()

    # 可选：从配置文件覆盖评测参数
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    os.makedirs(args.out_dir, exist_ok=True)
    prompts = load_prompts(args.prompts_file)
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompts_file}")

    if args.use_chat == -1:
        use_chat = (args.weight != 'pretrain')
    else:
        use_chat = bool(args.use_chat)

    setup_seed(args.seed)
    model, tokenizer = init_model(args)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.weight}_{timestamp}"
    out_path = os.path.join(args.out_dir, f"{run_name}.jsonl")

    print(f"[Eval] prompts: {len(prompts)} | use_chat={use_chat} | out={out_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            setup_seed(args.seed + i)
            if use_chat:
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = tokenizer.bos_token + prompt

            inputs = tokenizer(text, return_tensors="pt", truncation=True).to(args.device)
            start = time.time()
            with torch.no_grad():
                gen_ids = model.generate(
                    inputs=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=bool(args.do_sample),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.0
                )
            elapsed = time.time() - start
            gen_tokens = gen_ids.shape[-1] - inputs["input_ids"].shape[-1]
            response = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

            record = {
                "id": i,
                "prompt": prompt,
                "response": response,
                "gen_tokens": gen_tokens,
                "time_sec": round(elapsed, 4),
                "tokens_per_sec": round(gen_tokens / max(elapsed, 1e-6), 2)
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{i+1}/{len(prompts)}] tokens={gen_tokens} time={elapsed:.2f}s")

    print(f"[Done] saved to {out_path}")


if __name__ == "__main__":
    main()
