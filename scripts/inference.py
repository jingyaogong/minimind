import argparse
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from accelerate import Accelerator
from peft import PeftModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora



def parse_args():
    parser = argparse.ArgumentParser(
        description="Universal inference: CPU, single-GPU, single-node multi-GPU"
    )
    parser.add_argument("--model-path", default="/mnt/share/djx/minimind/out", help="Path or HF repo id of the model")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="Model dtype")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Select device kind")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs to use; 0 means CPU. If >0 and device=cuda, uses device_map=auto")
    parser.add_argument("--prompt", default=None, help="Prompt text; if omitted, read from stdin or --input-file")
    parser.add_argument("--input-file", default=None, help="Path to a text file; one prompt per line. Ignored if --prompt provided.")
    parser.add_argument("--output-file", default=None, help="Write generations to file. When --input-file is set, outputs JSONL with fields {prompt, generation}.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling; by default greedy if not set")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for parallel generation. Effective on GPU and multi-GPU with device_map=auto.")
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA adapter loading (requires PEFT)")
    parser.add_argument("--lora-path", default=None, help="Path to LoRA adapter weights (PEFT). If not set, --use-lora is ignored.")
    # Align with eval_llm.py for native MiniMind loading
    parser.add_argument('--save-dir', default='out', type=str, help="Model weights directory (native MiniMind)")
    parser.add_argument('--weight', default='full_sft', type=str, help="Weight prefix (pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo)")
    parser.add_argument('--lora-weight', default='None', type=str, help="LoRA weight name for native MiniMind (None to disable)")
    parser.add_argument('--hidden-size', default=512, type=int, help="Hidden size (512=Small-26M, 640=MoE-145M, 768=Base-104M)")
    parser.add_argument('--num-hidden-layers', default=8, type=int, help="Number of hidden layers (Small/MoE=8, Base=16)")
    parser.add_argument('--use-moe', default=0, type=int, choices=[0, 1], help="Use MoE architecture (0/1)")
    return parser.parse_args()


def select_dtype(dtype_str: str):
    if dtype_str == "auto":
        return None
    if dtype_str == "fp32":
        return torch.float32
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16
    return None


def build_device_map(device: str, num_gpus: int):
    if device == "cpu" or (device == "auto" and not torch.cuda.is_available()):
        return None  # no device_map; model stays on CPU
    if num_gpus <= 1:
        return {"": 0}  # place all on cuda:0
    # multi-GPU: let transformers shard automatically across available GPUs
    return "auto"


def load_model_and_tokenizer(args, dtype_opt, device_map_opt, accelerator: Optional[Accelerator] = None):
    # Align behavior with eval_llm.py
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    # Native MiniMind path if 'model' in path
    if 'out' in args.model_path:
        print("Loading native MiniMind model...")
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'{args.model_path}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        state = torch.load(ckp, map_location=args.device)
        model.load_state_dict(state, strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        kwargs = {}
        if dtype_opt is not None:
            kwargs["torch_dtype"] = dtype_opt
        if device_map_opt is not None:
            kwargs["device_map"] = device_map_opt
        kwargs["low_cpu_mem_usage"] = True
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, **kwargs)
        # PEFT LoRA only if proper adapter folder provided
        if args.use_lora and args.lora_path:
            try:
                model = PeftModel.from_pretrained(model, args.lora_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load PEFT LoRA at '{args.lora_path}': {e}")

    if accelerator is not None:
        model = accelerator.prepare(model)
        model.eval()
    return model, tok


def get_prompts(args) -> List[str]:
    if args.prompt is not None:
        return [args.prompt]
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines()]
            return [ln for ln in lines if ln]
    # read from stdin (single prompt)
    print("Enter prompt. Press Ctrl-D to finish:")
    return [sys.stdin.read().strip()]


def main():
    args = parse_args()
    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        device = "cpu"
    else:
        device = "cuda"
    
    # Optional Accelerate setup
    accelerator: Optional[Accelerator] = None
    if device == "cuda" and args.num_gpus > 1:
        accelerator = Accelerator()

    dtype = select_dtype(args.dtype)
    device_map = build_device_map(device, args.num_gpus)
    model, tok = load_model_and_tokenizer(
        args,
        dtype,
        device_map,
        accelerator,
    )

    if accelerator is None and device == "cuda" and device_map in (None, {"": 0}):
        # single-GPU: move to cuda:0 if not already via device_map
        model.to("cuda")

    prompts = get_prompts(args)
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
    }

    if device == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    outputs: List[str] = []
    bs = max(1, args.batch_size)

    if accelerator is None:
        # Single-process path with optional multi-GPU device_map
        for i in range(0, len(prompts), bs):
            batch_prompts = prompts[i:i+bs]
            inputs = tok(batch_prompts, return_tensors="pt", padding=True, truncation=False)
            # Remove unused keys (e.g., token_type_ids) to avoid generate() validation errors
            inputs.pop("token_type_ids", None)
            if device == "cuda" and device_map in (None, {"": 0}):
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.inference_mode():
                out = model.generate(**inputs, **gen_kwargs)
            for j, seq in enumerate(out):
                text = tok.decode(seq, skip_special_tokens=True)
                prompt_j = batch_prompts[j]
                gen = text[len(prompt_j):] if text.startswith(prompt_j) else text
                outputs.append(gen)
    else:
        # Accelerate multi-process path: shard prompts, gather results, keep stable order
        all_items: List[Tuple[int, str]] = [(i, p) for i, p in enumerate(prompts)]
        local_items = [it for idx, it in enumerate(all_items) if idx % accelerator.num_processes == accelerator.process_index]

        local_results: List[Tuple[int, str]] = []
        for i in range(0, len(local_items), bs):
            batch = local_items[i:i+bs]
            idxs = [it[0] for it in batch]
            batch_prompts = [it[1] for it in batch]
            inputs = tok(batch_prompts, return_tensors="pt", padding=True)
            inputs.pop("token_type_ids", None)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                out = model.generate(**inputs, **gen_kwargs)
            for j, seq in enumerate(out):
                text = tok.decode(seq, skip_special_tokens=True)
                prompt_j = batch_prompts[j]
                gen = text[len(prompt_j):] if text.startswith(prompt_j) else text
                local_results.append((idxs[j], gen))

        gathered = accelerator.gather(local_results)
        if accelerator.is_main_process:
            # restore order
            gathered_sorted = sorted(gathered, key=lambda x: x[0])
            outputs = [gen for _, gen in gathered_sorted]
        else:
            return

    if args.output_file:
        # If multiple prompts, write JSONL; else write plain text
        import json
        dirpath = os.path.dirname(args.output_file)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            if len(prompts) > 1:
                for p, g in zip(prompts, outputs):
                    f.write(json.dumps({"prompt": p, "generation": g}, ensure_ascii=False) + "\n")
            else:
                f.write(outputs[0])
    else:
        if len(outputs) == 1:
            print(outputs[0])
        else:
            for p, g in zip(prompts, outputs):
                print("===== PROMPT =====")
                print(p)
                print("=== GENERATION ===")
                print(g)
    
    
if __name__ == "__main__":
    main()




# python scripts/inference.py \
#   --model-path /mnt/share/djx/minimind/out/full_sft_512.pth \
#   --save-dir out \
#   --weight full_sft \
#   --hidden-size 512 \
#   --num-hidden-layers 8 \
#   --use-moe 0 \
#   --device cuda --num-gpus 1 \
#   --prompt "你好，介绍一下LoRA是什么？"