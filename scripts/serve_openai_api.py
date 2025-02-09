import argparse
import json
import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
import warnings
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.LMConfig import LMConfig
from model.model import MiniMindLM
from model.model_lora import apply_lora, load_lora

warnings.filterwarnings('ignore')

app = FastAPI()


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained('../model/minimind_tokenizer')
    if args.load == 0:
        moe_path = '_moe' if args.use_moe else ''
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason'}
        ckp = f'../{args.out_dir}/{modes[args.model_mode]}_{args.dim}{moe_path}.pth'

        model = MiniMindLM(LMConfig(
            dim=args.dim,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
            use_moe=args.use_moe
        ))

        state_dict = torch.load(ckp, map_location=device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)

        if args.lora_name != 'None':
            apply_lora(model)
            load_lora(model, f'../{args.out_dir}/{args.lora_name}_{args.dim}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(
            './MiniMind2',
            trust_remote_code=True
        )
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    return model.eval().to(device), tokenizer


class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    top_p: int = 0.92
    max_tokens: int = 8192
    stream: bool = False


def generate_stream_response(messages, temperature, top_p, max_tokens):
    try:
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)[-max_tokens:]
        x = tokenizer(new_prompt).data['input_ids']
        x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])
        with torch.no_grad():
            res_y = model.generate(
                x,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                rp=1.,
                pad_token_id=tokenizer.pad_token_id
            )
            history_idx = 0
            for y in res_y:
                answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                if (answer and answer[-1] == '�') or not answer:
                    continue
                delta = answer[history_idx:]
                history_idx = len(answer)
                json_data = {
                    'id': f'chatcmpl-{int(time.time())}',
                    'object': 'chat.completion.chunk',
                    'created': int(time.time()),
                    'model': 'minimind',
                    'choices': [{'index': 0, 'delta': {'content': delta}, 'finish_reason': None}]
                }
                yield f"data: {json.dumps(json_data)}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        if request.stream:
            return StreamingResponse(
                generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens
                ),
                media_type="text/event-stream"
            )
        else:
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True
            )[-request.max_tokens:]
            x = tokenizer(new_prompt).data['input_ids']
            x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])
            with torch.no_grad():
                res_y = model.generate(
                    x,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stream=False,
                    rp=1.,
                    pad_token_id=tokenizer.pad_token_id
                )
                answer = tokenizer.decode(res_y.squeeze()[x.shape[1]:].tolist(), skip_special_tokens=True)
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop"
                    }
                ]
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server for MiniMind")
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--lora_name', default='None', type=str)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument('--load', default=0, type=int, help="0: 从原生torch权重，1: 利用transformers加载")
    parser.add_argument('--model_mode', default=1, type=int, help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = init_model(parser.parse_args())

    uvicorn.run(app, host="0.0.0.0", port=8998)
