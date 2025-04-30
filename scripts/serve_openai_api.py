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

from threading import Thread
from queue import Queue
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora

warnings.filterwarnings('ignore')

app = FastAPI()


def init_model(args):
    if args.load == 0:
        tokenizer = AutoTokenizer.from_pretrained('../model/')
        moe_path = '_moe' if args.use_moe else ''
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason'}
        ckp = f'../{args.out_dir}/{modes[args.model_mode]}_{args.hidden_size}{moe_path}.pth'
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_seq_len=args.max_seq_len,
            use_moe=args.use_moe
        ))
        model.load_state_dict(torch.load(ckp, map_location=device), strict=True)
        if args.lora_name != 'None':
            apply_lora(model)
            load_lora(model, f'../{args.out_dir}/{args.lora_name}_{args.hidden_size}.pth')
    else:
        model_path = '../MiniMind2'
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    return model.eval().to(device), tokenizer


class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    stream: bool = False
    tools: list = []


class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


def generate_stream_response(messages, temperature, top_p, max_tokens):
    try:
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)[-max_tokens:]
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        def _generate():
            model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )

        Thread(target=_generate).start()

        while True:
            text = queue.get()
            if text is None:
                yield json.dumps({
                    "choices": [{
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }, ensure_ascii=False)
                break

            yield json.dumps({
                "choices": [{"delta": {"content": text}}]
            }, ensure_ascii=False)

    except Exception as e:
        yield json.dumps({"error": str(e)})


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        if request.stream:
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens
                )),
                media_type="text/event-stream"
            )
        else:
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True
            )[-request.max_tokens:]
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + request.max_tokens,
                    do_sample=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=request.top_p,
                    temperature=request.temperature
                )
                answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
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
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument('--load', default=0, type=int, help="0: 从原生torch权重，1: 利用transformers加载")
    parser.add_argument('--model_mode', default=1, type=int,
                        help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = init_model(parser.parse_args())
    uvicorn.run(app, host="0.0.0.0", port=8998)
