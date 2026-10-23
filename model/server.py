import os
import json
import time
import torch
from threading import Thread
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
import uvicorn

# 1. 实例化 FastAPI 异步 Web 服务
app = FastAPI(title="3GPP-LLM-Inference-Server")

# 2. 本地基础模型与我们微调好的 LoRA 权重路径
MODEL_DIR = r"C:\Users\ZoOoM\.cache\modelscope\hub\models\Qwen\Qwen2___5-3B-Instruct"
LORA_DIR = "output/qwen3b_ran2_qlora"

print("正在以 4-bit 精度加载 3GPP 垂直微调模型，保护 8G 显存...")

# 3. 初始化分词器（Tokenizer）
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. 配置刚才学到的 4-bit NF4 量化（将 6GB 权重压缩至 1.5GB）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 5. 加载基座 + 挂载我们微调的 LoRA 适配器
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=True
).eval()

model = PeftModel.from_pretrained(base_model, LORA_DIR).eval()
print("✅ 模型与 LoRA 适配器加载完成，推理服务就绪！")


# ==================== 6. 定义 OpenAI 兼容的请求体规范 ====================
class ChatCompletionRequest(BaseModel):
    messages: list           # 必填：对话历史 [{"role": "user", "content": "..."}]
    temperature: float = 0.2 # 采样温度（默认 0.2，保证严谨）
    max_tokens: int = 512    # 最大生成长度
    stream: bool = True      # 是否开启打字机流式传输
# ==================== 7. 实现核心流式接口 /v1/chat/completions ====================
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # 1. 使用分词器内置模板格式化整个对话历史
    prompt_text = tokenizer.apply_chat_template(
        req.messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda:0")
    # 2. 创建异步流式管道
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=0.9,
        do_sample=True if req.temperature > 0.0 else False,
        eos_token_id=tokenizer.eos_token_id
    )
    # 3. 启动独立工作线程执行计算（不阻塞 HTTP 主通道）
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    # 4. 定义 SSE 数据生成器
    def sse_event_generator():
        for token_text in streamer:
            chunk_data = {
                "choices": [{
                    "delta": {"content": token_text},
                    "finish_reason": None
                }]
            }
            # 按照标准 SSE 协议输出：data: <json>\n\n
            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
        
        # 结束标记（与 OpenAI 规范完全一致）
        yield "data: [DONE]\n\n"
    # 5. 返回长连接流式响应
    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")
# ==================== 8. 启动脚本入口 ====================
if __name__ == "__main__":
    # 在本地 8000 端口启动服务
    uvicorn.run(app, host="127.0.0.1", port=8000)