from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "minimind/out/merged_model"

# 加载 tokenizer（通常不需要 trust_remote_code）
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("EOS token:", tokenizer.eos_token)
print("EOS token ID:", tokenizer.eos_token_id)
print("PAD token ID:", tokenizer.pad_token_id)

# 加载模型：必须加 trust_remote_code=True
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    trust_remote_code=True
).to("cuda")  # 或 "cpu"

# 推理
messages = [
    {"role": "user", "content": "你好"}
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  # ← 关键！添加生成触发符
)
print("Prompt:", repr(prompt))
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    return_token_type_ids=False  # ← 关键！
).to(model.device)
# 关键：用 autocast 确保所有计算在 float16 下进行
with torch.autocast(device_type="cuda", dtype=torch.float16):
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.3,
        top_p=0.9
    )
print("Generated token IDs:", outputs[0].tolist())
print("Decoded with special tokens:", tokenizer.decode(outputs[0], skip_special_tokens=False))
print(tokenizer.decode(outputs[0], skip_special_tokens=True))