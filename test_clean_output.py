import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

model_dir = r"C:\Users\ZoOoM\.cache\modelscope\hub\models\Qwen\Qwen2___5-3B-Instruct"
lora_dir = "output/qwen3b_ran2_qlora"

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
base_model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=True
).eval()
model = PeftModel.from_pretrained(base_model, lora_dir).eval()

# 采用真正经过 382 篇清洗精调的 3GPP 专家 prompt 结构
system_prompt = "你是由知名 3GPP 国际标准化代表与 RAN2 资深协议架构师联合打造的 3GPP 垂直领域专家模型（3GPP-Agent）。请严格基于真实 3GPP 提案与技术讨论进行专业、精准、具备实战指导意义的分析。"
user_prompt = "在 3GPP RAN2#127bis 会议中，针对【LTM】特性的提案（R2-2408091：《MRO enhancements for LTM》），请全面解析该提案的技术方案与协议机制。"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to("cuda:0")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.1,
        top_p=0.85,
        eos_token_id=tokenizer.eos_token_id
    )

ans = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

with open("sample_output.txt", "w", encoding="utf-8") as f:
    f.write(ans)

print("Finished generating clean sample.")