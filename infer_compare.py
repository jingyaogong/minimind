import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from modelscope import snapshot_download

def generate_reply(model, tokenizer, prompt_text, system_prompt="你是由通信领域与知识产权专家调优的专业助手，擅长分析通信技术协议与专利权利要求书结构。"):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )
    # 截取生成的新 token
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Device] 当前推理设备: {device}")
    
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_dir = "output/qwen_telecom_lora"
    model_dir = snapshot_download(model_id)

    print("[ModelScope] 正在加载基座模型 Tokenizer 与 Model ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )

    test_prompt = (
        "请针对以下公开专利进行技术方案与权利要求拆解：\n"
        "【专利名称】：5G毫米波大规模天线阵列混合波束赋形方法及收发机\n"
        "【技术方案】：针对高频通信中射频链路(RF Chain)硬件受限问题，提出一种模拟移相网络与数字基带迫零(ZF)预编码分步解耦优化架构，在保证系统信噪比的同时大幅降低硬件功耗。"
    )

    print("正在测试原版 Base 模型...")
    base_reply = generate_reply(base_model, tokenizer, test_prompt)

    print("正在挂载 LoRA 适配层权重...")
    lora_model = PeftModel.from_pretrained(base_model, lora_dir)
    print("正在测试 LoRA 微调后模型...")
    lora_reply = generate_reply(lora_model, tokenizer, test_prompt)

    result_text = []
    result_text.append("="*40 + " 1. 原版 Base 模型输出 " + "="*40)
    result_text.append(base_reply)
    result_text.append("\n" + "="*40 + " 2. 加载 LoRA 微调权重后输出 " + "="*40)
    result_text.append(lora_reply)
    result_text.append("\n" + "="*90)
    
    full_output = "\n".join(result_text)
    print(full_output)
    
    with open("output/comparison_result.txt", "w", encoding="utf-8") as f:
        f.write(full_output)
    print("\n[Done] 完整中文对比结果已保存到 output/comparison_result.txt")

if __name__ == "__main__":
    main()
