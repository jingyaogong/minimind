import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from modelscope import snapshot_download

# ==================== 1. 动态填充 + Prompt 损失屏蔽数据集 ====================
class SFTDatasetWithLossMask(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=768):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]
        # 1. 构造 Prompt 前缀（用户输入部分）
        prompt_text = self.tokenizer.apply_chat_template(
            [messages[0]],
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        # 2. 构造完整对话文本
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # 尾部截断保护
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]

        # 3. 严格实现 SFT Prompt Loss Masking
        # Prompt 部分对应标签置为 -100，仅对 Assistant 内容计算交叉熵损失
        labels = list(full_ids)
        mask_len = min(prompt_len, len(labels))
        for i in range(mask_len):
            labels[i] = -100

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels
        }

def collate_fn(batch, tokenizer):
    max_len = max(len(x["input_ids"]) for x in batch)
    # 向上对齐到 8 的倍数，加速 Tensor Core 显存访问
    max_len = (max_len + 7) // 8 * 8

    pad_id = tokenizer.pad_token_id or 0
    input_ids_b, attention_mask_b, labels_b = [], [], []

    for x in batch:
        cur_len = len(x["input_ids"])
        pad_len = max_len - cur_len

        input_ids_b.append(x["input_ids"] + [pad_id] * pad_len)
        attention_mask_b.append(x["attention_mask"] + [0] * pad_len)
        labels_b.append(x["labels"] + [-100] * pad_len)

    return (
        torch.tensor(input_ids_b, dtype=torch.long),
        torch.tensor(attention_mask_b, dtype=torch.long),
        torch.tensor(labels_b, dtype=torch.long)
    )

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Device] 训练硬件: {device} ({torch.cuda.get_device_name(0) if 'cuda' in device else 'CPU'})", flush=True)

    model_id = "Qwen/Qwen2.5-3B-Instruct"
    print(f"[Model] 正在加载 Qwen2.5-3B-Instruct 并配置 4-bit NF4 量化...", flush=True)
    model_dir = snapshot_download(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ==================== 2. 配置 4-bit QLoRA (NF4 + 双重量化) ====================
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
        trust_remote_code=True
    )
    base_model = prepare_model_for_kbit_training(base_model)

    # 配置 LoRA 适配层 (r=16, alpha=32)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(base_model, peft_config)

    trainable_p, all_p = model.get_nb_trainable_parameters()
    print(f"[Params] 总参数: {all_p / 1e6:.2f} M | LoRA 可训参数: {trainable_p / 1e6:.2f} M | 占比: {100 * trainable_p / all_p:.2f}%", flush=True)

    # ==================== 3. 加载真实训练数据集 ====================
    train_path = "dataset/train_ffs_sft.jsonl"
    dataset = SFTDatasetWithLossMask(train_path, tokenizer, max_length=768)
    
    batch_size = 2
    accumulation_steps = 4  # 等效批大小 = 8
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    epochs = 2
    total_steps = (len(dataloader) * epochs) // accumulation_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    
    # 余弦学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=2e-5)

    print(f"[Training] 开始 4-bit QLoRA 严谨微调 (样本数: {len(dataset)}, Epochs: {epochs}, 优化步数: {total_steps} 步)...", flush=True)
    model.train()

    step = 0
    opt_step = 0
    running_loss = 0.0

    optimizer.zero_grad()
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---", flush=True)
        for input_ids, attention_mask, labels in dataloader:
            step += 1
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accumulation_steps
            loss.backward()

            running_loss += outputs.loss.item()

            if step % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                if opt_step % 10 == 0 or opt_step == total_steps:
                    avg_loss = running_loss / (10 * accumulation_steps) if opt_step % 10 == 0 else running_loss / (step % (10 * accumulation_steps))
                    vram_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
                    print(f"Step [{opt_step}/{total_steps}] | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | 显存占用: {vram_gb:.2f} GB", flush=True)
                    running_loss = 0.0

    # ==================== 4. 保存 3B QLoRA 适配器权重 ====================
    save_dir = "output/qwen3b_telecom_qlora"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\n[Saved] Qwen2.5-3B QLoRA 权重已成功保存至: {save_dir}", flush=True)

if __name__ == "__main__":
    main()
