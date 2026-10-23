import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from modelscope import snapshot_download

# ==================== 1. 动态批次填充 + 严格 Prompt Loss Masking 数据集 ====================
class RAN2SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=2048):
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

        # 1. 构造 Prompt 前缀（包含系统预设 system + 用户提问 user）
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        # 2. 构造完整对话文本（包含完整输出 assistant）
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        if not full_ids or full_ids[-1] != self.tokenizer.eos_token_id:
            full_ids.append(self.tokenizer.eos_token_id)

        # 最大长度保护截断
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]

        # 3. 严格实现 SFT Prompt Loss Masking
        # Prompt 部分（系统提示 + 用户提问）对应标签置为 -100，仅对 Assistant 答案计算交叉熵损失
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

    local_model_dir = r"C:\Users\ZoOoM\.cache\modelscope\hub\models\Qwen\Qwen2___5-3B-Instruct"
    if os.path.exists(local_model_dir):
        print(f"[Model] 本地权重直连秒启: {local_model_dir}", flush=True)
        model_dir = local_model_dir
    else:
        print(f"[Model] 从 ModelScope 下载权重...", flush=True)
        model_dir = snapshot_download("Qwen/Qwen2.5-3B-Instruct")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ==================== 2. 配置 4-bit QLoRA (NF4 + SDPA + 双重量化) ====================
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
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)

    # 配置 LoRA 适配层 (r=16, alpha=32, 针对核心注意力投影矩阵 q, k, v, o)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(base_model, peft_config)

    trainable_p, all_p = model.get_nb_trainable_parameters()
    print(f"[Params] 总参数: {all_p / 1e6:.2f} M | LoRA 可训参数: {trainable_p / 1e6:.2f} M | 占比: {100 * trainable_p / all_p:.2f}%", flush=True)

    # ==================== 3. 加载 382 条专业训练集 ====================
    train_path = "dataset/ran2_train_382.jsonl"
    max_length = 1536
    dataset = RAN2SFTDataset(train_path, tokenizer, max_length=max_length)
    
    batch_size = 1
    accumulation_steps = 8  # 等效批大小 = 8
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    epochs = 3
    total_steps = (len(dataloader) * epochs) // accumulation_steps
    import bitsandbytes as bnb
    optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=2e-5)

    print(f"[Training] 开始 3GPP-Agent 4-bit QLoRA 严谨微调 (极速低显存版):", flush=True)
    print(f"  - 训练集样本数: {len(dataset)} 条 (独立盲测集 50 条已严格封存)")
    print(f"  - 轮数 (Epochs): {epochs}")
    print(f"  - 上下文长度 (Max Length): {max_length}")
    print(f"  - Batch Size: {batch_size} (梯度累积: {accumulation_steps}, 等效 Batch: {batch_size * accumulation_steps})")
    print(f"  - 总优化步数: {total_steps} 步", flush=True)
    
    model.train()

    step = 0
    opt_step = 0
    running_loss = 0.0
    sample_count = 0

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
            sample_count += 1

            if step % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                if opt_step % 2 == 0 or opt_step == total_steps:
                    avg_loss = running_loss / sample_count
                    vram_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
                    print(f"Step [{opt_step:3d}/{total_steps:3d}] | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | 显存: {vram_gb:.2f} GB", flush=True)
                    running_loss = 0.0
                    sample_count = 0
                    torch.cuda.empty_cache()

    # ==================== 4. 保存 3B QLoRA 适配器权重 ====================
    save_dir = "output/qwen3b_ran2_qlora"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\n[Saved] Qwen2.5-3B 3GPP-Agent QLoRA 权重已成功保存至: {save_dir}", flush=True)

if __name__ == "__main__":
    main()
