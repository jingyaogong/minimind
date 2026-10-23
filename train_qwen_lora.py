import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from modelscope import snapshot_download

# ==================== 1. 动态填充的高效通信与专利数据集 ====================
class TelecomPatentDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=384):
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
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        # 不使用全局静态 padding，仅在此处截断，由 collate_fn 做动态批次 padding
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors=None
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }

def collate_fn(batch, tokenizer):
    # 动态批次填充：仅填充至当前 batch 的最大实际长度，节省 70% 显存与计算量
    max_len = max(len(x["input_ids"]) for x in batch)
    # 向上对齐到 8 的倍数，加速 Tensor Core 计算
    max_len = (max_len + 7) // 8 * 8
    
    pad_id = tokenizer.pad_token_id or 0
    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []
    
    for x in batch:
        cur_len = len(x["input_ids"])
        pad_len = max_len - cur_len
        
        input_ids = x["input_ids"] + [pad_id] * pad_len
        attention_mask = x["attention_mask"] + [0] * pad_len
        labels = x["input_ids"] + [-100] * pad_len
        
        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        labels_batch.append(labels)
        
    return (
        torch.tensor(input_ids_batch, dtype=torch.long),
        torch.tensor(attention_mask_batch, dtype=torch.long),
        torch.tensor(labels_batch, dtype=torch.long)
    )

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Device] 当前训练设备: {device} ({torch.cuda.get_device_name(0) if 'cuda' in device else 'CPU'})", flush=True)

    # ==================== 2. 加载 Qwen2.5 底座模型 ====================
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"[ModelScope] 正在加载模型 {model_id} ...", flush=True)
    model_dir = snapshot_download(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )

    # ==================== 3. 配置 LoRA 适配层 (PEFT) ====================
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    
    trainable_params, all_params = model.get_nb_trainable_parameters()
    print(f"[Params] 总参数: {all_params / 1e6:.2f} M | LoRA 可训参数: {trainable_params / 1e6:.2f} M | 占比: {100 * trainable_params / all_params:.2f}%", flush=True)

    # ==================== 4. 数据加载与优化器 ====================
    dataset_path = "dataset/telecom_patent_sft.jsonl"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"未找到数据集 {dataset_path}，请先运行 python prepare_data.py")

    dataset = TelecomPatentDataset(dataset_path, tokenizer, max_length=384)
    batch_size = 4
    accumulation_steps = 2
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    
    epochs = 1  # 2600条高质量数据跑 1 轮（650个批次）即已充分收敛
    total_steps = len(dataloader) // accumulation_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()

    print(f"[Training] 开始动态轻量微调 2600 条数据 (优化步数: {total_steps} 步) ...", flush=True)
    step = 0
    opt_step = 0
    total_loss = 0.0

    optimizer.zero_grad()
    for input_ids, attention_mask, labels in dataloader:
        step += 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / accumulation_steps
        loss.backward()

        total_loss += outputs.loss.item()

        if step % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            opt_step += 1

            if opt_step % 25 == 0 or opt_step == total_steps:
                print(f"Step [{opt_step}/{total_steps}] | Loss: {outputs.loss.item():.4f}", flush=True)

    avg_loss = total_loss / len(dataloader)
    print(f"[Done] 训练完成！平均 Loss: {avg_loss:.4f}", flush=True)

    # ==================== 5. 保存 LoRA 权重 ====================
    save_dir = "output/qwen_telecom_lora"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\n[Saved] 2600条数据训练成功！LoRA 权重已保存至: {save_dir}", flush=True)

if __name__ == "__main__":
    main()
