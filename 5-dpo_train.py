import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def init_model():
    device = "cuda:0"
    # Do model patching and add fast LoRA weights
    model_name_or_path = "minimind-v1-small"
    tokenizer_name_or_path = "./model/minimind_tokenizer"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    target_modules = find_all_linear_names(model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        inference_mode=False,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model = model.to(device)
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = init_model()
    training_args = DPOConfig(
        output_dir="./minimind_dpo",
        per_device_train_batch_size=1,
        remove_unused_columns=False,
    )

    ################
    # Dataset
    ################
    # 确保路径正确，文件存在
    dataset_path = "./dataset/dpo/train_data.json"

    # 加载数据集
    train_dataset = load_dataset("json", data_files=dataset_path)

    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=0.1,
        train_dataset=train_dataset["train"],
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=512,
    )
    dpo_trainer.train()
