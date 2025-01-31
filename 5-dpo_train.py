import os
import warnings
import wandb

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset

warnings.filterwarnings('ignore')


def init_model():
    device = 'cuda:0'
    # Do model patching and add fast LoRA weights
    model_name_or_path = "minimind-v1-small"
    tokenizer_name_or_path = "minimind-v1-small"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = init_model()
    
    # 初始化 wandb
    wandb.init(
        project="minimind-dpo",
        name=f"dpo-training-lr-4e-5-beta-0.1",
        config={
            "learning_rate": 4e-5,
            "beta": 0.1,
            "batch_size": 1,
            "max_length": 512,
        }
    )
    
    training_config = DPOConfig(
        output_dir="./minimind_dpo",
        per_device_train_batch_size=1,
        remove_unused_columns=False,
        report_to="wandb",  # 启用wandb记录
        save_steps=2000,
        learning_rate=4e-5,
        beta=0.1,
        max_length=512,
        max_prompt_length=512,
        logging_steps=50,  # 增加日志记录频率
        logging_first_step=True  # 记录第一步
        # 移除了 evaluation_strategy 和 eval_steps
    )

    dataset_path = './dataset/dpo/train_data.json'
    train_dataset = load_dataset('json', data_files=dataset_path)
    # 将数据集转换为训练集
    # train_dataset = train_dataset['train']
    print("-----------------")
    print(train_dataset)
    import sys
    # sys.exit(0)

    # copy_model = AutoModelForCausalLM.from_pretrained("minimind-v1-small", trust_remote_code=True)
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_config,
        # beta=0.1,
        train_dataset=train_dataset['train'],
        tokenizer=tokenizer,
        # max_length=512,
        # max_prompt_length=512
    )
    
    try:
        dpo_trainer.train()
    finally:
        wandb.finish()  # 确保正确关闭wandb
