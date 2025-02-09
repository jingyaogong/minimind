import torch
import warnings
import sys
import os

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.LMConfig import LMConfig
from model.model import MiniMindLM

warnings.filterwarnings('ignore', category=UserWarning)


def convert_torch2transformers(torch_path, transformers_path):
    def export_tokenizer(transformers_path):
        tokenizer = AutoTokenizer.from_pretrained('../model/minimind_tokenizer')
        tokenizer.save_pretrained(transformers_path)

    LMConfig.register_for_auto_class()
    MiniMindLM.register_for_auto_class("AutoModelForCausalLM")
    lm_model = MiniMindLM(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    export_tokenizer(transformers_path)
    print(f"模型已保存为 Transformers 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save(model.state_dict(), torch_path)
    print(f"模型已保存为 PyTorch 格式: {torch_path}")


# don't need to use
def push_to_hf(export_model_path):
    def init_model():
        tokenizer = AutoTokenizer.from_pretrained('../model/minimind_tokenizer')
        model = AutoModelForCausalLM.from_pretrained(export_model_path, trust_remote_code=True)
        return model, tokenizer

    model, tokenizer = init_model()
    # model.push_to_hub(model_path)
    # tokenizer.push_to_hub(model_path, safe_serialization=False)


if __name__ == '__main__':
    lm_config = LMConfig(dim=512, n_layers=8, max_seq_len=8192, use_moe=False)

    torch_path = f"../out/rlhf_{lm_config.dim}{'_moe' if lm_config.use_moe else ''}.pth"

    transformers_path = '../MiniMind2-Small'

    # convert torch to transformers model
    convert_torch2transformers(torch_path, transformers_path)

    # # convert transformers to torch model
    # convert_transformers2torch(transformers_path, torch_path)
