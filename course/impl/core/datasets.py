"""Dataset helpers to implement during Pretrain, SFT, and DPO stages."""

from __future__ import annotations

from torch.utils.data import Dataset
import random
from datasets import load_dataset, Features, Sequence, Value
import torch
def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations
    
class CoursePretrainDataset(Dataset):
    """TODO: read jsonl text samples and return input_ids/labels.

    Align with: dataset/lm_dataset.py::PretrainDataset
    """

    def __init__(self, data_path: str, tokenizer, max_length: int):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples=load_dataset('json', data_files=data_path, split='train', features=features)
        self.bos_id=tokenizer(f'{tokenizer.bos_token}assistant\n',add_special_tokens=False).input_ids
        self.eos_id=tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self) -> int:
        raise NotImplementedError("Implement in the Pretrain dataset lesson.")

    def __getitem__(self, index: int):
        raise NotImplementedError("Implement in the Pretrain dataset lesson.")


    
class CourseSFTDataset(Dataset):
    """TODO: render conversations and build SFT labels.

    Align with: dataset/lm_dataset.py::SFTDataset
    """

    def __init__(self, data_path: str, tokenizer, max_length: int):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples=load_dataset('json', data_files=data_path, split='train', features=features)
        self.bos_id=tokenizer(f'{tokenizer.bos_token}assistant\n',add_special_tokens=False).input_ids
        self.eos_id=tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def create_chat_prompt(self, conversations):
        """TODO: render conversations with the tokenizer chat template.

        Align with: dataset/lm_dataset.py::SFTDataset.create_chat_prompt
        """
        messages=[]
        tools=None
        for message in conversations:
            message=dict(message) 
            if message.get("role")=='system' and message.get("tools"):
                 tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get('tool_calls') and isinstance(message["tool_calls"], str):
                 message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )
        raise NotImplementedError("Implement in the SFT training lesson.")

    def generate_labels(self, input_ids: list[int]) -> list[int]:
        """TODO: keep assistant reply tokens and mask everything else as -100.

        Align with: dataset/lm_dataset.py::SFTDataset.generate_labels
        """
        labels=[-100]*len(input_ids)
        i=0
        while i<len(input_ids):
            if input_ids[i:i+len(self.bos_id)]==self.bos_id:
                start=i+len(self.bos_id)
                end=start
                while end<len(input_ids):
                    if input_ids[end:end+len(self.eos_id)]==self.eos_id:
                        break
                    end+=1
                for j in range(start,min(end+len(self.eos_id),self.max_length)):
                    labels[j]=input_ids[j]
                i=end+len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i=i+1
        return labels
        raise NotImplementedError("Implement in the SFT training lesson.")

    def __len__(self) -> int:
        return len(self.samples)
        raise NotImplementedError("Implement in the SFT dataset lesson.")

    def __getitem__(self, index: int):
        sample=self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt=self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids=self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids+=[self.tokenizer.pad_token_id]*(self.max_length-len(input_ids))
        labels=self.generate_labels(input_ids)
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    
        raise NotImplementedError("Implement in the SFT dataset lesson.")


class CourseDPODataset(Dataset):
    """TODO: build chosen/rejected training pairs.

    Align with: dataset/lm_dataset.py::DPODataset
    """

    def __init__(self, data_path: str, tokenizer, max_length: int):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        raise NotImplementedError("Implement in the DPO dataset lesson.")

    def __getitem__(self, index: int):
        raise NotImplementedError("Implement in the DPO dataset lesson.")
