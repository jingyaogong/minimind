import json

from torch.utils.data import Dataset

from agentic.data_analysis_env import AGENTIC_SYSTEM_PROMPT, format_agentic_user_prompt, get_agentic_tools


def load_agentic_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class AgenticRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer=None, max_length=2048):
        super().__init__()
        self.samples = load_agentic_jsonl(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tools = get_agentic_tools(sample.get("tools"))
        messages = [
            {"role": "system", "content": AGENTIC_SYSTEM_PROMPT},
            {"role": "user", "content": format_agentic_user_prompt(sample)},
        ]
        return {
            "id": sample.get("id", str(index)),
            "messages": messages,
            "tools": tools,
            "sample": sample,
        }
