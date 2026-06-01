import json
import os
from typing import Any, Dict, List

from torch.utils.data import Dataset
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"


SEARCH_SHORTQA_SYSTEM = (
    "你是一个搜索增强短问答助手。你只能根据给定检索片段回答问题；"
    "如果证据不足，回答“不确定”。输出必须是JSON，格式为"
    "{\"answer\":\"...\", \"citations\":[\"D1\"]}。answer必须简短，citations只能引用给定片段ID。"
)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_contexts(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    contexts = (
        sample.get("contexts")
        or sample.get("documents")
        or sample.get("ctxs")
        or sample.get("passages")
        or sample.get("snippets")
        or sample.get("search_results")
        or []
    )
    normalized = []
    for i, item in enumerate(contexts):
        if isinstance(item, str):
            normalized.append({"id": f"D{i + 1}", "title": "", "text": item})
            continue
        doc_id = str(item.get("id") or item.get("doc_id") or item.get("source") or f"D{i + 1}")
        title = str(item.get("title") or item.get("name") or "")
        text = str(item.get("text") or item.get("content") or item.get("snippet") or "")
        normalized.append({"id": doc_id, "title": title, "text": text})
    return normalized


def normalize_answers(sample: Dict[str, Any]) -> List[str]:
    answers = sample.get("answers")
    if answers is None:
        answers = sample.get("answer")
    if answers is None:
        answers = sample.get("short_answers")
    if answers is None:
        answers = sample.get("output")
    return [str(a) for a in _as_list(answers) if str(a).strip()]


def normalize_gold_doc_ids(sample: Dict[str, Any]) -> List[str]:
    doc_ids = sample.get("gold_doc_ids") or sample.get("citations") or sample.get("supporting_doc_ids")
    return [str(x) for x in _as_list(doc_ids)]


def select_contexts(sample: Dict[str, Any], max_contexts=6) -> List[Dict[str, str]]:
    contexts = normalize_contexts(sample)
    if max_contexts <= 0 or len(contexts) <= max_contexts:
        return contexts
    gold_ids = set(normalize_gold_doc_ids(sample))
    gold_contexts = [doc for doc in contexts if doc["id"] in gold_ids]
    other_contexts = [doc for doc in contexts if doc["id"] not in gold_ids]
    return (gold_contexts + other_contexts)[:max_contexts]


def format_search_prompt(sample: Dict[str, Any], max_contexts=6, max_chars_per_context=700) -> str:
    question = str(sample.get("question") or sample.get("query") or sample.get("prompt") or "")
    contexts = select_contexts(sample, max_contexts)
    lines = ["请基于以下检索片段回答问题。", "", f"问题：{question}", "", "检索片段："]
    for doc in contexts:
        title = f"｜{doc['title']}" if doc["title"] else ""
        text = doc["text"].replace("\n", " ").strip()[:max_chars_per_context]
        lines.append(f"[{doc['id']}]{title} {text}")
    lines.extend(["", "要求：输出JSON；answer简短；citations只填写真正支撑答案的片段ID。"])
    return "\n".join(lines)


def build_search_shortqa_messages(sample: Dict[str, Any], max_contexts=6, max_chars_per_context=700):
    return [
        {"role": "system", "content": SEARCH_SHORTQA_SYSTEM},
        {"role": "user", "content": format_search_prompt(sample, max_contexts, max_chars_per_context)},
    ]


def build_sft_conversation(sample: Dict[str, Any], max_contexts=6, max_chars_per_context=700):
    answers = normalize_answers(sample)
    gold_doc_ids = normalize_gold_doc_ids(sample)
    answer = answers[0] if answers else "不确定"
    assistant = {
        "answer": answer,
        "citations": gold_doc_ids[:3],
    }
    return build_search_shortqa_messages(sample, max_contexts, max_chars_per_context) + [
        {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)}
    ]


class SearchShortQARLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, max_contexts=6, max_chars_per_context=700):
        super().__init__()
        if load_dataset is None:
            raise ImportError("SearchShortQARLDataset requires the 'datasets' package. Install project requirements before training.")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_contexts = max_contexts
        self.max_chars_per_context = max_chars_per_context
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = dict(self.samples[index])
        messages = build_search_shortqa_messages(sample, self.max_contexts, self.max_chars_per_context)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=False,
        )
        return {
            "prompt": prompt,
            "question": str(sample.get("question") or sample.get("query") or ""),
            "answers": normalize_answers(sample),
            "gold_doc_ids": normalize_gold_doc_ids(sample),
            "contexts": select_contexts(sample, self.max_contexts),
        }
