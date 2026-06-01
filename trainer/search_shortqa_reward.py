import json
import re
import string
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple


def normalize_text(text: Any) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(string.punctuation + "，。！？；：“”‘’（）【】《》、")


def tokenize_text(text: Any) -> List[str]:
    text = normalize_text(text)
    zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
    words = re.findall(r"[a-z0-9]+", text)
    return zh_chars + words


def token_f1(prediction: str, gold_answers: Iterable[str]) -> float:
    pred_tokens = tokenize_text(prediction)
    if not pred_tokens:
        return 0.0
    best = 0.0
    for answer in gold_answers:
        gold_tokens = tokenize_text(answer)
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def exact_match(prediction: str, gold_answers: Iterable[str]) -> float:
    pred = normalize_text(prediction)
    return float(any(pred == normalize_text(answer) for answer in gold_answers))


def parse_search_response(text: str) -> Tuple[str, List[str], bool]:
    text = str(text or "").strip()
    json_text = text
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        json_text = fenced.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_text = match.group(0)
    try:
        data = json.loads(json_text)
        answer = str(data.get("answer", "")).strip()
        citations = data.get("citations", [])
        if isinstance(citations, str):
            citations = [citations]
        citations = [str(x) for x in citations]
        return answer, citations, True
    except Exception:
        return text, [], False


def repetition_penalty(text: str, n=3, cap=0.4) -> float:
    toks = tokenize_text(text)
    if len(toks) < n:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) / max(len(grams), 1))


def citation_metrics(citations: List[str], gold_doc_ids: Iterable[str], valid_doc_ids: Iterable[str]) -> Dict[str, float]:
    pred = [str(x) for x in citations]
    gold = set(str(x) for x in gold_doc_ids)
    valid = set(str(x) for x in valid_doc_ids)
    if not pred:
        return {"citation_precision": 0.0, "citation_recall": 0.0, "citation_valid": 0.0}
    valid_rate = sum(1 for x in pred if x in valid) / len(pred)
    if not gold:
        return {"citation_precision": valid_rate, "citation_recall": 1.0, "citation_valid": valid_rate}
    hit = sum(1 for x in pred if x in gold)
    return {
        "citation_precision": hit / len(pred),
        "citation_recall": hit / len(gold),
        "citation_valid": valid_rate,
    }


def score_search_shortqa_response(
    response: str,
    answers: Iterable[str],
    gold_doc_ids: Iterable[str],
    contexts: List[Dict[str, Any]],
    weights=None,
) -> Tuple[float, Dict[str, float]]:
    weights = weights or {
        "answer_f1": 0.45,
        "exact_match": 0.20,
        "citation": 0.20,
        "format": 0.10,
        "brevity": 0.05,
    }
    answer, citations, is_json = parse_search_response(response)
    valid_doc_ids = [doc.get("id") for doc in contexts]
    cite = citation_metrics(citations, gold_doc_ids, valid_doc_ids)

    answer_f1 = token_f1(answer, answers)
    em = exact_match(answer, answers)
    format_score = 1.0 if is_json and isinstance(citations, list) and answer else 0.0
    answer_len = len(answer)
    brevity = 1.0 if 1 <= answer_len <= 80 else max(0.0, 1.0 - abs(answer_len - 80) / 200)
    repeat = repetition_penalty(answer)
    citation_score = 0.5 * cite["citation_precision"] + 0.3 * cite["citation_recall"] + 0.2 * cite["citation_valid"]

    total = (
        weights["answer_f1"] * answer_f1
        + weights["exact_match"] * em
        + weights["citation"] * citation_score
        + weights["format"] * format_score
        + weights["brevity"] * brevity
        - 0.20 * repeat
    )
    parts = {
        "reward_total": total,
        "answer_f1": answer_f1,
        "exact_match": em,
        "citation_score": citation_score,
        "format_score": format_score,
        "brevity_score": brevity,
        "repetition_penalty": repeat,
    }
    parts.update(cite)
    return float(total), parts


def average_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    keys = sorted({key for item in metrics for key in item})
    return {key: sum(float(item.get(key, 0.0)) for item in metrics) / len(metrics) for key in keys}
