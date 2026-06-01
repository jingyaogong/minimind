import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def text_of(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "paragraph_text", "passage_text", "snippet", "body"):
            if key in value:
                return str(value[key])
    return str(value)


def tokenize(text):
    text = str(text or "").lower()
    zh = re.findall(r"[\u4e00-\u9fff]", text)
    words = re.findall(r"[a-z0-9]+", text)
    return zh + words


def normalize_answers(raw):
    answers = []
    for item in as_list(raw):
        if isinstance(item, dict):
            for key in ("text", "answer", "value"):
                if key in item and str(item[key]).strip():
                    answers.append(str(item[key]).strip())
        elif str(item).strip():
            answers.append(str(item).strip())
    seen = set()
    dedup = []
    for answer in answers:
        key = answer.lower()
        if key not in seen:
            seen.add(key)
            dedup.append(answer)
    return dedup


def normalize_context_item(item, fallback_id):
    if isinstance(item, str):
        return {"id": fallback_id, "title": "", "text": item}
    doc_id = str(
        item.get("id")
        or item.get("doc_id")
        or item.get("pid")
        or item.get("passage_id")
        or item.get("wikipedia_id")
        or item.get("source")
        or fallback_id
    )
    title = str(item.get("title") or item.get("document_title") or item.get("name") or "")
    text = text_of(item)
    return {"id": doc_id, "title": title, "text": text}


def extract_contexts(sample):
    candidates = (
        sample.get("contexts")
        or sample.get("documents")
        or sample.get("snippets")
        or sample.get("search_results")
        or sample.get("ctxs")
        or sample.get("passages")
        or sample.get("positive_ctxs")
        or sample.get("retrieved_passages")
        or []
    )
    contexts = []
    for i, item in enumerate(candidates):
        doc = normalize_context_item(item, f"D{i + 1}")
        if doc["text"].strip():
            contexts.append(doc)
    return contexts


def extract_gold_doc_ids(sample, contexts):
    explicit = sample.get("gold_doc_ids") or sample.get("citations") or sample.get("supporting_doc_ids")
    context_ids = {doc["id"] for doc in contexts}
    if explicit:
        matched = [str(x) for x in as_list(explicit) if str(x) in context_ids]
        if matched:
            return matched
    ids = []
    for doc in contexts:
        text = doc["text"]
        if any(answer and answer in text for answer in extract_answers(sample)):
            ids.append(doc["id"])
    if not ids and contexts:
        ids.append(contexts[0]["id"])
    return ids[:3]


def keep_gold_contexts_first(contexts, gold_doc_ids, max_contexts):
    if max_contexts <= 0 or len(contexts) <= max_contexts:
        return contexts
    gold = set(str(x) for x in gold_doc_ids)
    positives = [doc for doc in contexts if doc["id"] in gold]
    others = [doc for doc in contexts if doc["id"] not in gold]
    return (positives + others)[:max_contexts]


def extract_question(sample):
    return str(
        sample.get("question")
        or sample.get("query")
        or sample.get("prompt")
        or sample.get("input")
        or sample.get("id")
        or ""
    ).strip()


def extract_answers(sample):
    raw = sample.get("answers")
    if raw is None:
        raw = sample.get("answer")
    if raw is None:
        raw = sample.get("short_answers")
    if raw is None:
        raw = sample.get("output")
    return normalize_answers(raw)


def normalize_sample(sample, source_name, index):
    question = extract_question(sample)
    answers = extract_answers(sample)
    contexts = extract_contexts(sample)
    if not question or not answers or not contexts:
        return None
    gold_doc_ids = extract_gold_doc_ids(sample, contexts)
    return {
        "id": str(sample.get("id") or f"{source_name}_{index}_{stable_id(question)}"),
        "source": source_name,
        "question": question,
        "answer": answers[0],
        "answers": answers,
        "gold_doc_ids": gold_doc_ids,
        "contexts": contexts,
    }


def collect_corpus(samples):
    corpus = {}
    for sample in samples:
        for doc in sample["contexts"]:
            if doc["id"] not in corpus and doc["text"].strip():
                corpus[doc["id"]] = doc
    return list(corpus.values())


def build_idf(corpus):
    df = defaultdict(int)
    for doc in corpus:
        for tok in set(tokenize(doc["title"] + " " + doc["text"])):
            df[tok] += 1
    n_docs = max(len(corpus), 1)
    return {tok: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0) for tok, freq in df.items()}


def bm25_score(query_tokens, doc_tokens, idf, avgdl, k1=1.2, b=0.75):
    if not doc_tokens:
        return 0.0
    tf = Counter(doc_tokens)
    score = 0.0
    dl = len(doc_tokens)
    denom_base = k1 * (1 - b + b * dl / max(avgdl, 1.0))
    for tok in query_tokens:
        freq = tf.get(tok, 0)
        if freq:
            score += idf.get(tok, 0.0) * freq * (k1 + 1) / (freq + denom_base)
    return score


def add_bm25_hard_negatives(samples, num_negatives, max_contexts, seed):
    if num_negatives <= 0:
        return samples
    rng = random.Random(seed)
    corpus = collect_corpus(samples)
    if not corpus:
        return samples
    idf = build_idf(corpus)
    doc_tokens = {doc["id"]: tokenize(doc["title"] + " " + doc["text"]) for doc in corpus}
    avgdl = sum(len(toks) for toks in doc_tokens.values()) / max(len(doc_tokens), 1)
    doc_by_id = {doc["id"]: doc for doc in corpus}

    for sample in samples:
        existing = {doc["id"] for doc in sample["contexts"]}
        gold = set(sample["gold_doc_ids"])
        query_tokens = tokenize(sample["question"] + " " + sample["answer"])
        scored = []
        for doc in corpus:
            if doc["id"] in existing or doc["id"] in gold:
                continue
            score = bm25_score(query_tokens, doc_tokens[doc["id"]], idf, avgdl)
            if score > 0:
                scored.append((score, doc["id"]))
        scored.sort(reverse=True)
        top_ids = [doc_id for _, doc_id in scored[: num_negatives * 4]]
        rng.shuffle(top_ids)
        negatives = [doc_by_id[doc_id] for doc_id in top_ids[:num_negatives]]
        sample["contexts"] = keep_gold_contexts_first(sample["contexts"] + negatives, sample["gold_doc_ids"], max_contexts)
    return samples


def add_embedding_hard_negatives(samples, num_negatives, max_contexts, seed, model_name, batch_size):
    if num_negatives <= 0:
        return samples
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Embedding negative mining requires sentence_transformers. Install project requirements first.") from exc

    rng = random.Random(seed)
    corpus = collect_corpus(samples)
    if not corpus:
        return samples

    model = SentenceTransformer(model_name)
    doc_texts = [(doc["title"] + "\n" + doc["text"]).strip() for doc in corpus]
    query_texts = [(sample["question"] + "\n" + sample["answer"]).strip() for sample in samples]
    doc_emb = model.encode(
        doc_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    query_emb = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    doc_emb = np.asarray(doc_emb, dtype=np.float32)
    query_emb = np.asarray(query_emb, dtype=np.float32)
    scores = query_emb @ doc_emb.T

    for i, sample in enumerate(samples):
        existing = {doc["id"] for doc in sample["contexts"]}
        gold = set(sample["gold_doc_ids"])
        ranked = np.argsort(-scores[i])
        candidate_ids = []
        for doc_idx in ranked:
            doc_id = corpus[int(doc_idx)]["id"]
            if doc_id in existing or doc_id in gold:
                continue
            candidate_ids.append(doc_idx)
            if len(candidate_ids) >= num_negatives * 4:
                break
        rng.shuffle(candidate_ids)
        negatives = [corpus[int(doc_idx)] for doc_idx in candidate_ids[:num_negatives]]
        sample["contexts"] = keep_gold_contexts_first(sample["contexts"] + negatives, sample["gold_doc_ids"], max_contexts)
    return samples


def add_hard_negatives(samples, num_negatives, max_contexts, seed, method="bm25", embedding_model="", embedding_batch_size=64):
    if method == "bm25":
        return add_bm25_hard_negatives(samples, num_negatives, max_contexts, seed)
    if method == "embedding":
        return add_embedding_hard_negatives(samples, num_negatives, max_contexts, seed, embedding_model, embedding_batch_size)
    raise ValueError(f"Unsupported negative_mining method: {method}")


def split_rows(rows, train_size, dev_size, test_size, seed):
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    if train_size:
        train = rows[:train_size]
        rest = rows[train_size:]
    else:
        train, rest = rows, []
    dev = rest[:dev_size] if dev_size else []
    rest = rest[dev_size:] if dev_size else rest
    test = rest[:test_size] if test_size else []
    return train, dev, test


def load_sources(paths):
    rows = []
    for path in paths:
        source_name = os.path.splitext(os.path.basename(path))[0]
        for idx, sample in enumerate(iter_jsonl(path)):
            row = normalize_sample(sample, source_name, idx)
            if row is not None:
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build SearchShortQA JSONL from public QA/RAG style datasets")
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more local JSONL files")
    parser.add_argument("--output_dir", default="dataset/search_shortqa")
    parser.add_argument("--train_size", type=int, default=30000)
    parser.add_argument("--dev_size", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--max_contexts", type=int, default=6)
    parser.add_argument("--num_negatives", type=int, default=3)
    parser.add_argument("--negative_mining", choices=["bm25", "embedding"], default="bm25")
    parser.add_argument("--embedding_model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--embedding_batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = load_sources(args.inputs)
    if args.limit:
        rows = rows[: args.limit]
    rows = add_hard_negatives(
        rows,
        args.num_negatives,
        args.max_contexts,
        args.seed,
        method=args.negative_mining,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
    )
    rows = [row for row in rows if row["question"] and row["answers"] and row["contexts"]]
    train, dev, test = split_rows(rows, args.train_size, args.dev_size, args.test_size, args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    write_jsonl(os.path.join(args.output_dir, "search_shortqa_train.jsonl"), train)
    write_jsonl(os.path.join(args.output_dir, "search_shortqa_dev.jsonl"), dev)
    write_jsonl(os.path.join(args.output_dir, "search_shortqa_test.jsonl"), test)

    manifest = {
        "inputs": args.inputs,
        "total_normalized": len(rows),
        "train": len(train),
        "dev": len(dev),
        "test": len(test),
        "max_contexts": args.max_contexts,
        "num_negatives": args.num_negatives,
        "negative_mining": args.negative_mining,
        "embedding_model": args.embedding_model if args.negative_mining == "embedding" else "",
        "seed": args.seed,
        "format": "SearchShortQA v1",
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
