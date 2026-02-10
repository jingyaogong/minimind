import argparse
import os
import sys
from typing import Dict, Iterable, List
from tqdm import tqdm

if __package__ is None or __package__ == "":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from dataset.corpus.sources import multiplex_sources
from dataset.corpus.cleaning import Deduper, clean_record
from dataset.utils.util import write_jsonl


def _aggregate_text(rec: Dict) -> str:
    parts: List[str] = []
    for k, v in rec.items():
        if isinstance(v, str):
            if v.strip():
                parts.append(v)
        elif isinstance(v, list):
            # flatten list of strings (best-effort)
            parts.extend([str(x) for x in v if isinstance(x, (str, int, float)) and str(x).strip()])
        elif isinstance(v, (int, float)):
            parts.append(str(v))
    text = " \n".join(parts)
    # cap extremely long text to avoid excessive memory
    if len(text) > 20000:
        text = text[:20000]
    return text


def shard_records(records: Iterable[Dict], shard_size: int, out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    buf: List[Dict] = []
    shard_id = 0
    for rec in records:
        if not rec:
            continue
        buf.append(rec)
        if len(buf) >= shard_size:
            out_path = os.path.join(out_dir, f"{prefix}-{shard_id:05d}.jsonl")
            write_jsonl(out_path, buf)
            buf.clear()
            shard_id += 1
    if buf:
        out_path = os.path.join(out_dir, f"{prefix}-{shard_id:05d}.jsonl")
        write_jsonl(out_path, buf)


def build_pipeline(target_lang: str = "zh", max_items: int = 5000, shard_size: int = 1000, out_dir: str = "dataset/out/corpus_out"):
    deduper = Deduper(threshold=0.88)

    def cleaned_stream():
        for idx, rec in enumerate(tqdm(multiplex_sources(max_items=max_items, language=target_lang), total=max_items, desc=f"ingest[{target_lang}]")):
            key = f"{rec.get('source','unknown')}:{rec.get('title','')}-{idx}"
            agg_text = _aggregate_text(rec)
            if not agg_text:
                continue
            # Use cleaning on the aggregated text only for filtering, not for reformatting
            tmp = {"text": agg_text, "source": rec.get("source"), "title": rec.get("title")}
            cleaned = clean_record(tmp, target_lang=target_lang)
            if not cleaned:
                continue
            if deduper.is_duplicate(key, agg_text):
                continue
            # Keep original record shape as requested
            yield rec

    count_in = 0
    count_out = 0
    records_iter = cleaned_stream()
    # Materialize in small chunks to measure counts
    chunk: List[Dict] = []
    for rec in tqdm(records_iter, desc="clean+dedup"):
        count_in += 1
        chunk.append(rec)
        if len(chunk) >= shard_size:
            shard_records(chunk, shard_size=shard_size, out_dir=out_dir, prefix=f"corpus-{target_lang}")
            count_out += len(chunk)
            chunk.clear()
    if chunk:
        shard_records(chunk, shard_size=shard_size, out_dir=out_dir, prefix=f"corpus-{target_lang}")
        count_out += len(chunk)
    print(f"[build_corpus] input_records={count_in} written_records={count_out} out_dir={out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build cleaned corpus from open sources.")
    parser.add_argument("--lang", default="en", help="Target language code, e.g., zh, en")
    parser.add_argument("--max-items", type=int, default=5000, help="Max items to ingest across sources")
    parser.add_argument("--shard-size", type=int, default=1000, help="Number of records per shard")
    parser.add_argument("--out-dir", default="dataset/out/corpus_out", help="Output directory for JSONL shards")
    args = parser.parse_args()

    build_pipeline(target_lang=args.lang, max_items=args.max_items, shard_size=args.shard_size, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
