import time
from typing import Dict, Iterable, List, Optional

from datasets import load_dataset


def load_open_corpora(split: str = "train") -> Iterable[Dict]:
    """Unified loader: iterate over a list of HF datasets and yield raw rows.

    Keeps original keys. Adds a 'source' field if missing for provenance.
    """
    datasets_to_try: List[str] = [
        # Wikipedia
        "wikimedia/wikipedia",
        # arXiv (abstract-related)
        "ccdv/arxiv-summarization",
        "MMInstruction/ArxivQA",
        # StackOverflow
        "DmitriyGA/DPO-StackOverflow",
    ]
    for ds_name in datasets_to_try:
        try:
            ds = load_dataset(ds_name, split=split, trust_remote_code=True)
        except Exception:
            continue
        for row in ds:
            rec = dict(row)
            if "source" not in rec:
                rec["source"] = ds_name
            yield rec


def multiplex_sources(max_items: Optional[int] = None, **kwargs) -> Iterable[Dict]:
    """Multiplex multiple open sources in a single iterator.

    kwargs can carry filters like language or categories.
    """
    count = 0
    for item in load_open_corpora(split=kwargs.get("split", "train")):
        yield item
        count += 1
        if max_items and count >= max_items:
            return
