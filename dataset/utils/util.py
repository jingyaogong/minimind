from typing import Dict, Iterable, List
import os
import json


def write_jsonl(path: str, records: Iterable[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            if not rec:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")