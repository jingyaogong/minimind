"""Compare the saved legacy retriever and current retrieval on fixed local cases.

This measures document access and evidence handling, not protocol correctness.
The legacy class is extracted with AST to avoid importing/loading its model.
"""

import ast
import hashlib
import json
import os
import re
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from agent_core import SpecRetrieverTool
from rag_evidence import ROOT

CASES = [
    ("single_docx", "解释 R2-2408133", ["R2-2408133"], True),
    ("two_docx", "对比 R2-2408133 和 R2-2408134", ["R2-2408133", "R2-2408134"], True),
    ("three_docx", "比较 R2-2408133 R2-2408134 R2-2508703", ["R2-2408133", "R2-2408134", "R2-2508703"], True),
    ("unicode_id", "分析 r2–2408133", ["R2-2408133"], True),
    ("missing_id", "解释 R2-2999999 的 XR 方案", [], False),
    ("legacy_doc", "解释 R2-2408091", [], False),
    ("partial_comparison", "对比 R2-2408133 和 R2-2999999", ["R2-2408133"], False),
    ("title_search", "Essential requirements for NTN and way forward in the common design with TN", ["R2-2508703"], True),
]


def main():
    backup = ROOT / "output/agent_upgrade_backup_20261021/agent_core.py"
    source = backup.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SpecRetrieverTool")
    scope = {"os": os, "re": re, "json": json, "zipfile": zipfile, "ET": ET}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(backup), "exec"), scope)
    docs = ROOT / "3gpp-tdoc-batch-downloader/RAN2_Rel19_Rel20_Docs"
    legacy = scope["SpecRetrieverTool"](
        proposal_db_path=str(ROOT / "dataset/ran2_train_382.jsonl"),
        tdoc_cache_path=str(ROOT / "tdoc_index_cache.json"), docs_root=str(docs),
    )
    started = time.perf_counter()
    current = SpecRetrieverTool()
    report = {
        "scope": "Fixed local regression cases only; no answer-accuracy or semantic-support claim.",
        "baseline_sha256": hashlib.sha256(backup.read_bytes()).hexdigest(),
        "current_init_seconds": time.perf_counter() - started,
        "index_entries": len(current.tdoc_index), "cases": [],
    }
    for name, query, expected, should_answer in CASES:
        old = legacy.retrieve(query)  # Preserve the original default top_k=1.
        old_body_ids = {
            match for context in old if "原文片段摘要" in context
            for match in re.findall(r"提案编号: (R[1-4]-\d{6,7})", context)
        }
        started = time.perf_counter()
        new = current.retrieve_evidence(query)
        new_body_ids = {item.tdoc for item in new.evidence}
        old_coverage = set(expected).issubset(old_body_ids)
        new_coverage = set(expected).issubset(new_body_ids)
        report["cases"].append({
            "name": name, "query": query, "expected_body_ids": expected,
            "should_generate": should_answer,
            "before": {"body_ids": sorted(old_body_ids), "generates_without_source_gate": True,
                       "passes": old_coverage and should_answer},
            "after": {"body_ids": sorted(new_body_ids), "answerable": new.answerable,
                      "passes": new_coverage and new.answerable == should_answer,
                      "retrieval_seconds": time.perf_counter() - started,
                      "documents": new.documents, "warnings": new.warnings},
        })
    report["before_passed"] = sum(case["before"]["passes"] for case in report["cases"])
    report["after_passed"] = sum(case["after"]["passes"] for case in report["cases"])
    destination = ROOT / "output/agent_retrieval_regression.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False, indent=2))
    print(f"Saved: {destination}")
    if report["after_passed"] != len(CASES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
