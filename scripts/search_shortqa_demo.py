import argparse
import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.search_shortqa_dataset import normalize_answers, normalize_contexts, normalize_gold_doc_ids
from trainer.search_shortqa_reward import parse_search_response, score_search_shortqa_response


DEFAULT_DATA = "dataset/examples/search_shortqa_raw_demo.jsonl"


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def context_table(contexts):
    return pd.DataFrame(
        [
            {
                "id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }
            for doc in contexts
        ]
    )


def row_from_editor(sample, table):
    row = dict(sample)
    row["contexts"] = [
        {
            "id": str(item.get("id", "")).strip(),
            "title": str(item.get("title", "")).strip(),
            "text": str(item.get("text", "")).strip(),
        }
        for item in table.to_dict("records")
        if str(item.get("id", "")).strip() and str(item.get("text", "")).strip()
    ]
    return row


def default_prediction(sample):
    answers = normalize_answers(sample)
    gold_doc_ids = normalize_gold_doc_ids(sample)
    return json.dumps(
        {
            "answer": answers[0] if answers else "不确定",
            "citations": gold_doc_ids[:3],
        },
        ensure_ascii=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="SearchShortQA Streamlit demo")
    parser.add_argument("--data", default=DEFAULT_DATA)
    return parser.parse_args()


def main():
    args = parse_args()
    st.set_page_config(page_title="SearchShortQA", layout="wide")

    st.title("SearchShortQA")

    with st.sidebar:
        data_path = st.text_input("数据文件", value=args.data)
        rows = load_jsonl(data_path)
        if not rows:
            st.error("数据为空")
            return
        sample_ids = [str(row.get("id") or i) for i, row in enumerate(rows)]
        selected_id = st.selectbox("样本", sample_ids, index=0)

    sample = rows[sample_ids.index(selected_id)]
    sample = dict(sample)
    question = st.text_input("问题", value=str(sample.get("question") or sample.get("query") or sample.get("prompt") or ""))
    sample["question"] = question

    left, right = st.columns([1.2, 1.0])
    with left:
        st.subheader("检索片段")
        edited_contexts = st.data_editor(
            context_table(normalize_contexts(sample)),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "id": st.column_config.TextColumn("ID", width="small"),
                "title": st.column_config.TextColumn("标题", width="medium"),
                "text": st.column_config.TextColumn("正文", width="large"),
            },
        )
        sample = row_from_editor(sample, edited_contexts)

    with right:
        st.subheader("模型输出")
        prediction = st.text_area("JSON", value=default_prediction(sample), height=180)
        reward, parts = score_search_shortqa_response(
            prediction,
            answers=normalize_answers(sample),
            gold_doc_ids=normalize_gold_doc_ids(sample),
            contexts=normalize_contexts(sample),
        )
        answer, citations, is_json = parse_search_response(prediction)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reward", f"{reward:.3f}")
        c2.metric("F1", f"{parts['answer_f1']:.3f}")
        c3.metric("引用合法率", f"{parts['citation_valid']:.3f}")
        c4.metric("格式", f"{parts['format_score']:.3f}")

        st.subheader("奖励分解")
        st.dataframe(
            pd.DataFrame([parts]).T.rename(columns={0: "score"}),
            use_container_width=True,
        )

    st.subheader("引用检查")
    valid_ids = {doc["id"] for doc in normalize_contexts(sample)}
    cited = set(citations)
    citation_rows = []
    for doc in normalize_contexts(sample):
        citation_rows.append(
            {
                "id": doc["id"],
                "title": doc["title"],
                "cited": doc["id"] in cited,
                "gold": doc["id"] in set(normalize_gold_doc_ids(sample)),
                "valid": doc["id"] in valid_ids,
                "text": doc["text"],
            }
        )
    st.dataframe(pd.DataFrame(citation_rows), use_container_width=True)

    st.subheader("答案")
    st.write(answer if is_json else prediction)


if __name__ == "__main__":
    main()
