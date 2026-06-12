import argparse
import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentic.data_analysis_env import extract_final_answer, score_agentic_trajectory
from dataset.agentic_dataset import load_agentic_jsonl


DEFAULT_DATA = "dataset/agentic_data/dev.jsonl"


def trajectory_to_text(sample):
    chunks = []
    for item in sample.get("expert_trajectory", []):
        if item.get("role") == "assistant" and item.get("tool_call"):
            chunks.append("<tool_call>\n" + json.dumps(item["tool_call"], ensure_ascii=False, indent=2) + "\n</tool_call>")
        elif item.get("role") == "assistant":
            chunks.append(str(item.get("content", "")))
    return "\n---\n".join(chunks)


def parse_turns(text):
    if "\n---\n" in text:
        return [part.strip() for part in text.split("\n---\n") if part.strip()]
    return [text.strip()] if text.strip() else []


def parse_args():
    parser = argparse.ArgumentParser(description="Agentic DataAnalysis Streamlit demo")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--repo_root", default=".")
    return parser.parse_args()


def main():
    args = parse_args()
    st.set_page_config(page_title="Agentic DataAnalysis", layout="wide")
    st.title("Agentic DataAnalysis")

    rows = load_agentic_jsonl(args.data) if os.path.exists(args.data) else []
    if not rows:
        st.error(f"数据不存在或为空：{args.data}")
        return

    sample_ids = [row.get("id", str(i)) for i, row in enumerate(rows)]
    selected = st.sidebar.selectbox("样本", sample_ids)
    sample = rows[sample_ids.index(selected)]

    st.subheader("问题")
    st.write(sample.get("question", ""))

    left, right = st.columns([1.1, 1.0])
    with left:
        st.subheader("数据表")
        st.dataframe(pd.DataFrame(sample.get("tables", [])), use_container_width=True)
        st.subheader("指标文档")
        st.dataframe(pd.DataFrame(sample.get("documents", [])), use_container_width=True)

    with right:
        st.subheader("标准答案")
        st.write(sample.get("answer", ""))
        st.json(sample.get("checks", {}))
        st.subheader("期望工具")
        st.write(", ".join(sample.get("expected_tools", [])))

    st.subheader("轨迹与奖励")
    default_text = trajectory_to_text(sample)
    trajectory_text = st.text_area("轨迹文本（用 --- 分隔多个 assistant turn 可手动评测模型输出）", value=default_text, height=360)
    turn_outputs = [
        part
        for part in parse_turns(trajectory_text)
        if "<tool_response>" not in part
    ]
    reward, parts = score_agentic_trajectory(turn_outputs, sample, repo_root=args.repo_root)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Reward", f"{reward:.3f}")
    c2.metric("Success", f"{parts['task_success']:.3f}")
    c3.metric("Tool F1", f"{parts['tool_selection_f1']:.3f}")
    c4.metric("Invalid", f"{parts['invalid_action_rate']:.3f}")
    c5.metric("Turns", f"{parts['avg_turns']:.1f}")

    st.subheader("Reward 分解")
    st.dataframe(pd.DataFrame([parts]).T.rename(columns={0: "score"}), use_container_width=True)

    st.subheader("Final Answer")
    st.write(extract_final_answer(turn_outputs))


if __name__ == "__main__":
    main()
