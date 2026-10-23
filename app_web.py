"""Streamlit UI for inspecting sources and asking evidence-grounded TDoc questions."""

import json

import streamlit as st

from agent_core import ProposalAgent

st.set_page_config(page_title="3GPP 提案阅读助手", page_icon="📡", layout="wide")


@st.cache_resource(show_spinner="正在建立本地提案检索索引……")
def load_agent():
    return ProposalAgent()


def show_sources(sources):
    """Keep source excerpts visible independently of model loading or generation errors."""
    for warning in sources.get("warnings", []):
        st.warning(warning)
    if sources.get("documents"):
        st.dataframe([
            {"提案": item["tdoc"], "标题": item.get("title", ""), "原文状态": item["status"]}
            for item in sources["documents"]
        ], hide_index=True, width="stretch")
    for item in sources.get("evidence", []):
        with st.expander(f"[{item['source_id']}] {item['tdoc']} · 段落 {item['paragraph_start']}–{item['paragraph_end']}"):
            st.write(item["title"])
            st.caption(f"{item['meeting']} | {item['company']}")
            st.text(item["text"])
            st.caption(item["file_path"])


st.title("📡 3GPP 提案阅读助手")
st.caption("查找本地提案原文，查看依据，再生成带来源编号的回答。提案主张不等于已通过的标准。")

with st.sidebar:
    st.subheader("使用方式")
    retrieval_only = st.checkbox("只查原文，不调用模型", value=True)
    use_lora = st.checkbox("使用现有 LoRA 适配器", value=False, help="默认使用原模型。本轮少量实测中，现有适配器出现了引用缺失和无依据的流程描述；其整体效果尚未确认。")
    max_tokens = st.slider("最长生成 token 数", 128, 1536, 640, step=128)
    temperature = st.slider("采样温度", 0.0, 0.7, 0.0, step=0.1)
    st.caption("设为 0 可减少采样变化；不能保证事实正确。首次生成会加载本地模型。")
    st.divider()
    st.subheader("示例问题")
    examples = {
        "XR 速率控制": "请根据 R2-2408133 原文，说明提案要解决的问题、主要主张及尚未明确的部分。",
        "比较两篇 XR 提案": "请对比 R2-2408133 和 R2-2408134，分别说明目标和主要主张，再解释两篇提案是否解决同一个问题。",
        "NTN 设计要求": "请根据 R2-2508703 原文，说明 NTN 对 TN/NTN 共同设计提出了哪些要求。区分提案主张与已经达成的结论。",
        "按标题找资料": "XR rate control signaling",
    }
    for label, example in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state["question"] = example
    if st.button("清除结果", use_container_width=True):
        st.session_state.pop("last_run", None)
        st.session_state.pop("last_error", None)
        st.session_state.pop("last_sources", None)
    st.caption("支持编号检索、标题关键词排序和 DOCX 原文读取。旧版 .doc 暂不读取；缺少原文时不生成内容分析。")

question = st.text_area("问题或提案编号", key="question", height=130, placeholder="例如：对比 R2-2408133 和 R2-2408134 的目标与主张。")
extra = st.text_area("补充材料（选填，会标记为未经核实）", height=80)

if st.button("检索原文" if retrieval_only else "检索并分析", type="primary"):
    st.session_state.pop("last_run", None)
    st.session_state.pop("last_error", None)
    st.session_state.pop("last_sources", None)
    if not question.strip():
        st.warning("请输入问题或提案编号。")
    else:
        status = st.status("正在处理……", expanded=True)
        with st.container():
            source_area = st.container()
            output = st.empty()
        completed = False
        try:
            agent = load_agent()
            for event in agent.run_stream(
                question, extra_context=extra, max_new_tokens=max_tokens,
                temperature=temperature, retrieval_only=retrieval_only, use_lora=use_lora,
            ):
                stage = event["stage"]
                if stage in {"1. RETRIEVAL", "1. RETRIEVAL_DONE", "2. SYNTHESIS_START"}:
                    status.write(event["msg"])
                if stage == "1. RETRIEVAL_DONE":
                    st.session_state["last_sources"] = event["retrieval"]
                    with source_area:
                        show_sources(event["retrieval"])
                elif stage == "2. STREAMING":
                    output.markdown(event["full_text"] + " ▌")
                elif stage == "4. COMPLETE":
                    st.session_state["last_run"] = {**event, "question": question, "use_lora": use_lora}
                    completed = True
                elif stage == "ERROR":
                    st.session_state["last_error"] = event["msg"]
        except Exception as exc:
            st.session_state["last_error"] = f"初始化失败：{exc}"
        output.empty()
        status.update(label="处理完成" if completed else "处理失败", state="complete" if completed else "error", expanded=False)
        if st.session_state.get("last_error"):
            st.warning("生成未完成。上方原文仍可查看；不完整回答未保存为结果。")

if st.session_state.get("last_error"):
    st.error(st.session_state["last_error"])
    if st.session_state.get("last_sources"):
        with st.expander("查看本次已找到的原文"):
            show_sources(st.session_state["last_sources"])

if run := st.session_state.get("last_run"):
    st.divider()
    if run.get("mode") == "needs_review":
        st.warning("以下为待核对草稿：来源编号或提案引用不完整。")
    st.caption("对应问题：" + run.get("question", ""))
    st.markdown(run["result"])
    with st.expander("查看所有来源及原文"):
        show_sources(run["sources"])
    for warning in run.get("validation", {}).get("warnings", []):
        st.warning(warning)
    metrics = run.get("metrics", {})
    labels = {"retrieval_seconds": "检索", "first_text_seconds": "首段文本", "total_seconds": "总耗时"}
    st.caption(" | ".join(f"{labels.get(name, name)}: {value:.2f} s" for name, value in metrics.items() if value is not None))
    st.caption("首段文本时间包含检索和首次模型加载，不等于纯模型首 token 延迟。")
    report = run["result"] + "\n\n---\n\n来源记录：\n\n" + "\n\n".join(
        f"[{item['source_id']}] {item['tdoc']} | {item['title']}\n"
        f"{item['file_path']}（段落 {item['paragraph_start']}–{item['paragraph_end']}）\n{item['text']}"
        for item in run["sources"].get("evidence", [])
    )
    st.download_button("下载回答与来源 Markdown", report, "3gpp_analysis.md", "text/markdown")
    st.download_button("下载完整运行记录 JSON", json.dumps(run, ensure_ascii=False, indent=2), "3gpp_run.json", "application/json")
