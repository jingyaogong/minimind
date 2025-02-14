import random
import re
import time

import numpy as np
import streamlit as st
import torch

st.set_page_config(page_title="MiniMind", initial_sidebar_state="collapsed")

# 在文件开头的 CSS 样式中修改按钮样式
st.markdown("""
    <style>
        /* 添加操作按钮样式 */
        .stButton button {
            border-radius: 50% !important;  /* 改为圆形 */
            width: 32px !important;         /* 固定宽度 */
            height: 32px !important;        /* 固定高度 */
            padding: 0 !important;          /* 移除内边距 */
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;         /* 更柔和的颜色 */
            margin: 5px 10px 5px 0 !important;  /* 调整按钮间距 */
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        .stMainBlockContainer > div:first-child {
            margin-top: -50px !important;
        }
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }
        
        /* 重置按钮基础样式 */
        .stButton > button {
            all: unset !important;  /* 重置所有默认样式 */
            box-sizing: border-box !important;
            border-radius: 50% !important;
            width: 18px !important;
            height: 18px !important;
            min-width: 18px !important;
            min-height: 18px !important;
            max-width: 18px !important;
            max-height: 18px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #888 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            margin: 0 2px !important;  /* 调整这里的 margin 值 */
        }

    </style>
""", unsafe_allow_html=True)

system_prompt = []
device = "cuda" if torch.cuda.is_available() else "cpu"


def process_assistant_content(content):
    if 'R1' not in MODEL_PATHS[selected_model][1]:
        return content

    if '<think>' in content and '</think>' in content:
        content = re.sub(r'(<think>)(.*?)(</think>)',
                         r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理内容（展开）</summary>\2</details>',
                         content,
                         flags=re.DOTALL)

    if '<think>' in content and '</think>' not in content:
        content = re.sub(r'<think>(.*?)$',
                         r'<details open style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理中...</summary>\1</details>',
                         content,
                         flags=re.DOTALL)

    if '<think>' not in content and '</think>' in content:
        content = re.sub(r'(.*?)</think>',
                         r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理内容（展开）</summary>\1</details>',
                         content,
                         flags=re.DOTALL)

    return content


@st.cache_resource
def load_model_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    model = model.eval().to(device)
    return model, tokenizer


def clear_chat_messages():
    del st.session_state.messages
    del st.session_state.chat_messages


def init_chat_messages():
    if "messages" in st.session_state:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                with st.chat_message("assistant", avatar=image_url):
                    st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
                    # 在消息内容下方添加按钮
                    if st.button("🗑", key=f"delete_{i}"):
                        st.session_state.messages.pop(i)
                        st.session_state.messages.pop(i - 1)
                        st.session_state.chat_messages.pop(i)
                        st.session_state.chat_messages.pop(i - 1)
                        st.rerun()
            else:
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: #ddd; border-radius: 10px; color: black;">{message["content"]}</div></div>',
                    unsafe_allow_html=True)

    else:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    return st.session_state.messages


# 添加这两个辅助函数
def regenerate_answer(index):
    st.session_state.messages.pop()
    st.session_state.chat_messages.pop()
    st.rerun()


def delete_conversation(index):
    st.session_state.messages.pop(index)
    st.session_state.messages.pop(index - 1)
    st.session_state.chat_messages.pop(index)
    st.session_state.chat_messages.pop(index - 1)
    st.rerun()


# 侧边栏模型选择
st.sidebar.title("模型设定调整")

st.sidebar.text("【注】训练数据偏差，增加上下文记忆时\n多轮对话（较单轮）容易出现能力衰减")
st.session_state.history_chat_num = st.sidebar.slider("Number of Historical Dialogues", 0, 6, 0, step=2)
# st.session_state.history_chat_num = 0
st.session_state.max_new_tokens = st.sidebar.slider("Max Sequence Length", 256, 8192, 8192, step=1)
st.session_state.top_p = st.sidebar.slider("Top-P", 0.8, 0.99, 0.85, step=0.01)
st.session_state.temperature = st.sidebar.slider("Temperature", 0.6, 1.2, 0.85, step=0.01)

# 模型路径映射
MODEL_PATHS = {
    "MiniMind2-R1 (0.1B)": ["../MiniMind2-R1", "MiniMind2-R1"],
    "MiniMind2-Small-R1 (0.02B)": ["../MiniMind2-Small-R1", "MiniMind2-Small-R1"],
    "MiniMind2 (0.1B)": ["../MiniMind2", "MiniMind2"],
    "MiniMind2-MoE (0.15B)": ["../MiniMind2-MoE", "MiniMind2-MoE"],
    "MiniMind2-Small (0.02B)": ["../MiniMind2-Small", "MiniMind2-Small"],
    "MiniMind-V1 (0.1B)": ["../minimind-v1", "MiniMind-V1"],
    "MiniMind-V1-MoE (0.1B)": ["../minimind-v1-moe", "MiniMind-V1-MoE"],
    "MiniMind-V1-Small (0.02B)": ["../minimind-v1-small", "MiniMind-V1-Small"],
}

selected_model = st.sidebar.selectbox('Models', list(MODEL_PATHS.keys()), index=2)  # 默认选择 MiniMind2
model_path = MODEL_PATHS[selected_model][0]

slogan = f"Hi, I'm {MODEL_PATHS[selected_model][1]}"

image_url = "https://www.modelscope.cn/api/v1/studio/gongjy/MiniMind/repo?Revision=master&FilePath=images%2Flogo2.png&View=true"

st.markdown(
    f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    '<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
    f'<img src="{image_url}" style="width: 45px; height: 45px; "> '
    f'<span style="font-size: 26px; margin-left: 10px;">{slogan}</span>'
    '</div>'
    '<span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">内容完全由AI生成，请务必仔细甄别<br>Content AI-generated, please discern with care</span>'
    '</div>',
    unsafe_allow_html=True
)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    model, tokenizer = load_model_tokenizer(model_path)

    # 初始化消息列表
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    # Use session state messages
    messages = st.session_state.messages

    # 在显示历史消息的循环中
    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar=image_url):
                st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
                if st.button("×", key=f"delete_{i}"):
                    # 删除当前消息及其之后的所有消息
                    st.session_state.messages = st.session_state.messages[:i - 1]
                    st.session_state.chat_messages = st.session_state.chat_messages[:i - 1]
                    st.rerun()
        else:
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: gray; border-radius: 10px; color:white; ">{message["content"]}</div></div>',
                unsafe_allow_html=True)

    # 处理新的输入或重新生成
    prompt = st.chat_input(key="input", placeholder="给 MiniMind 发送消息")

    # 检查是否需要重新生成
    if hasattr(st.session_state, 'regenerate') and st.session_state.regenerate:
        prompt = st.session_state.last_user_message
        regenerate_index = st.session_state.regenerate_index  # 获取重新生成的位置
        # 清除所有重新生成相关的状态
        delattr(st.session_state, 'regenerate')
        delattr(st.session_state, 'last_user_message')
        delattr(st.session_state, 'regenerate_index')

    if prompt:
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: gray; border-radius: 10px; color:white; ">{prompt}</div></div>',
            unsafe_allow_html=True)
        messages.append({"role": "user", "content": prompt})
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant", avatar=image_url):
            placeholder = st.empty()
            random_seed = random.randint(0, 2 ** 32 - 1)
            setup_seed(random_seed)

            st.session_state.chat_messages = system_prompt + st.session_state.chat_messages[
                                                             -(st.session_state.history_chat_num + 1):]
            new_prompt = tokenizer.apply_chat_template(
                st.session_state.chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )[-(st.session_state.max_new_tokens - 1):]

            x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=device).unsqueeze(0)
            with torch.no_grad():
                res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=st.session_state.max_new_tokens,
                                       temperature=st.session_state.temperature,
                                       top_p=st.session_state.top_p, stream=True)
                try:
                    for y in res_y:
                        answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                        if (answer and answer[-1] == '�') or not answer:
                            continue
                        placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)
                except StopIteration:
                    print("No answer")

                assistant_answer = answer.replace(new_prompt, "")
                messages.append({"role": "assistant", "content": assistant_answer})
                st.session_state.chat_messages.append({"role": "assistant", "content": assistant_answer})

                with st.empty():
                    if st.button("×", key=f"delete_{len(messages) - 1}"):
                        st.session_state.messages = st.session_state.messages[:-2]
                        st.session_state.chat_messages = st.session_state.chat_messages[:-2]
                        st.rerun()


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    main()
