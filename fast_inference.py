import json
import random
import numpy as np
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

st.set_page_config(page_title="MiniMind-V1")
st.title("MiniMind-V1")

model_id = "./minimind-v1"


@st.cache_resource
def load_model_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,
        trust_remote_code=True
    )
    model = model.eval()
    generation_config = GenerationConfig.from_pretrained(model_id)
    return model, tokenizer, generation_config


def clear_chat_messages():
    del st.session_state.messages
    del st.session_state.chat_messages


def init_chat_messages():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("我是由JingyaoGong创造的MiniMind，很高兴为您服务😄  \n"
                    "注：所有AI生成内容的准确性和立场无法保证，不代表我们的态度或观点。")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = "🫡" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    return st.session_state.messages


st.sidebar.title("设定调整")
st.session_state.history_chat_num = st.sidebar.slider("携带历史对话条数", 0, 6, 0, step=2)
st.session_state.max_new_tokens = st.sidebar.slider("最大输入/生成长度", 256, 768, 512, step=1)
st.session_state.top_k = st.sidebar.slider("top_k", 0, 16, 14, step=1)
st.session_state.temperature = st.sidebar.slider("temperature", 0.3, 1.3, 0.5, step=0.01)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    model, tokenizer, generation_config = load_model_tokenizer()
    messages = init_chat_messages()

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
            messages.append({"role": "user", "content": prompt})
            st.session_state.chat_messages.append({"role": "user", "content": '请问，' + prompt + '？'})
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            # Generate a random seed
            random_seed = random.randint(0, 2 ** 32 - 1)
            setup_seed(random_seed)

            new_prompt = tokenizer.apply_chat_template(
                st.session_state.chat_messages[-(st.session_state.history_chat_num + 1):],
                tokenize=False,
                add_generation_prompt=True
            )[-(st.session_state.max_new_tokens - 1):]

            x = tokenizer(new_prompt).data['input_ids']
            x = (torch.tensor(x, dtype=torch.long)[None, ...])
            with torch.no_grad():
                res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=st.session_state.max_new_tokens,
                                       temperature=st.session_state.temperature,
                                       top_k=st.session_state.top_k, stream=True)
                try:
                    y = next(res_y)
                except StopIteration:
                    return

                while y != None:
                    answer = tokenizer.decode(y[0].tolist())
                    if answer and answer[-1] == '�':
                        try:
                            y = next(res_y)
                        except:
                            break
                        continue
                    if not len(answer):
                        try:
                            y = next(res_y)
                        except:
                            break
                        continue
                    placeholder.markdown(answer)
                    try:
                        y = next(res_y)
                    except:
                        break

            assistant_answer = answer.replace(new_prompt, "")
            messages.append({"role": "assistant", "content": assistant_answer})
            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_answer})

    st.button("清空对话", on_click=clear_chat_messages)


if __name__ == "__main__":
    main()
