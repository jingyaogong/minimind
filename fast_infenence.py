import json
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

st.set_page_config(page_title="MiniMind-V1 Demo(无历史上文)")
st.title("MiniMind-V1 Demo(无历史上文)")

model_id = "minimind-v1"

# -----------------------------------------------------------------------------
temperature = 0.7
top_k = 8
max_seq_len = 1 * 1024
# -----------------------------------------------------------------------------


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


def init_chat_messages():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是由Joya开发的MiniMind，很高兴为您服务😄")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


# max_new_tokens = st.sidebar.slider("max_new_tokens", 0, 1024, 512, step=1)
# top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
# top_k = st.sidebar.slider("top_k", 0, 100, 0, step=1)
# temperature = st.sidebar.slider("temperature", 0.0, 2.0, 1.0, step=0.01)
# do_sample = st.sidebar.checkbox("do_sample", value=False)


def main():
    model, tokenizer, generation_config = load_model_tokenizer()
    messages = init_chat_messages()

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
            messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()

            chat_messages = []
            chat_messages.append({"role": "user", "content": prompt})
            # print(messages)
            new_prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )[-(max_seq_len - 1):]

            x = tokenizer(new_prompt).data['input_ids']
            x = (torch.tensor(x, dtype=torch.long)[None, ...])

            response = ''

            with torch.no_grad():
                res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=max_seq_len, temperature=temperature,
                                       top_k=top_k, stream=True)
                try:
                    y = next(res_y)
                except StopIteration:
                    return

                history_idx = 0
                while y != None:
                    answer = tokenizer.decode(y[0].tolist())
                    if answer and answer[-1] == '�':
                        try:
                            y = next(res_y)
                        except:
                            break
                        continue
                    # print(answer)
                    if not len(answer):
                        try:
                            y = next(res_y)
                        except:
                            break
                        continue
                    placeholder.markdown(answer)
                    response = answer
                    try:
                        y = next(res_y)
                    except:
                        break

            # if contain_history_chat:
            #     assistant_answer = answer.replace(new_prompt, "")
            #     messages.append({"role": "assistant", "content": assistant_answer})

        messages.append({"role": "assistant", "content": response})
        # print("messages: ", json.dumps(response, ensure_ascii=False), flush=True)

    st.button("清空对话", on_click=clear_chat_messages)


if __name__ == "__main__":
    main()
