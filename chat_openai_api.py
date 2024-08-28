from openai import OpenAI

client = OpenAI(
    api_key="none",
    base_url="http://202.195.167.142:8000/v1"
)

# 初始化对话历史列表
conversation_history_origin = []
conversation_history = conversation_history_origin.copy()
while True:
    conversation_history = conversation_history_origin.copy()
    query = input('Enter Your Q: ')

    # 将用户的问题添加到对话历史中
    conversation_history.append({"role": "user", "content": query})

    # Chat completion API
    stream = client.chat.completions.create(
        model="minimind",
        messages=conversation_history,  # 传递整个对话历史
        stream=True
    )

    print('minimind: ', end='')
    assistant_res = ''
    for chunk in stream:
        # 将生成的回复实时打印出来
        print(chunk.choices[0].delta.content or "", end="")
        assistant_res += chunk.choices[0].delta.content or ""

    # 当完成生成回复后，将LLM的回答也添加到对话历史中
    conversation_history.append({"role": "assistant", "content": assistant_res})
    print()

# # Example: reuse your existing OpenAI setup
# from openai import OpenAI
#
# # Point to the local server
# client = OpenAI(base_url="http://202.195.167.206:8000/v1", api_key="none")
#
# completion = client.chat.completions.create(
#     model="minimind",
#     messages=[{"role": "user", "content": "世界上最高的山是？"}],
#     stream=False
# )
#
# print(completion.choices[0].message)
