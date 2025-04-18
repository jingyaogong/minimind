from openai import OpenAI

client = OpenAI(
    api_key="none",
    base_url="http://localhost:8998/v1"
)
stream = True
conversation_history_origin = []
conversation_history = conversation_history_origin.copy()
history_messages_num = 2  # 设置为偶数（Q+A），为0则每次不携带历史对话进行独立QA
while True:
    query = input('[Q]: ')
    conversation_history.append({"role": "user", "content": query})
    response = client.chat.completions.create(
        model="minimind",
        messages=conversation_history[-history_messages_num:],
        stream=stream
    )
    if not stream:
        assistant_res = response.choices[0].message.content
        print('[A]: ', assistant_res)
    else:
        print('[A]: ', end='')
        assistant_res = ''
        for chunk in response:
            print(chunk.choices[0].delta.content or "", end="")
            assistant_res += chunk.choices[0].delta.content or ""

    conversation_history.append({"role": "assistant", "content": assistant_res})
    print('\n\n')
