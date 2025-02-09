from openai import OpenAI

client = OpenAI(
    api_key="none",
    base_url="http://localhost:8998/v1"
)
stream = True
conversation_history_origin = []
conversation_history = conversation_history_origin.copy()
while True:
    conversation_history = conversation_history_origin.copy()
    query = input('[Q]: ')
    conversation_history.append({"role": "user", "content": query})
    response = client.chat.completions.create(
        model="minimind",
        messages=conversation_history,
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
