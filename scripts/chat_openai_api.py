# 导入OpenAI API客户端
from openai import OpenAI

# 初始化OpenAI客户端
# 使用本地部署的模型服务，通过OpenAI兼容的API接口访问
client = OpenAI(
    api_key="ollama",  # API密钥，这里使用ollama作为占位符
    base_url="http://127.0.0.1:8998/v1"  # 本地模型服务的API地址
)
# 是否使用流式输出，实现打字机效果
stream = True
# 初始化对话历史
conversation_history_origin = []
conversation_history = conversation_history_origin.copy()
# 设置历史对话轮数，必须为偶数（一轮包含一问一答）
# 为0则不携带历史对话，每次都是独立的问答
history_messages_num = 2
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
