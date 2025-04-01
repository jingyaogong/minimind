from flask import Flask, request, Response, render_template_string
import json
from openai import OpenAI
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 初始化OpenAI客户端
client = OpenAI(
    api_key="none",
    base_url="http://localhost:8998/v1"
)

# 对话配置
MODEL_NAME = "model-identifier"
SYSTEM_PROMPT = "你是一个有帮助的AI助手，请用中文回答"
MAX_HISTORY = 6  # 保留最近3轮对话

class ChatManager:
    def __init__(self):
        self.conversations = {}  # 存储对话历史的字典

    def get_conversation(self, session_id):
        """获取或初始化对话历史"""
        if session_id not in self.conversations:
            self.conversations[session_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
        return self.conversations[session_id]

    def update_conversation(self, session_id, user_input, assistant_response):
        """更新对话历史"""
        conv = self.conversations[session_id]
        conv.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_response}
        ])
        # 保留最近N轮对话（考虑系统提示）
        self.conversations[session_id] = conv[-MAX_HISTORY*2:]

chat_manager = ChatManager()

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Chat</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .chat-container { height: 500px; border: 1px solid #ddd; padding: 20px; overflow-y: auto; margin-bottom: 20px; }
            .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
            .user-message { background-color: #e3f2fd; }
            .ai-message { background-color: #f5f5f5; }
            .input-area { display: flex; gap: 10px; }
            input { flex: 1; padding: 10px; }
            button { padding: 10px 20px; background-color: #007bff; color: white; border: none; cursor: pointer; }
            button:disabled { background-color: #6c757d; }
        </style>
    </head>
    <body>
        <div class="chat-container" id="chatBox"></div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="输入你的问题...">
            <button onclick="sendMessage()" id="sendBtn">发送</button>
        </div>
        
        <script>
            const chatBox = document.getElementById('chatBox');
            const userInput = document.getElementById('userInput');
            const sendBtn = document.getElementById('sendBtn');
            
            function appendMessage(role, content) {
                const div = document.createElement('div');
                div.className = `message ${role}-message`;
                div.innerHTML = `<strong>${role === 'user' ? '你' : 'AI'}:</strong> ${content}`;
                chatBox.appendChild(div);
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            async function sendMessage() {
                const message = userInput.value.trim();
                if (!message) return;
                
                // 禁用输入
                userInput.value = '';
                userInput.disabled = true;
                sendBtn.disabled = true;
                
                // 显示用户消息
                appendMessage('user', message);
                
                // 创建AI消息占位符
                const aiDiv = document.createElement('div');
                aiDiv.className = 'message ai-message';
                aiDiv.innerHTML = '<strong>AI:</strong> ';
                chatBox.appendChild(aiDiv);
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ message: message })
                    });

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let aiResponse = '';

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        const chunk = decoder.decode(value);
                        try {
                            const data = JSON.parse(chunk);
                            aiResponse += data.content;
                            aiDiv.innerHTML = `<strong>AI:</strong> ${aiResponse}`;
                        } catch (e) {
                            console.error('解析错误:', e);
                        }
                    }
                } catch (error) {
                    aiDiv.innerHTML += ' [连接错误]';
                    console.error('Error:', error);
                } finally {
                    userInput.disabled = false;
                    sendBtn.disabled = false;
                    userInput.focus();
                }
            }

            // 支持回车发送
            userInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendMessage();
            });
        </script>
    </body>
    </html>
    ''')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = 'default'  # 实际应用中应使用真实会话ID
        
        if not user_message:
            return json.dumps({"error": "空输入"}), 400

        # 获取对话历史
        conversation = chat_manager.get_conversation(session_id)
        conversation.append({"role": "user", "content": user_message})

        # 创建流式响应
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=conversation,
            temperature=0.7,
            max_tokens=512,
            stream=True
        )

        def generate():
            full_response = ''
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                full_response += content
                yield json.dumps({"content": content}) + "\n"
            
            # 更新对话历史
            chat_manager.update_conversation(session_id, user_message, full_response)
            logger.info(f"会话 {session_id} 更新，历史长度: {len(conversation)}")

        return Response(generate(), content_type='text/event-stream')

    except Exception as e:
        logger.error(f"请求失败: {str(e)}")
        return json.dumps({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
