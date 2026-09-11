# 第 27 课：模型转换、OpenAI API 与 WebUI

这一课讲 MiniMind 的部署侧链路：训练脚本产出的原生 PyTorch 权重如何变成 transformers 生态可加载的模型目录，CLI/API/WebUI 又如何复用同一套 tokenizer、chat template 和 `generate` 约定。

本节不讲生产级服务治理。这里的目标是看懂 MiniMind 项目中已有的三个部署入口：模型格式转换、OpenAI 兼容 API、Streamlit WebUI。

## 目录

- [0. 本节主线](#l27-mainline)
- [1. 本节要懂的 6 个原理](#l27-principles)
- [2. 完整原理：部署侧把训练产物包装成可调用接口](#l27-complete-principle)
- [3. 源码阅读顺序图](#l27-reading-order)
- [4. MiniMind 源码走读](#l27-source-walkthrough)
- [5. 本节必须会写 / 暂时不要求](#l27-must-write)
- [6. 实验验证](#l27-experiment)
- [7. 阶段组装](#l27-stage-assembly)
- [8. 本节检查](#l27-check)
- [9. 下一课](#l27-next)

<a id="l27-mainline"></a>
## 0. 本节主线

MiniMind 的部署链路是：

```text
训练得到 out/full_sft_768.pth
-> convert_model.py 把原生 torch state_dict 转成 transformers 模型目录
-> transformers 目录保存 config、tokenizer、chat template、model weights
-> CLI / API / WebUI 用 AutoTokenizer + AutoModelForCausalLM 加载
-> messages 通过 apply_chat_template 变成 prompt 文本
-> tokenizer(prompt) 得到 input_ids / attention_mask
-> model.generate 生成新 token
-> tokenizer.decode 得到 assistant 文本
-> API/WebUI 再把 <think>、<tool_call> 文本协议转成 reasoning_content / tool_calls / 可视化结果
```

一句话：

```text
训练代码负责学会生成协议；
部署代码负责加载权重、套模板、调用 generate、再把文本协议包装成用户能调用的接口。
```

这一课最容易混的三个边界：

```text
模型格式：
  原生 torch 权重是 state_dict；transformers 目录是 config + tokenizer + 权重文件。

聊天协议：
  messages/tools/open_thinking 不是直接喂给模型，而是先由 chat template 渲染成文本 prompt。

接口协议：
  MiniMind 内部生成的是 <think> / <tool_call> 文本；OpenAI API 返回的是 reasoning_content / tool_calls 字段。
```

<a id="l27-principles"></a>
## 1. 本节要懂的 6 个原理

| 原理 | 要理解什么 | 源码位置 |
|---|---|---|
| 转换的本质是保存结构合同和权重 | transformers 目录必须同时有 config、tokenizer、chat template 和模型权重 | `scripts/convert_model.py:16-36`, `scripts/convert_model.py:40-96`, `minimind-3/config.json:1-38` |
| MiniMind 有两种加载路径 | `load_from` 指向 `model` 时走原生 torch 权重；指向模型目录时走 transformers 加载 | `eval_llm.py:12-30`, `scripts/serve_openai_api.py:28-47` |
| API 服务只是把 OpenAI 请求转成本地 generate | request 的 `messages/tools/open_thinking` 先进入 chat template，再生成 token | `scripts/serve_openai_api.py:50-68`, `scripts/serve_openai_api.py:171-225` |
| 部署侧要解析文本协议 | `<think>` 和 `<tool_call>` 是模型生成文本，服务端要转换成 API 字段 | `scripts/serve_openai_api.py:83-102`, `minimind-3/chat_template.jinja:1-11`, `minimind-3/chat_template.jinja:78-84` |
| WebUI 是本地多轮循环 | WebUI 维护 session messages、加载模型、流式生成，并可执行工具后继续生成 | `scripts/web_demo.py:198-209`, `scripts/web_demo.py:239-248`, `scripts/web_demo.py:350-413` |
| 客户端只认 OpenAI 兼容接口 | `chat_api.py` 不关心模型内部结构，只向 base_url 发 chat completions 请求 | `scripts/chat_api.py:3-22`, `scripts/chat_api.py:27-39` |

学完本节，你应该能解释这些对象：

```text
torch state_dict:
  trainer 保存的原生 PyTorch 参数字典，通常位于 out/*.pth。

transformers model directory:
  包含 config.json、tokenizer.json、tokenizer_config.json、model.safetensors 等文件的目录。

chat template:
  把 OpenAI 风格 messages/tools 渲染成 MiniMind 实际训练过的文本协议。

OpenAI-compatible API:
  对外暴露 /v1/chat/completions，但内部仍然调用本地 tokenizer 和 model.generate。

reasoning_content:
  从模型生成的 <think>...</think> 中解析出来的思考字段。

tool_calls:
  从模型生成的 <tool_call>...</tool_call> JSON 中解析出来的 OpenAI 风格工具调用字段。
```

<a id="l27-complete-principle"></a>
## 2. 完整原理：部署侧把训练产物包装成可调用接口

### 2.1 训练产物不能直接等同于可部署模型目录

训练脚本通常保存的是：

```text
out/full_sft_768.pth
```

这个文件主要是 PyTorch `state_dict`。它知道每个参数张量的名字和值，但它本身不完整表达：

```text
模型结构配置
tokenizer 文件
chat template
special tokens
第三方推理框架需要的 config 字段
```

transformers 目录则是一个更完整的部署包。当前本地 `minimind-3/` 里至少包含：

```text
config.json
tokenizer.json
tokenizer_config.json
chat_template.jinja
model.safetensors
```

所以 `convert_model.py` 做的不是“训练”，而是把训练后的参数放进一个生态更容易识别的目录结构。

### 2.2 部署入口复用训练时的文本协议

从第 4、5、26 课已经看到，MiniMind 不是直接训练裸文本问答，而是训练了带角色、思考、工具的模板协议：

```text
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
<think>
...
</think>

<tool_call>
{"name": "...", "arguments": {...}}
</tool_call>
```

部署时不能绕过这个协议。CLI、API、WebUI 都会做同一件事：

```text
messages -> tokenizer.apply_chat_template(...) -> prompt -> tokenizer(prompt) -> generate
```

区别只是外壳不同：

```text
CLI:
  从终端读 input，终端打印 response。

API:
  从 HTTP request 读 messages，返回 OpenAI 兼容 JSON / SSE。

WebUI:
  从浏览器输入框读 prompt，用 Streamlit session 保存历史并渲染结果。
```

### 2.3 OpenAI 兼容不等于模型变成 OpenAI 模型

`serve_openai_api.py` 暴露的是 OpenAI 风格接口：

```text
POST /v1/chat/completions
```

请求格式像这样：

```json
{
  "model": "minimind",
  "messages": [{"role": "user", "content": "你好"}],
  "stream": true,
  "tools": [],
  "chat_template_kwargs": {"open_thinking": true}
}
```

但内部仍然是本地 MiniMind：

```text
AutoTokenizer.from_pretrained(...)
AutoModelForCausalLM.from_pretrained(...)
tokenizer.apply_chat_template(...)
model.generate(...)
```

也就是说，OpenAI 兼容只是接口层协议兼容，不是训练目标或模型结构发生了变化。

<a id="l27-reading-order"></a>
## 3. 源码阅读顺序图

按这个顺序读源码：

```text
1. minimind-3/config.json:1-38
   先看 transformers 目录里的结构合同。

2. scripts/convert_model.py:16-36, 40-96
   看原生 torch 权重如何保存成 transformers 目录。

3. eval_llm.py:12-30, 71-88
   用最简单的 CLI 入口理解 load -> template -> generate -> decode。

4. scripts/serve_openai_api.py:50-68
   看 API request 支持哪些字段。

5. scripts/serve_openai_api.py:83-102
   看服务端如何解析 <think> 和 <tool_call>。

6. scripts/serve_openai_api.py:171-225
   看非流式 API 的完整调用链。

7. scripts/serve_openai_api.py:105-165
   再看流式响应如何拆 reasoning/content/tool_calls。

8. scripts/web_demo.py:239-248, 350-413
   最后看 WebUI 如何选模型、维护历史、处理工具循环。

9. scripts/chat_api.py:3-39
   用客户端视角确认 OpenAI 兼容接口的意义。
```

<a id="l27-source-walkthrough"></a>
## 4. MiniMind 源码走读

### 4.1 transformers 目录保存结构合同

File: `minimind-3/config.json:1-38`

Read this to understand: transformers 模型目录必须告诉加载器使用什么模型结构、多少层、多少 head、词表多大。

Code/config/template excerpt:

```json
{
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "hidden_size": 768,
  "model_type": "qwen3",
  "num_attention_heads": 8,
  "num_hidden_layers": 8,
  "num_key_value_heads": 4,
  "vocab_size": 6400
}
```

This code shows:

- 当前下载好的 `minimind-3` 对齐的是 `Qwen3ForCausalLM` 生态接口。
- 模型结构参数和第 8-12 课学过的 hidden size、layer、head、vocab 是同一类合同。
- 部署加载时不只需要权重，还需要这些配置字段。

### 4.2 转换脚本有两条路线

File: `scripts/convert_model.py:16-36`

Read this to understand: MiniMind 自定义结构也可以注册成 transformers 可加载模型。

Code/config/template excerpt:

```python
def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.float16):
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    lm_model = MiniMindForCausalLM(lm_config)
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
```

This code shows:

- `torch_path` 是训练保存的原生权重。
- `save_pretrained` 写出 transformers 目录。
- tokenizer 会一起保存，否则部署端无法复原 chat template 和 token id。

File: `scripts/convert_model.py:40-96`

Read this to understand: 项目主线也支持转换成 Qwen3 兼容结构，方便 vLLM、Ollama 等生态加载。

Code/config/template excerpt:

```python
common_config = {
    "vocab_size": lm_config.vocab_size,
    "hidden_size": lm_config.hidden_size,
    "intermediate_size": lm_config.intermediate_size,
    "num_hidden_layers": lm_config.num_hidden_layers,
    "num_attention_heads": lm_config.num_attention_heads,
    "num_key_value_heads": lm_config.num_key_value_heads,
    "rope_theta": lm_config.rope_theta,
}
qwen_config = Qwen3Config(**common_config, use_sliding_window=False, sliding_window=None)
qwen_model = Qwen3ForCausalLM(qwen_config)
qwen_model.load_state_dict(state_dict, strict=True)
qwen_model.save_pretrained(transformers_path)
tokenizer.save_pretrained(transformers_path)
```

This code shows:

- 转换时把 MiniMind config 映射到 Qwen3 config。
- `strict=True` 要求 state_dict 名称和目标结构完全对齐。
- 转换后的目录能被更多 transformers 生态工具识别。

### 4.3 CLI 与 API 共享两种加载路径

File: `eval_llm.py:12-30`

Read this to understand: MiniMind 推理入口如何区分原生 torch 权重和 transformers 目录。

Code/config/template excerpt:

```python
tokenizer = AutoTokenizer.from_pretrained(args.load_from)
if 'model' in args.load_from:
    model = MiniMindForCausalLM(MiniMindConfig(...))
    ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
    model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
else:
    model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
return model.half().eval().to(args.device), tokenizer
```

This code shows:

- `--load_from model` 走项目自定义 `MiniMindForCausalLM`。
- `--load_from ./minimind-3` 走 transformers `AutoModelForCausalLM`。
- 两条路径最后都返回 `model, tokenizer`，后面的推理流程相同。

注意：这里用的是字符串判断 `if 'model' in args.load_from`。学习时够用；做更稳的工程时应改成显式参数或检查目录结构，避免路径名误判。

File: `scripts/serve_openai_api.py:28-47`

Read this to understand: API 服务端复用了同一套加载边界。

Code/config/template excerpt:

```python
tokenizer = AutoTokenizer.from_pretrained(args.load_from)
if 'model' in args.load_from:
    model = MiniMindForCausalLM(MiniMindConfig(...))
    model.load_state_dict(torch.load(ckp, map_location=device), strict=True)
else:
    model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
return model.half().eval().to(device), tokenizer
```

This code shows:

- API 服务不是另一套模型，它仍然先加载本地模型和 tokenizer。
- `load_from` 决定加载原生权重还是 transformers 目录。
- `model.eval()` 表示部署侧不训练，只推理。

### 4.4 CLI 推理是最小部署闭环

File: `eval_llm.py:71-88`

Read this to understand: 最小推理闭环就是 messages、chat template、tokenizer、generate、decode。

Code/config/template excerpt:

```python
conversation.append({"role": "user", "content": prompt})
inputs = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True,
    open_thinking=bool(args.open_thinking)
)
inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)
generated_ids = model.generate(
    inputs=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=args.max_new_tokens,
    do_sample=True,
)
response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
```

This code shows:

- `conversation` 是 OpenAI 风格消息列表。
- `apply_chat_template` 把消息列表变成模型实际看到的 prompt。
- decode 时只取 prompt 后面新生成的 token。

### 4.5 chat template 固化工具和思考协议

File: `minimind-3/chat_template.jinja:1-11`

Read this to understand: 有 tools 时，模板会把工具 schema 写进 system prompt，并要求模型输出 `<tool_call>` JSON。

Code/config/template excerpt:

```jinja
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:" }}
```

This code shows:

- 工具不是 Python 函数直接进模型，而是 schema 文本进入 prompt。
- 模型学到的是输出 `<tool_call>` 这类文本动作。
- 这和第 26 课的 Agent rollout 使用的是同一套协议。

File: `minimind-3/chat_template.jinja:78-84`

Read this to understand: `open_thinking` 控制 assistant 起始处的 `<think>` 形态。

Code/config/template excerpt:

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if open_thinking is defined and open_thinking is true %}
        {{- '<think>\n' }}
    {%- else %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
```

This code shows:

- 开启 thinking 时，模型从未闭合的 `<think>` 后继续生成。
- 不开启 thinking 时，模板给一个空 thinking 块，让模型直接回答。
- 部署侧的 `open_thinking` 本质是 chat template 参数，不是模型结构开关。

### 4.6 API 请求模型定义

File: `scripts/serve_openai_api.py:50-68`

Read this to understand: OpenAI 兼容接口接收哪些字段，以及 thinking 如何兼容不同客户端传法。

Code/config/template excerpt:

```python
class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    stream: bool = True
    tools: list = []
    open_thinking: bool = False
    chat_template_kwargs: dict = None

    def get_open_thinking(self) -> bool:
        if self.open_thinking:
            return True
        if self.chat_template_kwargs:
            return self.chat_template_kwargs.get('open_thinking', False) or \
                   self.chat_template_kwargs.get('enable_thinking', False)
        return False
```

This code shows:

- API 请求保留 OpenAI 风格的 `messages` 和 `tools`。
- `open_thinking` 可以直接传，也可以通过 `chat_template_kwargs` 传。
- 服务端最终只需要知道是否把 `open_thinking=True` 传给 template。

### 4.7 服务端解析模型输出协议

File: `scripts/serve_openai_api.py:83-102`

Read this to understand: 模型输出是文本，服务端要把文本协议转换成 OpenAI 风格字段。

Code/config/template excerpt:

```python
def parse_response(text):
    reasoning_content = None
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    tool_calls = []
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        call = json.loads(m.strip())
        tool_calls.append({
            "id": f"call_{int(time.time())}_{i}",
            "type": "function",
            "function": {"name": call.get("name", ""), "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)}
        })
    return text.strip(), reasoning_content, tool_calls or None
```

This code shows:

- `reasoning_content` 来自 `<think>...</think>`。
- `tool_calls` 来自 `<tool_call>...</tool_call>` 中的 JSON。
- `content` 会移除 thinking 和 tool_call，只留下普通回答文本。

### 4.8 非流式 API 完整路径

File: `scripts/serve_openai_api.py:187-225`

Read this to understand: 非流式接口如何从 request 到 response。

Code/config/template excerpt:

```python
new_prompt = tokenizer.apply_chat_template(
    request.messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=request.tools or None,
    open_thinking=request.get_open_thinking()
)
inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
generated_ids = model.generate(
    inputs["input_ids"],
    max_length=inputs["input_ids"].shape[1] + request.max_tokens,
    attention_mask=inputs["attention_mask"],
    top_p=request.top_p,
    temperature=request.temperature
)
answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
content, reasoning_content, tool_calls = parse_response(answer)
message = {"role": "assistant", "content": content}
```

This code shows:

- API 和 CLI 一样先套 chat template。
- `answer` 只 decode 新生成部分。
- 返回前调用 `parse_response`，把 MiniMind 文本协议转成 API 字段。

### 4.9 流式 API 额外处理 reasoning/content/tool_calls

File: `scripts/serve_openai_api.py:105-165`

Read this to understand: 流式响应为什么要在生成过程中区分 reasoning 和 content。

Code/config/template excerpt:

```python
queue = Queue()
streamer = CustomStreamer(tokenizer, queue)
Thread(target=_generate).start()

full_text = ""
thinking_ended = not bool(open_thinking)
while True:
    text = queue.get()
    if text is None:
        break
    full_text += text
    if not thinking_ended:
        pos = full_text.find('</think>')
        ...
        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_r}}]}, ensure_ascii=False)
    else:
        yield json.dumps({"choices": [{"delta": {"content": new_c}}]}, ensure_ascii=False)

_, _, tool_calls = parse_response(full_text)
if tool_calls:
    yield json.dumps({"choices": [{"delta": {"tool_calls": tool_calls}}]}, ensure_ascii=False)
```

This code shows:

- `TextStreamer` 把生成结果持续放进 queue。
- `</think>` 出现前的增量作为 `reasoning_content`。
- 工具调用要等完整文本生成后再解析，因为 JSON 需要闭合标签。

### 4.10 WebUI 加载模型并维护多轮上下文

File: `scripts/web_demo.py:198-209`

Read this to understand: WebUI 加载的是 transformers 模型目录，而不是训练脚本对象。

Code/config/template excerpt:

```python
@st.cache_resource
def load_model_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = model.half().eval().to(device)
    return model, tokenizer
```

This code shows:

- WebUI 默认服务的是 transformers 格式模型目录。
- `st.cache_resource` 避免每次交互都重新加载权重。
- WebUI 和 API 一样只做推理。

File: `scripts/web_demo.py:239-248`

Read this to understand: WebUI 为什么要求把模型目录放到 `scripts/` 下面。

Code/config/template excerpt:

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {}
for d in sorted(os.listdir(script_dir), reverse=True):
    full_path = os.path.join(script_dir, d)
    if os.path.isdir(full_path) and not d.startswith('.') and not d.startswith('_'):
        if any(f.endswith(('.bin', '.safetensors', '.pt')) or os.path.exists(os.path.join(full_path, 'model.safetensors.index.json')) for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))):
            MODEL_PATHS[d] = [d, d]
if not MODEL_PATHS:
    MODEL_PATHS = {"No models found": ["", "No models"]}
```

This code shows:

- WebUI 扫描的是 `scripts/` 目录下的子目录。
- 子目录里要有 `.safetensors`、`.bin` 或 `.pt` 这类权重文件。
- 所以 README 才要求把 `minimind-3` 复制到 `scripts/minimind-3`。

File: `scripts/web_demo.py:350-413`

Read this to understand: WebUI 如何把一次用户输入变成多轮工具循环。

Code/config/template excerpt:

```python
tools = [t for t in TOOLS if t['function']['name'] in st.session_state.get('selected_tools', [])] or None
st.session_state.chat_messages = sys_prompt + st.session_state.chat_messages[-(st.session_state.history_chat_num + 1):]
template_kwargs = {"tokenize": False, "add_generation_prompt": True}
if st.session_state.get('enable_thinking', False):
    template_kwargs["open_thinking"] = True
if tools:
    template_kwargs["tools"] = tools
new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)
inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
Thread(target=model.generate, kwargs=generation_kwargs).start()
...
for _ in range(16):
    tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', answer, re.DOTALL)
    if not tool_calls:
        break
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    ...
    st.session_state.chat_messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
    new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)
```

This code shows:

- WebUI 里的工具执行和第 26 课 `rollout_single` 很像：生成 tool_call，执行工具，把 tool message 拼回上下文，再生成。
- `history_chat_num` 控制携带多少历史消息。
- WebUI 使用 streamer，让页面逐步显示生成结果。

### 4.11 客户端只需要 OpenAI 兼容协议

File: `scripts/chat_api.py:3-22`

Read this to understand: 对客户端来说，本地 MiniMind 服务就是一个 OpenAI-compatible endpoint。

Code/config/template excerpt:

```python
client = OpenAI(
    api_key="sk-123",
    base_url="http://localhost:11434/v1"
)
response = client.chat.completions.create(
    model="minimind-local:latest",
    messages=conversation_history[-(history_messages_num or 1):],
    stream=stream,
    temperature=0.8,
    max_tokens=2048,
    top_p=0.8,
    extra_body={"chat_template_kwargs": {"open_thinking": True}, "reasoning_effort": "medium"}
)
```

This code shows:

- 客户端通过 `base_url` 连接本地或第三方 OpenAI 兼容服务。
- `messages` 仍然是标准聊天消息。
- `open_thinking` 被放进 `extra_body.chat_template_kwargs`。

File: `scripts/chat_api.py:27-39`

Read this to understand: 流式客户端如何接收 `reasoning_content` 和普通 `content`。

Code/config/template excerpt:

```python
for chunk in response:
    delta = chunk.choices[0].delta
    r = getattr(delta, 'reasoning_content', None) or ""
    c = delta.content or ""
    if r:
        print(f'\033[90m{r}\033[0m', end="", flush=True)
    if c:
        print(c, end="", flush=True)
    assistant_res += c
```

This code shows:

- 客户端把 `reasoning_content` 和 `content` 分开显示。
- 对话历史只保存普通回答 `assistant_res`。
- 客户端不需要知道模型内部 logits、mask 或 loss。

<a id="l27-must-write"></a>
## 5. 本节必须会写 / 暂时不要求

本节必须会写：

```text
1. 说明 torch state_dict 和 transformers 模型目录的区别。
2. 画出 CLI/API/WebUI 共同的推理链路：
   messages -> chat_template -> tokenizer -> generate -> decode。
3. 解释 open_thinking 为什么是 template 参数。
4. 解释 <think> 和 <tool_call> 如何变成 reasoning_content 和 tool_calls。
5. 用 OpenAI SDK 写一个最小 chat.completions 请求。
```

本节暂时不要求：

```text
1. 手写生产级 HTTP 服务。
2. 实现并发调度、批量推理、KV cache 复用服务。
3. 实现鉴权、限流、日志、监控。
4. 重新实现 Streamlit WebUI。
5. 调通 vLLM / Ollama / llama.cpp 的所有参数。
```

部署侧属于工程外围。课程里优先复用原项目脚本，把学习重点放在接口边界和数据流上。

<a id="l27-experiment"></a>
## 6. 实验验证

### 实验 A：检查部署侧三个协议面

这个实验不启动服务，也不加载完整模型权重。它检查三件事：

```text
transformers 模型目录是否齐全；
chat template 如何把 tools/open_thinking 渲染进 prompt；
serve_openai_api.parse_response 如何把模型文本拆成 API 字段。
```

命令：

```bash
cd /home/sun/minimind
python course/labs/trace_deploy_surfaces.py
```

记录：

```text
model_type =
architectures =
contains_tools_tag =
contains_tool_call_instruction =
has_open_think_prompt =
reasoning_content =
tool_calls =
```

输出怎么看：

```text
contains_tools_tag=True:
  说明工具 schema 已经进入 prompt。

contains_tool_call_instruction=True:
  说明模板明确要求模型用 <tool_call>...</tool_call> 输出动作。

has_open_think_prompt=True:
  说明 open_thinking=True 时，assistant 起始处留下未闭合的 <think>。

tool_calls 非空:
  说明服务端能把 MiniMind 文本协议转成 OpenAI 风格 tool_calls 字段。
```

### 实验 B：用本地 transformers 模型跑一次真实推理

这个实验验证 `minimind-3/` 目录能被当前环境加载，并能完成一次真实生成。

命令：

```bash
cd /home/sun/minimind
python course/labs/run_minimind3_once.py --device cpu --max_new_tokens 32 --prompt "MiniMind 是什么？"
```

记录：

```text
input_ids.shape =
generated_ids.shape =
new_tokens =
response =
```

输出怎么看：

```text
input_ids.shape:
  prompt 被 chat template 和 tokenizer 转成了模型输入。

generated_ids.shape:
  模型完成了自回归生成。

new_tokens:
  新生成 token 数量，不包含 prompt。
```

### 实验 C：可选启动 OpenAI 兼容服务

当前课程默认不要求启动长驻服务。准备好 transformers 模型目录后，可以手动试：

命令：

```bash
cd /home/sun/minimind/scripts
python serve_openai_api.py --load_from ../minimind-3 --device cpu
```

另开一个终端调用：

```bash
cd /home/sun/minimind/scripts
python chat_api.py
```

注意：

```text
chat_api.py 默认 base_url 是 http://localhost:11434/v1；
serve_openai_api.py 默认端口是 8998。
如果要直接连这个服务，需要把 chat_api.py 里的 base_url 改成 http://localhost:8998/v1，
或者用兼容 OpenAI 的客户端自己指定 base_url。
```

### 实验 D：可选启动 WebUI

WebUI 会扫描 `scripts/` 目录下的模型子目录。当前本地模型在项目根目录 `minimind-3/`，所以需要先复制或软链接到 `scripts/` 下。

命令：

```bash
cd /home/sun/minimind
ln -s ../minimind-3 scripts/minimind-3
cd scripts
streamlit run web_demo.py
```

如果 `streamlit` 未安装，这个实验会失败。当前课程的必须验收仍然是实验 A 和实验 B。

<a id="l27-stage-assembly"></a>
## 7. 阶段组装

到第 27 课为止，完整 MiniMind 链路可以这样收束：

```text
第 1-7 课：
  数据、tokenizer、pretrain/SFT 训练入口。

第 8-13 课：
  MiniMind 模型结构、Attention、RoPE、KV cache、FFN/MoE。

第 14-19 课：
  训练机制、checkpoint、full SFT、LoRA。

第 20-26 课：
  distillation、DPO、PPO、GRPO/CISPO、reward/logprob/KL、Agentic RL。

第 27 课：
  把训练出的模型接到 CLI、OpenAI-compatible API、WebUI。
```

部署阶段不新增手写核心模型模块。它主要复用：

```text
model/model_minimind.py:
  原生 torch 权重加载时使用。

minimind-3/:
  transformers 生态加载时使用。

scripts/convert_model.py:
  在原生权重和 transformers 目录之间转换。

scripts/serve_openai_api.py:
  暴露 OpenAI 兼容 HTTP 接口。

scripts/web_demo.py:
  提供本地浏览器交互界面。

course/labs/trace_deploy_surfaces.py:
  验证部署协议面，不依赖长驻服务。
```

阶段验收命令：

```bash
cd /home/sun/minimind
python course/labs/check_project_readiness.py
python course/labs/trace_deploy_surfaces.py
python course/labs/run_minimind3_once.py --device cpu --max_new_tokens 16 --prompt "你好"
```

Portfolio 记录可以写：

```text
读懂并验证了 MiniMind 从训练权重到部署接口的链路：
torch state_dict -> transformers model directory -> AutoTokenizer/AutoModelForCausalLM -> chat template -> generate -> OpenAI-compatible response。
能够解释 <think> / <tool_call> 文本协议如何在 API/WebUI 侧转换为 reasoning_content / tool_calls。
```

<a id="l27-check"></a>
## 8. 本节检查

1. 原生 `out/full_sft_768.pth` 和 `minimind-3/` transformers 目录有什么区别？
2. `convert_model.py` 为什么要保存 tokenizer，而不只是保存模型权重？
3. `eval_llm.py --load_from model` 和 `eval_llm.py --load_from ./minimind-3` 分别走哪条加载路径？
4. `open_thinking=True` 最终影响的是模型结构、采样参数，还是 chat template？
5. API 服务端为什么要把 `<tool_call>` 文本解析成 `tool_calls` 字段？
6. WebUI 为什么要重新 `apply_chat_template` after tool result？
7. OpenAI-compatible API 里的“compatible”具体兼容的是哪一层？

<a id="l27-next"></a>
## 9. 下一课

第 28 课进入总复盘与小项目。

下一课要把全课程收束成一个可展示的成果：

```text
从 tiny 数据跑通关键实验；
整理 tokenizer / model / train / RL / deploy 的完整调用链；
列出教学版实现和原项目实现的差异；
把 portfolio 中的实验记录、实现说明和简历描述整理成最终版本。
```
