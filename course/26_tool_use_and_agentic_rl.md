# 第 26 课：Tool Use 与 Agentic RL

这一课讲 `trainer/train_agent.py`：MiniMind 如何把工具调用从“会输出一段 `<tool_call>` 文本”推进到“多轮 rollout 中调用工具、观察结果、继续生成，并用延迟 reward 做 GRPO/CISPO 更新”。

本节不覆盖完整 Agent 系统设计。这里的 Agentic RL 是 MiniMind 项目里的狭义版本：固定工具集、mock 工具环境、有限多轮交互、规则/GT/reward model 组合打分。

## 目录

- [0. 本节主线](#l26-mainline)
- [1. 本节要懂的 7 个原理](#l26-principles)
- [2. 完整原理：从 tool call 到 trajectory reward](#l26-complete-principle)
- [3. 源码阅读顺序图](#l26-reading-order)
- [4. MiniMind 源码走读](#l26-source-walkthrough)
- [5. 本节必须会写 / 暂时不要求](#l26-must-write)
- [6. 手写与实验模块](#l26-handwrite)
- [7. 实验验证](#l26-experiment)
- [8. 阶段组装](#l26-stage-assembly)
- [9. 本节检查](#l26-check)
- [10. 下一课](#l26-next)

<a id="l26-mainline"></a>
## 0. 本节主线

MiniMind 的 Tool Use / Agentic RL 链路是：

```text
样本提供 messages + tools + gt
-> chat_template 把 tools 渲染到 system prompt
-> policy 生成 assistant response
-> 从 response 中解析 <tool_call>{json}</tool_call>
-> 执行 mock tool，得到 tool observation
-> 把 tool observation 作为 tool message 拼回 messages
-> 继续下一轮生成，直到没有 tool_call 或达到 max_turns
-> 对整条 trajectory 计算 reward
-> 同 prompt 多条 trajectory 做 group-relative advantage
-> 用 GRPO/CISPO + reference KL 更新 policy
-> eval/API 把模型文本中的 <tool_call> 转成 OpenAI 风格 tool_calls
```

一句话：

```text
Tool calling 是模型生成结构化动作；
Agentic RL 是把动作执行后的观察接回上下文，再用整条轨迹的结果训练模型。
```

本节最容易混的三个对象：

```text
tool schema:
  告诉模型有哪些工具、每个工具需要哪些参数。

tool_call:
  模型生成的动作，格式是 <tool_call>{"name": "...", "arguments": {...}}</tool_call>。

tool_response:
  环境执行动作后的观察，格式是 <tool_response>{...}</tool_response>。
```

第 25 课的 `reward / logprob / KL` 在这里没有换概念，只是 reward 从“单条回答分数”变成“多轮工具轨迹分数”。

<a id="l26-principles"></a>
## 1. 本节要懂的 7 个原理

| 原理 | 要理解什么 | 源码位置 |
|---|---|---|
| 工具协议先由 tokenizer/template 固化 | `<tool_call>`、`<tool_response>` 是模型要学习和解析的文本协议 | `trainer/train_tokenizer.py:35-38`, `model/tokenizer_config.json:333` |
| Agent RL 数据不是 labels 数据 | `AgentRLDataset` 返回 `messages/tools/gt`，训练时在线 rollout，不直接返回 `input_ids/labels` | `dataset/lm_dataset.py:226-252` |
| rollout 是环境循环 | 每轮生成 response，解析 tool_call，执行工具，把 tool observation 拼回上下文 | `trainer/train_agent.py:98-157` |
| 工具调用只是 JSON 文本动作 | 训练端用正则找 `<tool_call>...</tool_call>`，再 `json.loads` 成动作 | `trainer/train_agent.py:76-95`, `scripts/eval_toolcall.py:70-78` |
| reward 是 trajectory 级信号 | 工具合法性、GT 命中、格式闭合、未完成、多余重复、RM 分数共同决定 reward | `trainer/train_agent.py:188-239` |
| policy update 沿用 GRPO/CISPO | Agent 训练仍然使用 group reward、old/current/ref logprob、reference KL 和 mask | `trainer/train_agent.py:242-333` |
| eval/API 要把文本协议转成 tool_calls 字段 | 本地评测和 OpenAI API 服务端都要解析模型输出中的 tool_call | `scripts/eval_toolcall.py:177-199`, `scripts/serve_openai_api.py:83-102` |

学完本节，你应该能解释这些变量：

```text
messages_batch:
  batch 内每条样本的多轮消息前缀，来自 AgentRLDataset。

tools_batch:
  每条样本可用的工具 schema。

gt_batch:
  最终答案校验目标，用来给工具轨迹打规则 reward。

turn_outputs_batch:
  一条 trajectory 中每轮 assistant 生成的文本。

response_ids:
  policy 生成的 response token，加上后续 tool observation delta。

response_mask:
  1 表示 policy 生成 token 参与 loss，0 表示 tool observation / prompt 部分不参与 loss。

response_old_logps:
  rollout 时生成 token 的 old logprob；tool observation token 对应 0.0。

rewards:
  每条 trajectory 的最终标量 reward，shape 是 `[B * num_generations]`。
```

<a id="l26-complete-principle"></a>
## 2. 完整原理：从 tool call 到 trajectory reward

### 2.1 Tool calling 是结构化动作，不是普通文本答案

没有工具时，模型只要生成自然语言答案：

```text
user -> assistant answer
```

有工具时，模型要先生成一个动作：

```text
assistant -> <tool_call>{"name":"calculate_math","arguments":{"expression":"256 * 37"}}</tool_call>
```

这个动作不会直接结束对话。训练脚本会解析它，调用环境里的 mock 工具，拿到观察：

```text
tool -> {"result": "9472"}
```

然后把观察重新放进上下文，让模型继续生成最终答案：

```text
assistant -> 256 乘以 37 等于 9472。
```

所以 Agentic RL 训练的对象不是单个 response，而是一条 trajectory：

```text
prompt
-> assistant action_1
-> tool observation_1
-> assistant action_2
-> tool observation_2
-> ...
-> assistant final_answer
```

### 2.2 为什么需要 response_mask

在一条 trajectory 里，不是所有 token 都是模型生成的。

```text
policy 生成的 assistant token:
  应该参与 policy loss。

tool observation token:
  是环境返回，不是 policy 动作，不应该参与 policy loss。

prompt / system / user token:
  是条件上下文，也不应该参与 policy loss。
```

因此 `train_agent.py` 在 rollout 中维护：

```text
response_ids:
  训练用的 token 序列后半部分。

response_mask:
  哪些 response_ids 是 policy 生成动作。

response_old_logps:
  policy 生成动作对应 rollout logprob；非 policy token 用 0.0 占位。
```

这就是本节比普通 GRPO 多出来的关键点：同一段 `response_ids` 中混有 policy 动作和环境观察，必须用 mask 区分。

### 2.3 Reward 是延迟的

工具调用本身不一定立刻知道好坏。比如：

```text
<tool_call>{"name":"calculate_math","arguments":{"expression":"256 * 37"}}</tool_call>
```

这个动作可能是合法的，但最终答案仍然可能错。

MiniMind 的 Agent reward 因此按整条轨迹打分：

```text
R(trajectory)
= tool call 对齐分
+ final answer GT 命中分
+ thinking / 长度 / 格式分
+ reward model score
- 标签不闭合惩罚
- 未完成惩罚
- 重复惩罚
```

然后 GRPO/CISPO 把同一个 prompt 的多条 trajectory reward 做组内标准化：

```text
advantage = (reward - group_mean) / (group_std + 1e-4)
```

也就是说，模型不是学习“某个 token 本身是否好”，而是学习：

```text
在这个上下文中，生成这些动作 token 后，整条轨迹是否比同题其他轨迹更好。
```

### 2.4 Agentic RL 和 SFT Tool Calling 的区别

SFT Tool Calling 学的是数据里的参考轨迹：

```text
给定已有 messages/tools/tool_calls/tool_response/final answer
-> 让模型拟合 assistant 部分 token
```

Agentic RL 学的是在线采样轨迹：

```text
给定 messages/tools/gt
-> 让模型自己采样 tool_call / final answer
-> 环境执行工具
-> 根据最终轨迹打 reward
-> 用 RL 更新模型
```

所以 `AgentRLDataset` 不需要提前给出完整 assistant labels。它只提供起始状态和校验目标。

<a id="l26-reading-order"></a>
## 3. 源码阅读顺序图

建议按这个顺序读：

```text
1. trainer/train_tokenizer.py:35-38
   先看 tool/reasoning 标签如何进入 tokenizer 训练。

2. model/tokenizer_config.json:333
   看 chat_template 如何把 tools/tool_calls/tool role 渲染成文本协议。

3. dataset/lm_dataset.py:226-252
   看 AgentRLDataset 返回 messages/tools/gt，而不是 labels。

4. trainer/train_agent.py:40-95
   看工具定义、mock 执行、参数校验、tool_call 解析。

5. trainer/train_agent.py:98-180
   看多轮 rollout 如何执行工具并拼回 observation。

6. trainer/train_agent.py:188-239
   看 trajectory reward 的规则项。

7. trainer/train_agent.py:242-333
   看 reward 如何进入 GRPO/CISPO 更新。

8. scripts/eval_toolcall.py:177-199
   看推理侧如何反复 generate -> tool call -> tool result。

9. scripts/serve_openai_api.py:83-102
   看服务端如何把文本中的 <tool_call> 转成 OpenAI 风格 tool_calls 字段。
```

<a id="l26-source-walkthrough"></a>
## 4. MiniMind 源码走读

### 4.1 工具标签进入 tokenizer/template

#### 源码证据 A：工具标签作为额外 token

文件：`trainer/train_tokenizer.py:35-38`

看它是为了理解：MiniMind 在训练 tokenizer 时显式加入工具调用和思考标签。

代码摘录：

```python
additional_tokens_list = [
    "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>",
    "<think>", "</think>"
]
```

这段代码说明：

- 工具协议不是临时字符串，它被纳入 tokenizer 训练配置。
- `<tool_call>` 和 `<tool_response>` 都是模型需要稳定生成/识别的结构标签。
- `<think>` 与 tool calling 在同一套 chat template 里共存。

#### 源码证据 B：chat template 描述工具协议

文件：`model/tokenizer_config.json:333`

看它是为了理解：当 `tools` 存在时，template 会把工具 schema 放入 system prompt，并告诉模型按 `<tool_call>` 包裹 JSON。

代码摘录：

```jinja
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {{- "# Tools\n\nYou may call one or more functions ..." }}
    {{- "function signatures within <tools></tools> XML tags:\n<tools>" }}
    ...
    {{- "For each function call, return a json object ... within <tool_call></tool_call> XML tags:" }}
{%- endif %}
```

这段代码说明：

- 工具 schema 是 prompt 的一部分，不是模型内部结构。
- 模型输出 tool call 的约束来自 template 提示和 SFT/RL 训练。
- 后续解析器依赖 `<tool_call>...</tool_call>` 标签边界。

### 4.2 数据集只提供起始状态和 GT

#### 源码证据：`AgentRLDataset`

文件：`dataset/lm_dataset.py:226-252`

看它是为了理解：Agent RL 样本不是普通 SFT 的 `input_ids/labels`，而是用于在线 rollout 的环境初始状态。

代码摘录：

```python
class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        ...
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}
```

这段代码说明：

- `messages[:-1]` 会去掉最后一轮，留下模型要开始 rollout 的上下文。
- `tools` 从 system message 里解析出来。
- `gt` 是最终校验目标，用于 reward，不是 labels。
- `__getitem__` 不返回 `input_ids`，因为 tokenization 会在 rollout 时根据多轮上下文动态发生。

README 里的数据格式也说明了这个约定。

文件：`README.md:824-838`

看它是为了理解：SFT/Agent 数据使用 OpenAI 风格消息，`tools` 挂在 system，`tool_calls` 挂在 assistant。

代码摘录：

```json
{"role": "system", "content": "# Tools ...", "tools": "[...]"}
{"role": "user", "content": "帮我算一下 256 乘以 37 等于多少"}
{"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"calculate_math\",\"arguments\":{\"expression\":\"256 * 37\"}}]"}
{"role": "tool", "content": "{\"result\":\"9472\"}"}
{"role": "assistant", "content": "256 乘以 37 等于 9472。"}
```

这段代码说明：

- 监督数据里可以有完整 tool trajectory。
- Agent RL 数据会额外使用 `gt` 作为最终校验目标。
- 当前工作区没有实际 `dataset/agent_rl.jsonl` 文件；训练命令需要你另外准备数据。

### 4.3 工具定义、解析和执行

#### 源码证据 A：固定工具 schema

文件：`trainer/train_agent.py:40-47`

看它是为了理解：训练脚本里的工具集是固定的 function schema 列表。

代码摘录：

```python
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", ...}},
    {"type": "function", "function": {"name": "unit_converter", ...}},
    {"type": "function", "function": {"name": "get_current_weather", ...}},
    {"type": "function", "function": {"name": "get_current_time", ...}},
    {"type": "function", "function": {"name": "get_exchange_rate", ...}},
    {"type": "function", "function": {"name": "translate_text", ...}},
]
```

这段代码说明：

- Agentic RL 训练不是开放世界工具系统。
- 每个工具都有 `name`、`description`、`parameters`、`required`。
- reward 校验会用这些 schema 判断 tool call 是否有效。

#### 源码证据 B：mock 工具执行和参数校验

文件：`trainer/train_agent.py:56-74`

看它是为了理解：工具环境是训练脚本内置的 mock executor，不是外部真实服务。

代码摘录：

```python
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(eval(...))},
    "unit_converter": lambda args: {"result": round(...)},
    "get_current_weather": lambda args: ...,
    "get_current_time": lambda args: ...,
    "get_exchange_rate": lambda args: ...,
    "translate_text": lambda args: ...,
}

CHECK_ARGS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    ...
}
```

这段代码说明：

- `MOCK_RESULTS` 决定动作执行后能拿到什么 observation。
- `CHECK_ARGS` 决定 reward 里的“有效工具调用”。
- 学这节时不要把 mock executor 当成生产级工具沙箱。

#### 源码证据 C：解析 `<tool_call>` 文本

文件：`trainer/train_agent.py:76-95`

看它是为了理解：模型输出先是文本，训练脚本再用正则和 JSON 解析成动作。

代码摘录：

```python
def parse_tool_calls(text):
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try: calls.append(json.loads(m.strip()))
        except: pass
    return calls

def execute_tool(name, args):
    fn = MOCK_RESULTS.get(name)
    if not fn: return None
    try:
        signal.signal(signal.SIGALRM, ...)
        signal.alarm(1)
        return fn(args)
    except:
        return None
    finally:
        try: signal.alarm(0)
        except: pass
```

这段代码说明：

- 标签不闭合或 JSON 非法时，解析不到有效 tool call。
- 未知工具返回 `None`。
- 执行被 1 秒 alarm 限制，避免简单 mock 函数卡死。
- 解析失败被吞掉，reward 侧再通过格式/GT/工具数量扣分。

### 4.4 多轮 rollout：动作、观察、再生成

#### 源码证据：`rollout_single`

文件：`trainer/train_agent.py:98-157`

看它是为了理解：Agent rollout 不只是 `model.generate` 一次，而是最多多轮“生成 -> 执行工具 -> 拼回上下文”。

代码摘录：

```python
for turn in range(max_turns):
    context = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools,
        open_thinking=open_thinking
    )
    inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(device)
    ...
    rollout_result = rollout_engine.rollout(...)
    new_ids = rollout_result.completion_ids[0].tolist()
    new_logps = rollout_result.per_token_logps[0].tolist()
    ...
    calls = parse_tool_calls(new_text)
    if not calls:
        break
    messages.append({"role": "assistant", "content": new_text})
    for call in calls:
        ...
        result = execute_tool(name, raw)
        result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]
        messages.append({"role": "tool", "content": result_str})
```

这段代码说明：

- 每一轮都会重新把 `messages` 渲染成 prompt。
- 只有模型生成的 `new_ids/new_logps` 才是 policy action。
- 如果这一轮没有 tool call，trajectory 结束。
- 如果有 tool call，就把 assistant 文本和 tool result 追加到 `messages`，下一轮继续。
- `result_str[:2048]` 防止工具结果太长撑爆后续 tokenizer。

#### 源码证据：tool observation 不参与 policy loss

文件：`trainer/train_agent.py:146-153`

看它是为了理解：工具返回内容会进入训练序列，但 mask 为 0。

代码摘录：

```python
observe_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
current_len = len(prompt_ids) + len(response_ids)
obs_delta = observe_ids[current_len:]
response_ids.extend(obs_delta)
response_mask.extend([0] * len(obs_delta))
response_old_logps.extend([0.0] * len(obs_delta))
```

这段代码说明：

- `obs_delta` 是本轮工具观察造成的新增上下文 token。
- 它们会进入 `response_ids`，因为下一轮生成需要看到这些上下文。
- 它们的 `response_mask` 是 0，因此不参与 policy loss。
- 它们没有 rollout logprob，所以 `response_old_logps` 用 0.0 占位。

### 4.5 trajectory reward

#### 源码证据 A：reward 输入

文件：`trainer/train_agent.py:188-200`

看它是为了理解：reward 函数拿到的是 prompt、completion、GT、tools 和多轮输出。

代码摘录：

```python
def calculate_rewards(prompts, completions, gt_batch, tools_batch, num_gen, reward_model=None, device="cuda", turn_outputs_batch=None, unfinished_batch=None):
    rewards = torch.zeros(len(completions), device=device)
    for idx, response in enumerate(completions):
        reward, answer = 0.0, response
        sample_idx = idx // num_gen
        tools = tools_batch[sample_idx]
        turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch is not None else [response]
        unfinished = unfinished_batch[idx] if unfinished_batch is not None else False
        turn_answers = [turn.split('</think>', 1)[-1].strip() if '</think>' in turn else turn.strip() for turn in turn_outputs]
        answer = turn_answers[-1] if turn_answers else response.strip()
        valid_names = {t['function']['name'] for t in tools} if tools else set()
        tool_calls = []
        for turn_answer in turn_answers: tool_calls.extend(parse_tool_calls(turn_answer))
```

这段代码说明：

- `idx // num_gen` 把 flat trajectory 映射回原始样本。
- reward 会看一条 trajectory 的所有 turn，不只看最后一段 completion。
- `valid_names` 来自当前样本可用工具，不是全局工具全集。

#### 源码证据 B：没有 tool call 时走普通回答 reward

文件：`trainer/train_agent.py:201-218`

看它是为了理解：Agent 训练仍然允许不用工具的回答。

代码摘录：

```python
reward -= 0.5 * sum(abs(turn.count('<tool_call>') - turn.count('</tool_call>')) for turn in turn_answers)
if not tool_calls:
    reward += 0.5 if 5 <= len(response.strip()) <= 800 else -0.5
    if '</think>' in response:
        think, answer = response.split('</think>', 1)
        reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5
        reward += 0.25 if response.count('</think>') == 1 else -0.25
        answer = answer.strip()
    if reward_model is not None:
        ...
        score = reward_model.get_score(messages, answer)
        reward += score
    reward -= rep_penalty(answer)
    rewards[idx] = max(min(reward, 3.0), -3.0)
```

这段代码说明：

- 标签开闭数量不一致会扣分。
- 没有工具调用时，reward 类似 SFT/RLAIF 的普通回答打分。
- reward model 是可选叠加项。
- 分数最终 clip 到 `[-3.0, 3.0]`。

#### 源码证据 C：有 tool call 时看工具合法性和 GT 命中

文件：`trainer/train_agent.py:219-239`

看它是为了理解：工具轨迹的主要 reward 来自“调用是否对齐”和“最终答案是否命中 GT”。

代码摘录：

```python
else:
    gt = gt_batch[sample_idx]
    valid_call_count = 0
    for tool_call in tool_calls:
        name, raw = tool_call.get("name", ""), tool_call.get("arguments", {})
        if isinstance(raw, str):
            try: raw = json.loads(raw)
            except: raw = {}
        check = CHECK_ARGS.get(name)
        valid_call_count += int(bool(name in valid_names and check and check(raw)))
    tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)
    reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap

    final_text = "" if unfinished else (answer.split('</tool_call>')[-1] if '</tool_call>' in answer else answer)
    verified = validate_gt_in_text(final_text, gt) if gt else set()
    if gt: reward += 2.5 * len(verified) / len(gt)
    if unfinished: reward -= 0.5
    reward -= rep_penalty(final_text if final_text else answer)
    rewards[idx] = max(min(reward, 3.0), -3.0)
```

这段代码说明：

- 工具名必须在当前样本可用工具里。
- 参数必须通过 `CHECK_ARGS`。
- `tool_gap` 同时惩罚工具数量不匹配和无效调用。
- `validate_gt_in_text` 只检查最终文本，不检查 tool observation 本身。
- 如果达到 `max_turns` 仍未完成，会扣 `unfinished` 分。

### 4.6 Agent RL 更新：GRPO/CISPO 没换

#### 源码证据 A：rollout 后打包 token、mask、old logprob

文件：`trainer/train_agent.py:242-273`

看它是为了理解：多轮 trajectory 最后会被整理成训练张量。

代码摘录：

```python
messages_batch = batch['messages']
tools_batch = batch['tools']
gt_batch = batch['gt']
...
completions, contexts, prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch, turn_outputs_batch, unfinished_batch = rollout_batch(...)
...
for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch):
    ids = p + r
    mask = [0] * len(p) + m
    old_logps = [0.0] * max(len(p) - 1, 0) + old_lp
    ...
input_ids = torch.tensor(...)
full_response_masks = torch.tensor(...)
old_per_token_logps = torch.tensor(...)
...
rewards = calculate_rewards(...)
```

这段代码说明：

- `ids = prompt + response` 是完整模型输入。
- `mask = prompt_mask_0 + response_mask` 决定哪些 token 是 policy action。
- `old_logps` 要和 `input_ids[:, 1:]` 对齐，所以 prompt 部分长度是 `len(p)-1`。
- reward 在打包后才计算，仍然是每条 trajectory 一个标量。

#### 源码证据 B：current/ref logprob、EOS mask、group advantage

文件：`trainer/train_agent.py:275-318`

看它是为了理解：Agent 训练复用第 24-25 课的 GRPO/CISPO 张量逻辑。

代码摘录：

```python
res = model_unwrapped(input_ids, attention_mask=full_mask)
logits = res.logits[:, :-1, :]
per_token_logps = F.log_softmax(logits, dim=-1).gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

with torch.no_grad():
    ref_per_token_logps = compute_per_token_logps(ref_model, input_ids, input_ids.size(1) - 1, attention_mask=full_mask)

completion_mask = full_response_masks[:, 1:]
...
completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
token_counts = completion_mask.sum(dim=1)
...
grouped_rewards = rewards.view(-1, args.num_generations)
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
advantages = (rewards - mean_r) / (std_r + 1e-4)
```

这段代码说明：

- `per_token_logps` 是 current policy 对每个 next token 的 logprob。
- `ref_per_token_logps` 来自冻结 reference model。
- `completion_mask` 从 `full_response_masks[:, 1:]` 来，是为了和 next-token logprob 对齐。
- group advantage 的 shape 仍然是 `[B * num_generations]`。

#### 源码证据 C：GRPO/CISPO loss 和 rollout policy 同步

文件：`trainer/train_agent.py:319-364`

看它是为了理解：Agentic RL 的策略更新公式与第 24 课一致，只是 reward 和 mask 来自多轮工具轨迹。

代码摘录：

```python
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1
ratio = torch.exp(per_token_logps - old_per_token_logps)
if args.loss_type == "cispo":
    clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
    per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
else:
    clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
    per_token_loss1 = ratio * advantages.unsqueeze(1)
    per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
    per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
policy_loss = ...
loss.backward()
...
if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)
```

这段代码说明：

- `ratio` 比较 current policy 和 rollout old policy。
- `per_token_kl` 比较 current policy 和 frozen reference model。
- `completion_mask` 保证只有 policy 生成 token 参与 loss。
- 更新/保存后要同步 rollout engine，否则后续采样还会使用旧 policy。

### 4.7 推理侧：eval 和 API 都要解析 tool_call

#### 源码证据 A：本地/API 评测循环

文件：`scripts/eval_toolcall.py:177-199`

看它是为了理解：推理时也要循环“模型输出 -> 执行工具 -> 添加 tool message”。

代码摘录：

```python
def run_case(prompt, tools, args, model=None, tokenizer=None, client=None):
    messages = [{"role": "user", "content": prompt}]
    while True:
        if args.backend == 'local':
            content = generate(model, tokenizer, messages, tools, args)
            tool_calls = parse_tool_calls(content)
        else:
            content, tool_calls = chat_api(client, messages, tools, args, stream=bool(args.stream))
        if not tool_calls:
            break
        ...
        messages.append(...)
        for tc in tool_calls:
            result = execute_tool(...)
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)} ...)
```

这段代码说明：

- eval 不是只看第一轮输出。
- 只要模型继续生成 tool call，脚本就继续执行工具并扩展 messages。
- 本地后端使用文本解析；API 后端使用 OpenAI 风格 `tool_calls` 字段。

#### 源码证据 B：OpenAI API 服务端解析 tool_calls

文件：`scripts/serve_openai_api.py:83-102`

看它是为了理解：服务端把 MiniMind 原生文本协议转换成 OpenAI 兼容响应字段。

代码摘录：

```python
def parse_response(text):
    reasoning_content = None
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    ...
    tool_calls = []
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        try:
            call = json.loads(m.strip())
            tool_calls.append({"id": f"call_{int(time.time())}_{i}", "type": "function", "function": {"name": call.get("name", ""), "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)}})
        except Exception:
            pass
    if tool_calls:
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    return text.strip(), reasoning_content, tool_calls or None
```

这段代码说明：

- MiniMind 内部仍然生成 `<tool_call>` 文本。
- API 层把它转成 `message.tool_calls`。
- 如果有 tool call，正文里会移除 `<tool_call>...</tool_call>` 片段。
- 这是部署兼容层，不是训练 loss 的一部分。

<a id="l26-must-write"></a>
## 5. 本节必须会写 / 暂时不要求

必须会写：

```text
1. 从一段模型输出中解析 <tool_call>...</tool_call>。
2. 判断 tool name 是否在当前样本可用工具列表里。
3. 判断 arguments 是否满足 CHECK_ARGS。
4. 执行一个 mock tool，得到 JSON observation。
5. 把 tool observation 作为 tool message 加回 messages。
6. 区分 policy token mask 和 tool observation mask。
7. 根据 tool_gap、GT 命中、unfinished、格式闭合计算 trajectory reward。
```

暂时不要求：

```text
1. 实现生产级工具沙箱。
2. 接入真实天气、汇率、搜索 API。
3. 训练长期记忆、多任务规划、浏览器控制等完整 Agent 系统。
4. 手写 SGLang rollout server。
5. 重新实现 OpenAI API 服务端。
6. 在没有 agent_rl 数据和 reward model 的情况下跑完整 train_agent.py。
```

当前工作区缺少 `dataset/agent_rl.jsonl` 和默认 `reward_model_path` 指向的 reward model。第 26 课的验收实验因此选择验证解析和 reward 逻辑，不把完整训练作为本地必跑项。

<a id="l26-handwrite"></a>
## 6. 手写与实验模块

本节不新增 `course/impl/core/` 下的核心 RL loss。第 24-25 课已经把 `GRPO/CISPO`、`reward/logprob/KL` 的核心张量逻辑放进了手写复现路线。

本节新增一个源码观察实验：

```text
course/labs/trace_agent_tool_reward.py
```

它对齐的源码函数：

```text
trainer/train_agent.py:76-82      parse_tool_calls
trainer/train_agent.py:84-95      execute_tool
trainer/train_agent.py:183-186    validate_gt_in_text
trainer/train_agent.py:188-239    calculate_rewards
trainer/train_agent.py:314-318    group reward normalization
```

实验检查的行为：

```text
1. 合法 tool_call 能被解析。
2. 合法工具和参数能被 mock executor 执行。
3. 错误参数、未知工具、标签不闭合会改变 reward。
4. 最终答案命中 gt 会增加 reward。
5. 同 prompt 多条 completion 的 reward 可以转成 GRPO-style advantage。
```

刻意简化：

```text
1. 不加载 tokenizer。
2. 不加载 MiniMind 模型。
3. 不采样真实 rollout。
4. 不初始化 reward model。
5. 不做反向传播。
```

这样做是为了单独验证 Agent reward 逻辑，避免被模型权重、数据下载、GPU、reward model 路径干扰。

<a id="l26-experiment"></a>
## 7. 实验验证

### 实验：解析 tool_call 并计算 trajectory reward

这个实验验证：

```text
模型输出文本
-> parse_tool_calls
-> execute_tool
-> validate_gt_in_text
-> calculate_rewards
-> group advantage
```

命令：

```bash
cd /home/sun/minimind
python course/labs/trace_agent_tool_reward.py
```

需要记录：

```text
available_tools =
case=0 parsed_calls =
case=0 executed =
case=0 verified_gt =
case=0 reward =
case=1 reward =
case=2 reward =
case=3 reward =
advantages =
adv_mean =
```

输出怎么看：

```text
case=0:
  合法 calculate_math 调用，最终文本命中 9472，reward 应该最高。

case=1:
  工具名合法，但 arguments 缺少 expression，CHECK_ARGS 不通过。

case=2:
  工具名不在当前可用工具里，即使最终文本包含 9472，也会有工具调用惩罚。

case=3:
  <tool_call> 没有闭合标签，parse_tool_calls 解析不到有效调用，格式惩罚会出现。

advantages:
  同一 prompt 的多条候选 completion 被放进一个 group。
  reward 高于组均值的 completion advantage > 0。
  reward 低于组均值的 completion advantage < 0。
```

可选观察：如果你已经准备好了 `agent_rl.jsonl`、reward model 和模型权重，可以再跑真实训练入口。

命令：

```bash
cd /home/sun/minimind/trainer
python train_agent.py --data_path ../dataset/agent_rl.jsonl --num_workers 0 --debug_mode --debug_interval 1
```

这个命令不是当前课程的默认验收命令，因为当前工作区没有 `dataset/agent_rl.jsonl`。

<a id="l26-stage-assembly"></a>
## 8. 阶段组装

到第 26 课为止，RL/Agent 阶段可以这样组合：

```text
第 21-22 课：
  DPO 数据、chosen/rejected logprob、偏好 loss。

第 23 课：
  PPO rollout、old logprob、critic/value、GAE、policy update。

第 24 课：
  GRPO group advantage、CISPO loss 分支。

第 25 课：
  reward / old logprob / current logprob / ref logprob / KL penalty 的统一地图。

第 26 课：
  tool schema、tool_call、tool_response、多轮 rollout、trajectory reward。
```

教学版实现建议：

```text
course/impl/core/grpo.py:
  继续作为 GRPO/CISPO loss 的核心实现。

course/labs/trace_agent_tool_reward.py:
  作为 Agent reward 和工具协议的观察实验。

原项目 trainer/train_agent.py:
  继续承担真实多轮工具环境、reward model、rollout engine、checkpoint、分布式训练。
```

为什么不急着手写完整 Agent 训练脚本：

```text
Agentic RL 的难点不只在 loss，而在环境循环、工具执行、安全边界、数据准备、rollout engine 同步、reward 设计。
课程的核心目标是看懂 MiniMind 源码中这些边界如何连接，而不是复制所有工程外围。
```

阶段验收命令：

```bash
cd /home/sun/minimind
python course/labs/trace_grpo_cispo_loss.py
python course/labs/trace_reward_logprob_kl.py
python course/labs/trace_agent_tool_reward.py
```

Portfolio 记录可以写：

```text
读懂并复现实验验证了 MiniMind Agentic RL 的工具调用链路：
tool schema -> <tool_call> JSON -> mock tool execution -> <tool_response> observation -> trajectory reward -> GRPO/CISPO update。
能够解释 policy token 与 tool observation token 的 mask 区别，以及为什么 Agent reward 是延迟的 trajectory-level signal。
```

<a id="l26-check"></a>
## 9. 本节检查

1. `tool schema`、`tool_call`、`tool_response` 分别是什么？
2. 为什么 `AgentRLDataset.__getitem__` 返回 `messages/tools/gt`，而不是 `input_ids/labels`？
3. `rollout_single` 为什么在 tool observation 之后要重新 `apply_chat_template`？
4. 为什么 `obs_delta` 要进入 `response_ids`，但 `response_mask` 是 0？
5. `tool_gap` 同时惩罚了哪两类错误？
6. `validate_gt_in_text` 检查的是 tool observation 还是最终文本？
7. Agentic RL 里的 `rewards.view(-1, num_generations)` 为什么仍然要求同 prompt 的多条轨迹连续排列？

<a id="l26-next"></a>
## 10. 下一课

第 27 课进入模型转换、OpenAI API 与 WebUI。

下一课要解决：

```text
convert_model.py 如何把原生 MiniMind 权重转换成 transformers 格式；
serve_openai_api.py 如何包装成 OpenAI 兼容接口；
web_demo.py 如何把 tokenizer/template/generate 接成可交互 UI；
tool_calls / reasoning_content / open_thinking 如何在部署侧流转；
部署脚本复用了哪些训练时已经学过的模型与 tokenizer 约定。
```
