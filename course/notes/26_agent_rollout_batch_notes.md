# 第26节：Agent Rollout 单条轨迹与批量采样笔记

> 我的问题笔记 | 聚焦 `rollout_single` 为什么是单条轨迹，以及能不能多个样本一起采样

---

## 目录

1. [Q1: `rollout_single` 是只跑一条 message 吗？](#q1-rollout_single-是只跑一条-message-吗)
2. [Q2: 当前 `rollout_batch` 真的是批量生成吗？](#q2-当前-rollout_batch-真的是批量生成吗)
3. [Q3: 底层 `rollout_engine.rollout` 支持 batch 吗？](#q3-底层-rollout_enginerollout-支持-batch-吗)
4. [Q4: 为什么 Agent rollout 没有直接写成 batch？](#q4-为什么-agent-rollout-没有直接写成-batch)
5. [Q5: 如果要改成批量采样，主循环应该长什么样？](#q5-如果要改成批量采样主循环应该长什么样)
6. [Q6: old_logps 和 completion_mask 为什么要右移对齐？](#q6-logprob-mask-align)
7. [一句话总结](#一句话总结)

---

## Q1: `rollout_single` 是只跑一条 message 吗？

不是。

`rollout_single` 一次跑的是 **一个样本的一条 trajectory**，不是一条单独的 message。

**代码** `trainer/train_agent.py:98-119`

```python
def rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    ...
    for turn in range(max_turns):
        context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
        inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(device)
        ...
        rollout_result = rollout_engine.rollout(
            prompt_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )
```

这里的 `messages` 是消息列表，可能长这样：

```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
]
```

也可能已经包含历史 assistant/tool 消息。

所以它的边界是：

```text
单个样本
-> 单条采样轨迹
-> 轨迹内部最多 max_turns 轮 assistant/tool 交互
```

因为 `context` 是单个字符串：

```python
inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False)
```

所以这里的 tensor 形状通常是：

```text
inputs["input_ids"].shape      = [1, prompt_len]
inputs["attention_mask"].shape = [1, prompt_len]
```

这个 `1` 表示 batch size 是 1。

---

## Q2: 当前 `rollout_batch` 真的是批量生成吗？

不是严格意义上的批量生成。

当前 `rollout_batch` 只是外层名字叫 batch，内部还是循环调用 `rollout_single`。

**代码** `trainer/train_agent.py:159-180`

```python
def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, num_gen, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    ...
    for messages, tools in zip(messages_batch, tools_batch):
        for _ in range(num_gen):
            msgs_copy = [dict(m) for m in messages]
            completion, context, prompt_ids, response_ids, response_mask, response_old_logps, turn_outputs, unfinished = rollout_single(
                rollout_engine, tokenizer, msgs_copy, tools, max_turns, max_new_tokens, thinking_ratio, device
            )
            all_completions.append(completion)
            ...
```

这段实际执行顺序是：

```text
batch 里的第 1 个样本
  -> 第 1 条 trajectory
  -> 第 2 条 trajectory
  -> ...

batch 里的第 2 个样本
  -> 第 1 条 trajectory
  -> 第 2 条 trajectory
  -> ...
```

也就是：

```text
messages_batch 提供多个样本；
num_gen 给每个样本采多条轨迹；
但每条轨迹仍然是单独调用 rollout_single。
```

所以当前写法简单，但生成效率不高。

---

## Q3: 底层 `rollout_engine.rollout` 支持 batch 吗？

支持。

至少 `TorchRolloutEngine.rollout` 本身是按 batched tensor 写的。

**代码** `trainer/rollout_engine.py:75-92`

```python
output_ids = model.generate(
    input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
    attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
    max_new_tokens=max_new_tokens,
    do_sample=True,
    temperature=temperature,
    num_return_sequences=1,
    pad_token_id=self.tokenizer.pad_token_id,
    eos_token_id=self.tokenizer.eos_token_id,
).clone()  # [B, P+R]
prompt_len = prompt_ids.size(1)
completion_ids = output_ids[:, prompt_len:]  # [B, R]
...
completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
```

如果输入是：

```text
prompt_ids.shape      = [B, P]
attention_mask.shape  = [B, P]
num_generations       = G
```

那么这两行：

```python
prompt_ids.repeat_interleave(num_generations, dim=0)
attention_mask.repeat_interleave(num_generations, dim=0)
```

会把 batch 展开成：

```text
[B * G, P]
```

生成结果大致是：

```text
output_ids.shape       = [B * G, P + R]
completion_ids.shape   = [B * G, R]
per_token_logps.shape  = [B * G, R]
completions            = 长度 B * G 的字符串列表
```

所以限制不在底层 rollout engine，而在 agent 多轮工具循环的组织方式。

---

## Q4: 为什么 Agent rollout 没有直接写成 batch？

因为 Agent rollout 不是“一次 prompt -> 一次 answer”的普通生成，而是带环境反馈的动态循环。

**代码** `trainer/train_agent.py:132-152`

```python
calls = parse_tool_calls(new_text)
if not calls:
    break
unfinished = turn == max_turns - 1
messages.append({"role": "assistant", "content": new_text})
for call in calls:
    name, raw = call.get("name", ""), call.get("arguments", {})
    ...
    result = execute_tool(name, raw)
    result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]
    messages.append({"role": "tool", "content": result_str})

observe_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
current_len = len(prompt_ids) + len(response_ids)
obs_delta = observe_ids[current_len:]
response_ids.extend(obs_delta)
response_mask.extend([0] * len(obs_delta))
response_old_logps.extend([0.0] * len(obs_delta))
```

这段说明了几个 batch 难点：

| 难点 | 为什么麻烦 |
|------|------------|
| prompt 长度不同 | 多个 `context` 要 padding 成同一个 tensor |
| 是否继续不同 | 有的样本没有 tool call，要结束；有的样本要继续下一轮 |
| tool 数量不同 | 一条 response 里可能 0 个、1 个或多个 tool call |
| messages 会变 | 每条 trajectory 都要追加自己的 assistant/tool 消息 |
| observation 不是 policy token | tool observation 要放进 `response_ids`，但 `response_mask` 要填 0 |
| logprob 长度不同 | policy 生成 token 有 old logprob，环境 observation token 没有 |

所以不能简单地把：

```python
tokenizer(context, return_tensors="pt")
```

改成：

```python
tokenizer(contexts, return_tensors="pt", padding=True)
```

然后就结束了。

真正的问题是：每一轮之后，batch 里的 trajectory 会分化。

---

## Q5: 如果要改成批量采样，主循环应该长什么样？

正确思路是做 **动态 active batch**。

也就是先把每个样本展开成多条 trajectory，然后每一轮只把还没结束的 trajectory 收集起来批量生成。

伪代码：

```python
trajectories = []

for messages, tools in zip(messages_batch, tools_batch):
    for _ in range(num_gen):
        trajectories.append({
            "messages": [dict(m) for m in messages],
            "tools": tools,
            "active": True,
            "open_thinking": random.random() < thinking_ratio,
            "prompt_ids": None,
            "response_ids": [],
            "response_mask": [],
            "response_old_logps": [],
            "turn_outputs": [],
            "unfinished": False,
        })

for turn in range(max_turns):
    active_trajs = [t for t in trajectories if t["active"]]
    if not active_trajs:
        break

    contexts = [
        tokenizer.apply_chat_template(
            t["messages"],
            tokenize=False,
            add_generation_prompt=True,
            tools=t["tools"],
            open_thinking=t["open_thinking"],
        )
        for t in active_trajs
    ]

    inputs = tokenizer(
        contexts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(device)

    rollout_result = rollout_engine.rollout(
        prompt_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_generations=1,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
    )

    for row, traj in enumerate(active_trajs):
        new_text = rollout_result.completions[row]
        calls = parse_tool_calls(new_text)

        if not calls:
            traj["active"] = False
            continue

        traj["messages"].append({"role": "assistant", "content": new_text})
        for call in calls:
            result = execute_tool(call["name"], call["arguments"])
            traj["messages"].append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
```

这里最重要的是：

```text
batch 维度不是固定的原始样本 batch；
batch 维度是当前 turn 仍然 active 的 trajectory 集合。
```

实现时还要额外小心：

| 点 | 要保证什么 |
|----|------------|
| `prompt_ids` | 每条 trajectory 只记录最初 prompt 的有效 token |
| padding | 批量 tokenizer 后要用 `attention_mask` 找真实长度 |
| `response_mask` | policy 生成 token 填 1，tool observation delta 填 0 |
| `response_old_logps` | policy token 填 rollout logprob，observation token 填 0.0 |
| `final_context` | 每条 trajectory 各自维护，不要被 batch 内其它样本污染 |
| `num_generations` | 展开成 trajectory 后，rollout 每轮通常用 `num_generations=1` 更好管理 |

---

<a id="q6-logprob-mask-align"></a>
## Q6: old_logps 和 completion_mask 为什么要右移对齐？

核心原因是：Causal LM 的 loss/logprob 不是直接对 `input_ids[:, i]` 计分，而是用第 `i` 个位置的 logits 预测第 `i + 1` 个 token。

**代码** `trainer/train_agent.py:255-285`

```python
for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch):
    ids = p + r
    mask = [0] * len(p) + m
    old_logps = [0.0] * max(len(p) - 1, 0) + old_lp
    ...

full_response_masks = torch.tensor(...)
old_per_token_logps = torch.tensor(...)

res = model_unwrapped(input_ids, attention_mask=full_mask)
logits = res.logits[:, :-1, :]
per_token_logps = F.log_softmax(logits, dim=-1).gather(
    2,
    input_ids[:, 1:].unsqueeze(-1)
).squeeze(-1)

completion_mask = full_response_masks[:, 1:]
```

这里有三组长度：

| 张量/列表 | 长度 | 含义 |
|---|---:|---|
| `ids = p + r` | `L` | 完整 token 序列 |
| `mask = [0] * len(p) + m` | `L` | 按原始 token 位置标记哪些 response token 参与训练 |
| `per_token_logps` | `L - 1` | 对 `input_ids[:, 1:]` 这些目标 token 的 logprob |

所以 `per_token_logps[:, i]` 对应的不是 `input_ids[:, i]`，而是：

```text
per_token_logps[:, i]
= log P(input_ids[:, i + 1] | input_ids[:, :i + 1])
```

举例：

```text
p = [A, B, C]
r = [D, E]

input_ids = [A, B, C, D, E]
```

模型实际计分的位置是：

```text
per_token_logps:
预测 B, 预测 C, 预测 D, 预测 E
```

### 为什么 `old_logps` 前面补 `len(p) - 1` 个 0？

`old_lp` 只来自 rollout 期间生成的 response token，不包含 prompt 内部 token 的 old logprob。

但是训练时的 `per_token_logps` 包含 prompt 内部的目标位置：

```text
预测 B
预测 C
预测 D
预测 E
```

其中 `预测 B`、`预测 C` 这两个位置数量正好是：

```python
len(p) - 1
```

所以要写成：

```python
old_logps = [0.0] * max(len(p) - 1, 0) + old_lp
```

对上面的例子就是：

```text
old_lp    = [old_logp(D), old_logp(E)]
old_logps = [0.0, 0.0, old_logp(D), old_logp(E)]
```

前面的 `0.0` 不是有意义的 old policy 概率，只是占位。它们之后会被 `completion_mask` 盖掉，不参与 policy loss。

注意这里不是补 `len(p)` 个 0。因为 `per_token_logps` 已经丢掉了第一个 token 的目标位置，只剩 `input_ids[:, 1:]`，prompt 内部还需要对齐的目标 token 是 `B, C`，数量是 `len(p) - 1`。

### 为什么 `completion_mask = full_response_masks[:, 1:]`？

`full_response_masks` 是按原始 `input_ids` 位置标的：

```text
input_ids:            [A, B, C, D, E]
full_response_masks:  [0, 0, 0, 1, 1]
```

但 `per_token_logps` 是按目标 token 位置标的：

```text
per_token_logps:      [预测 B, 预测 C, 预测 D, 预测 E]
```

所以 mask 也必须右移成：

```python
completion_mask = full_response_masks[:, 1:]
```

对应结果是：

```text
completion_mask:      [0, 0, 1, 1]
```

这样才表示：

```text
预测 B: 不训练
预测 C: 不训练
预测 D: 训练
预测 E: 训练
```

如果错用 `full_response_masks[:, :-1]`，会得到：

```text
[0, 0, 0, 1]
```

这样第一个 response token `D` 的 logprob 会被 mask 掉，loss 位置整体错一格。

最后要把 tool observation 也放进这个逻辑里理解：`m` 里 policy 生成 token 是 `1`，tool observation delta 是 `0`；`old_lp` 里 observation token 也用 `0.0` 占位。真正决定哪些 token 参与 loss 的是右移后的 `completion_mask`。

---

## 一句话总结

`rollout_single` 当前一次只跑 **一条 trajectory**，所以 tensor 是 `[1, seq_len]`。

底层 `rollout_engine.rollout` 支持 `[B, seq_len]` 批量生成，但 Agent rollout 有 tool call、tool observation、提前结束、多轮上下文变长这些动态分支。

因此可以批量采样，但要写成 **按 active trajectory 动态组 batch**，不能只把外层 `for` 循环删掉。
