# 第 22 课：DPO loss 源码

这一课只解决一个问题：MiniMind 的 `train_dpo.py` 如何把 policy/reference 对 chosen/rejected 的 token logprob 变成一个可反向传播的 DPO loss。

## 目录

- [0. 本节主线](#l22-mainline)
- [1. 本节要懂的 6 个原理](#l22-principles)
- [2. DPO loss 完整原理](#l22-complete-principle)
- [3. 源码阅读顺序图](#l22-reading-order)
- [4. MiniMind 源码走读](#l22-source-walkthrough)
- [5. 本节必须会写 / 暂时不要求](#l22-must-write)
- [6. 手写模块](#l22-handwrite)
- [7. 实验验证](#l22-experiment)
- [8. 阶段组装](#l22-stage-assembly)
- [9. 本节检查](#l22-check)
- [10. 下一课](#l22-next)

<a id="l22-mainline"></a>
## 0. 本节主线

DPO loss 的计算链路是：

```text
policy/ref 分别 forward 同一批 x
-> logits_to_log_probs 取出每个 y token 的 logprob
-> mask 去掉 prompt/padding，只保留 assistant 回复 token
-> 每条序列沿 seq 维求和，得到 sequence logprob
-> batch 前半是 chosen，后半是 rejected
-> policy chosen - policy rejected 得到 policy logratio
-> ref chosen - ref rejected 得到 reference logratio
-> 两个 logratio 相减，得到 policy 相对 reference 的偏好提升
-> loss = -logsigmoid(beta * improvement)
```

一句话：

```text
DPO 奖励的是：policy 相比 reference，更明显地偏向 chosen 而不是 rejected。
```

这节课不是重新讲 `DPODataset`。第 21 课已经解决：

```text
chosen/rejected -> x/y/mask
policy/ref -> token logprob
```

第 22 课只处理：

```text
token logprob -> sequence logprob -> logratio -> DPO loss
```

<a id="l22-principles"></a>
## 1. 本节要懂的 6 个原理

| 原理 | 要理解什么 | 源码位置 |
|---|---|---|
| token logprob 来自 gather | 从 `[batch, seq, vocab]` 里取出目标 token 的 log probability | `trainer/train_dpo.py:25-31` |
| sequence logprob 是 masked sum | 只把 assistant 区域 token logprob 加起来 | `trainer/train_dpo.py:34-37` |
| batch 前半/后半有语义 | 前半是 chosen，后半是 rejected | `trainer/train_dpo.py:39-44` |
| policy logratio 表示 policy 偏好 | `chosen_policy - rejected_policy` 越大，policy 越偏向 chosen | `trainer/train_dpo.py:46` |
| reference logratio 是基准偏好 | DPO 看 policy 相对 ref 的偏好变化，不只看绝对概率 | `trainer/train_dpo.py:47-48` |
| loss 用 `-logsigmoid` | improvement 越大，loss 越小；beta 控制强度 | `trainer/train_dpo.py:49-50`, `README.md:951-956` |

学完本节，你应该能手算一条偏好样本的：

```text
chosen_policy_logprob
rejected_policy_logprob
chosen_ref_logprob
rejected_ref_logprob
pi_logratio
ref_logratio
logits = pi_logratio - ref_logratio
loss = -logsigmoid(beta * logits)
```

<a id="l22-complete-principle"></a>
## 2. DPO loss 完整原理

### 2.1 为什么不是只最大化 chosen

如果只最大化 chosen，就会退回到类似 SFT：

```text
让 chosen 的概率变高
```

但 DPO 的数据里还有 rejected。偏好信息不是：

```text
chosen 是唯一正确答案
```

而是：

```text
chosen 比 rejected 更好
```

所以 DPO 关注的是两条回答的相对概率：

```text
log p(chosen) - log p(rejected)
```

这就是 logratio。

### 2.2 为什么还要 reference model

只看 policy 的 logratio：

```text
policy_logratio = log pi(chosen) - log pi(rejected)
```

会鼓励 policy 强烈偏向 chosen。但这没有约束 policy 偏离原始 SFT 模型多远。

reference model 提供一个基准：

```text
ref_logratio = log ref(chosen) - log ref(rejected)
```

DPO 看的是：

```text
policy_logratio - ref_logratio
```

含义是：

```text
policy 是否比 reference 更偏向 chosen。
```

如果 reference 本来就更喜欢 chosen，而 policy 只是保持原样，那么提升不大；如果 policy 在训练后更明显地喜欢 chosen，提升变大，loss 变小。

### 2.3 为什么 sequence logprob 要用 sum

一条回复有多个 token。

假设 chosen 回复 token 是：

```text
y = [y1, y2, y3]
```

模型给这条回复的条件概率可以写成：

```text
p(y|x) = p(y1|x) * p(y2|x,y1) * p(y3|x,y1,y2)
```

取 log 后乘法变加法：

```text
log p(y|x) = log p(y1|...) + log p(y2|...) + log p(y3|...)
```

所以源码里：

```python
(log_probs * mask).sum(dim=1)
```

得到的是每条回复的 sequence logprob。

注意：这里用 sum，不是 mean。sum 对应整段序列的 log probability。第 22 课先按源码理解，不展开不同长度归一化的变体。

### 2.4 `-logsigmoid` 在优化什么

DPO 源码最终计算：

```text
loss = -logsigmoid(beta * logits)
```

其中：

```text
logits = policy_logratio - ref_logratio
```

如果 `logits` 很大：

```text
policy 相对 ref 更偏向 chosen
sigmoid(beta * logits) 接近 1
logsigmoid 接近 0
loss 接近 0
```

如果 `logits` 很小甚至为负：

```text
policy 没有比 ref 更偏向 chosen，甚至更偏向 rejected
sigmoid(beta * logits) 小
logsigmoid 是更大的负数
loss 变大
```

所以 DPO loss 的梯度会推动 policy：

```text
提高 chosen sequence logprob
降低 rejected sequence logprob
同时相对 reference 控制变化
```

### 2.5 beta 控制偏好优化强度

`beta` 乘在：

```text
policy_logratio - ref_logratio
```

前面。

直观理解：

```text
beta 越大，模型越强烈响应偏好差异；
beta 越小，更新更温和。
```

MiniMind 默认：

```text
beta = 0.15
```

第 22 课只要求理解 beta 是 DPO loss 的强度系数，不要求从 KL 约束推导它。

<a id="l22-reading-order"></a>
## 3. 源码阅读顺序图

建议按这个顺序读：

```text
1. README.md:948-956
   先看 DPO 公式长什么样。

2. trainer/train_dpo.py:25-31
   看 logits_to_log_probs 如何得到 token logprob。

3. trainer/train_dpo.py:34-37
   看 token logprob 如何通过 mask 变成 sequence logprob。

4. trainer/train_dpo.py:39-44
   看 chosen/rejected 如何按 batch 前后半拆开。

5. trainer/train_dpo.py:46-50
   看 policy/ref logratio 和最终 loss。

6. trainer/train_dpo.py:73-85
   看 train_epoch 如何把 ref/policy logprob 接到 dpo_loss。
```

和前几课的关系：

```text
第 10 课：logits、labels、next-token 对齐
第 21 课：DPO x/y/mask 和 reference model
第 22 课：logprob 和 DPO loss
第 23 课：PPO 训练链路
```

<a id="l22-source-walkthrough"></a>
## 4. MiniMind 源码走读

### 4.1 README 里的 DPO 公式

#### 源码证据：DPO 数学形式

文件：`README.md:948-956`

看它是为了理解：代码里的 `pi_logratios - ref_logratios` 对应公式里的哪一部分。

公式摘录：

$$
\mathcal{L}_{\mathrm{DPO}}
= -\mathbb{E}\left[
\log \sigma\left(
\beta \left[
\log \frac{\pi_{\theta}(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)}
-
\log \frac{\pi_{\theta}(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
\right]
\right)
\right]
$$

把分式里的 log 展开：

$$
\log \frac{\pi_{\theta}(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)}
=
\log \pi_{\theta}(y_w \mid x)
-
\log \pi_{\mathrm{ref}}(y_w \mid x)
$$

所以中括号里的量可以改写成：

$$
\left[
\log \pi_{\theta}(y_w \mid x)
-
\log \pi_{\theta}(y_l \mid x)
\right]
-
\left[
\log \pi_{\mathrm{ref}}(y_w \mid x)
-
\log \pi_{\mathrm{ref}}(y_l \mid x)
\right]
$$

这段公式说明：

- `y_w` 是 winner，也就是 chosen。
- `y_l` 是 loser，也就是 rejected。
- policy 是 `pi_theta`。
- reference 是 `pi_ref`。
- DPO 比的是 chosen/rejected 在 policy 和 reference 下的相对概率。

把公式映射到源码变量：

| 公式符号 | 源码变量 | 含义 |
|---|---|---|
| $\log \pi_{\theta}(y_w \mid x)$ | `chosen_policy_log_probs` | policy 对 chosen 回复的 sequence logprob |
| $\log \pi_{\theta}(y_l \mid x)$ | `reject_policy_log_probs` | policy 对 rejected 回复的 sequence logprob |
| $\log \pi_{\mathrm{ref}}(y_w \mid x)$ | `chosen_ref_log_probs` | reference 对 chosen 回复的 sequence logprob |
| $\log \pi_{\mathrm{ref}}(y_l \mid x)$ | `reject_ref_log_probs` | reference 对 rejected 回复的 sequence logprob |
| $\log \pi_{\theta}(y_w \mid x) - \log \pi_{\theta}(y_l \mid x)$ | `pi_logratios` | policy 对 chosen 相比 rejected 的偏好 |
| $\log \pi_{\mathrm{ref}}(y_w \mid x) - \log \pi_{\mathrm{ref}}(y_l \mid x)$ | `ref_logratios` | reference 对 chosen 相比 rejected 的基准偏好 |
| `pi_logratios - ref_logratios` | `logits` | policy 相对 reference 的偏好提升 |
| $-\log\sigma(\beta \cdot \text{logits})$ | `loss` | DPO loss |

对应到代码就是：

```python
pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
logits = pi_logratios - ref_logratios
loss = -F.logsigmoid(beta * logits)
```

### 4.2 logits_to_log_probs 取每个 token 的 logprob

#### 源码证据：token logprob

文件：`trainer/train_dpo.py:25-31`

看它是为了理解：模型输出 logits 后如何取出目标 token 的 logprob。

代码摘录：

```python
def logits_to_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token
```

这段代码说明：

- `logits` shape 是 `[batch, seq, vocab]`。
- `labels` shape 是 `[batch, seq]`。
- `F.log_softmax(logits, dim=2)` 在 vocab 维得到 log probability。
- `labels.unsqueeze(2)` 变成 `[batch, seq, 1]`，作为 gather 索引。
- 返回值 shape 是 `[batch, seq]`，每个位置是目标 token 的 logprob。

注意：

```text
DPO dataset 已经把 x/y 对齐好；
这里不再做 logits[..., :-1, :] 和 labels[..., 1:]。
```

如果你在这里再 shift 一次，就会错位。

### 4.3 mask 后求和得到 sequence logprob

#### 源码证据：masked sum

文件：`trainer/train_dpo.py:34-37`

看它是为了理解：token-level logprob 如何变成一条回复的 logprob。

代码摘录：

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)
```

这段代码说明：

- 输入的 `ref_log_probs` 和 `policy_log_probs` 都是 `[batch, seq]`。
- `mask` 也是 `[batch, seq]`。
- prompt/padding 区域 mask 为 0，不参与求和。
- assistant 回复区域 mask 为 1，参与求和。
- `sum(dim=1)` 后变成 `[batch]`，每个元素是一条序列的 logprob。

这里覆盖第 21 课的一个关键点：

```text
DPO 的忽略区域靠 mask，不靠 label=-100。
```

### 4.4 按 batch 前后半拆 chosen/rejected

#### 源码证据：batch split

文件：`trainer/train_dpo.py:39-44`

看它是为了理解：为什么 train_epoch 拼接顺序不能乱。

代码摘录：

```python
batch_size = ref_log_probs.shape[0]
chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
reject_ref_log_probs = ref_log_probs[batch_size // 2:]
chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
reject_policy_log_probs = policy_log_probs[batch_size // 2:]
```

这段代码说明：

- `dpo_loss` 假设 batch 前半是 chosen。
- batch 后半是 rejected。
- chosen/ref 和 policy/ref 都按同样位置切分。
- 如果 train loop 拼接顺序错，loss 语义就错。

对应第 21 课的拼接：

```python
x = torch.cat([x_chosen, x_rejected], dim=0)
y = torch.cat([y_chosen, y_rejected], dim=0)
mask = torch.cat([mask_chosen, mask_rejected], dim=0)
```

### 4.5 logratio 表示 chosen 相对 rejected 的偏好

#### 源码证据：policy/ref logratio

文件：`trainer/train_dpo.py:46-48`

看它是为了理解：DPO 为什么不是单独看 chosen 概率。

代码摘录：

```python
pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
logits = pi_logratios - ref_logratios
```

这段代码说明：

- `pi_logratios` 是 policy 对 chosen 相比 rejected 的偏好。
- `ref_logratios` 是 reference 对 chosen 相比 rejected 的基准偏好。
- `logits` 是 policy 相对 reference 的偏好提升。
- 如果 `logits > 0`，policy 比 reference 更偏向 chosen。
- 如果 `logits < 0`，policy 比 reference 更偏向 rejected，或偏好提升不够。

变量名里的 `logits` 这里不是 vocab logits，而是 DPO 二分类目标里的标量分数。它 shape 是：

```text
[batch_size / 2]
```

不要和模型输出的：

```text
[batch, seq, vocab]
```

混淆。

### 4.6 `-logsigmoid` 把偏好提升变成 loss

#### 源码证据：最终 loss

文件：`trainer/train_dpo.py:49-50`

看它是为了理解：为什么 `logits` 越大 loss 越小。

代码摘录：

```python
loss = -F.logsigmoid(beta * logits)
return loss.mean()
```

这段代码说明：

- `beta * logits` 是带强度系数的偏好提升。
- `logsigmoid` 越接近 0，说明 sigmoid 越接近 1。
- 前面加负号后，偏好提升越大，loss 越小。
- `mean()` 对 batch 里的偏好对取平均。

手算时可以记住：

```text
logits 很大 -> loss 接近 0
logits = 0 -> loss = -log(0.5) = 0.693...
logits 为负 -> loss 大于 0.693...
```

### 4.7 train_epoch 如何接入 DPO loss

#### 源码证据：训练步骤

文件：`trainer/train_dpo.py:73-85`

看它是为了理解：DPO loss 如何进入普通训练循环。

代码摘录：

```python
with torch.no_grad():
    ref_outputs = ref_model(x)
    ref_logits = ref_outputs.logits
ref_log_probs = logits_to_log_probs(ref_logits, y)

outputs = model(x)
logits = outputs.logits
policy_log_probs = logits_to_log_probs(logits, y)

dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
loss = dpo_loss_val + outputs.aux_loss
loss = loss / args.accumulation_steps
```

这段代码说明：

- ref model forward 不构建梯度。
- policy model forward 正常构建梯度。
- 两边 logprob 的 shape 都是 `[2 * batch, seq]`。
- `dpo_loss` 返回标量。
- 如果 policy 是 MoE，还会加 `outputs.aux_loss`。
- 后续 backward/clip/step 和普通训练一样。

<a id="l22-must-write"></a>
## 5. 本节必须会写 / 暂时不要求

必须会写：

```text
1. 从 logits 和 labels 得到 token logprob：
   log_probs = log_softmax(logits, dim=-1)
   token_logps = gather(log_probs, labels)

2. 从 token logprob 得到 sequence logprob：
   seq_logps = (token_logps * mask).sum(dim=1)

3. 拆 chosen/rejected：
   chosen = seq_logps[:batch_size//2]
   rejected = seq_logps[batch_size//2:]

4. 计算 logratio：
   pi_logratio = chosen_policy - rejected_policy
   ref_logratio = chosen_ref - rejected_ref

5. 计算 DPO loss：
   logits = pi_logratio - ref_logratio
   loss = -logsigmoid(beta * logits).mean()
```

暂时不要求：

```text
1. 从 PPO KL 约束推导 DPO。
2. beta 的系统调参。
3. reference logprob 预缓存。
4. IPO / KTO / ORPO 等变体。
5. 长度归一化、reward margin、label smoothing 等 DPO 变体。
```

<a id="l22-handwrite"></a>
## 6. 手写模块

本节对应教学版文件：

```text
course/impl/core/losses.py
```

### 6.1 补 `sequence_log_probs`

接口：

```python
def sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    ...
```

对齐源码：

```text
trainer/train_dpo.py::logits_to_log_probs
trainer/train_dpo.py::dpo_loss 的 masked sum
```

行为要求：

```text
输入:
  logits: [batch, seq, vocab]
  labels: [batch, seq]
  mask:   [batch, seq]

输出:
  seq_logps: [batch]
```

内部步骤：

```text
log_softmax
gather labels
乘 mask
sum(dim=1)
```

### 6.2 补 `dpo_loss`

接口：

```python
def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    ...
```

对齐源码：

```text
trainer/train_dpo.py::dpo_loss
```

行为要求：

```text
pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = ref_chosen_logps - ref_rejected_logps
logits = pi_logratios - ref_logratios
loss = -logsigmoid(beta * logits).mean()
```

教学版接口把 chosen/rejected 拆开传入，比原源码更显式：

```text
原源码：输入拼接后的 ref_log_probs / policy_log_probs / mask，再内部拆前后半。
教学版：先在调用处得到四个 sequence logprob，再传给 dpo_loss。
```

这样更容易测试和手算。

<a id="l22-experiment"></a>
## 7. 实验验证

本节新增实验：

```text
course/labs/trace_dpo_loss.py
```

它复用：

```text
course/labs/tiny_dpo.jsonl
```

实验验证：

```text
1. 从 policy/ref logits 得到 token logprob。
2. mask 后得到 chosen/rejected sequence logprob。
3. 手算 pi_logratio 和 ref_logratio。
4. 手算 logits = pi_logratio - ref_logratio。
5. 手算 -logsigmoid(beta * logits)。
6. 和 train_dpo.py::dpo_loss 输出对齐。
```

运行命令：

```bash
cd /home/sun/minimind
python course/labs/trace_dpo_loss.py \
  --data_path course/labs/tiny_dpo.jsonl \
  --max_length 96 \
  --beta 0.15
```

重点记录：

```text
chosen_policy_logp =
rejected_policy_logp =
chosen_ref_logp =
rejected_ref_logp =
pi_logratio =
ref_logratio =
dpo_logits =
manual_dpo_loss =
source_dpo_loss =
abs_diff =
```

初始 policy/ref 权重相同时，常见现象是：

```text
pi_logratio == ref_logratio
dpo_logits 接近 0
loss 接近 0.693147
```

因为训练还没开始，policy 相对 reference 没有偏好提升。

<a id="l22-stage-assembly"></a>
## 8. 阶段组装

DPO 阶段现在有两课：

```text
第 21 课：
  chosen/rejected 数据
  DPODataset
  policy/ref model
  token logprob 输入

第 22 课：
  sequence logprob
  policy/ref logratio
  DPO loss
```

后续教学版 DPO 最小闭环可以这样组装：

```text
CourseDPODataset
-> policy/ref model
-> sequence_log_probs
-> dpo_loss
-> optimizer 更新 policy
-> 保存 dpo 权重
```

### 8.1 阶段验收顺序

建议先跑两个 lab：

```bash
cd /home/sun/minimind
python course/labs/trace_dpo_dataset_reference.py --data_path course/labs/tiny_dpo.jsonl --max_length 96
python course/labs/trace_dpo_loss.py --data_path course/labs/tiny_dpo.jsonl --max_length 96 --beta 0.15
```

如果补了 `course/impl/core/losses.py`，后续可新增：

```text
course/impl/tests/test_dpo.py
```

测试内容：

```text
sequence_log_probs 对齐源码 gather + mask sum
dpo_loss 对齐 train_dpo.py::dpo_loss
```

### 8.2 Portfolio 记录

完成本节后，可以在 `course/portfolio/experiments.md` 记录：

```text
DPO loss trace:
- 用 tiny chosen/rejected 样本构造 x/y/mask。
- 计算 policy/ref token logprob。
- mask 后求 sequence logprob。
- 拆 chosen/rejected，计算 policy/ref logratio。
- 手算 DPO loss 并和源码函数输出对齐。
```

在 `course/notes/mistakes.md` 记录容易错的点：

```text
1. 把 DPO logits 和模型 vocab logits 混淆。
2. 对 DPO 的 x/y 又额外 shift 一次。
3. 忘记 mask，导致 prompt/padding 进入 sequence logprob。
4. chosen/rejected 前后半拆反。
5. 把 alpha/beta 和蒸馏课的 alpha 混淆。
```

<a id="l22-check"></a>
## 9. 本节检查

1. `logits_to_log_probs` 的输出 shape 是什么？
2. 为什么 sequence logprob 用 `(token_logps * mask).sum(dim=1)`？
3. DPO 源码为什么假设 batch 前半是 chosen，后半是 rejected？
4. `pi_logratios` 和 `ref_logratios` 分别代表什么？
5. `logits = pi_logratios - ref_logratios` 里的 `logits` 为什么不是 vocab logits？
6. 当 policy 和 reference 完全一样时，DPO loss 大约是多少？为什么？
7. beta 变大时，`-logsigmoid(beta * logits)` 对同一个 logits 会发生什么变化？

<a id="l22-next"></a>
## 10. 下一课

第 23 课进入 [PPO 训练链路](23_ppo_training_flow.md)。

下一课要解决：

```text
PPO 为什么需要 rollout；
actor、critic、reference、reward model 分别是什么；
response logprob、reward、advantage 如何进入 PPO loss；
clip ratio 和 KL penalty 在源码里如何实现；
PPO 和 DPO 的训练数据、模型数量、更新方式有什么不同。
```
