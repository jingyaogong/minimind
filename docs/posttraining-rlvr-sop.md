# 后训练实习项目 SOP：RLVR 训练动态审计

**版本**：v1.1  
**日期**：2026-09-09  
**适用仓库**：`minimind` 及其 fork  
**目标岗位**：LLM post-training / alignment / RL / reasoning / data quality

> 这是一份执行文档，不是论文综述。照着做，最终应得到：可复现代码、受控实验、评估报告、失败案例和能经得住追问的简历材料。文中没有填写的数字，禁止提前写成“提升了 X%”。

> **当前仓库状态（重要）**：本 SOP 新规划的 `math_verifier.py`、`math_rl_dataset.py`、
> `clean_math_data.py`、`eval_heldout.py`、tests 和 `train_grpo.py` 参数目前不是仓库现成
> 功能。第 8--10 节的命令只能在第 6 节“实现门槛”全部通过后执行；直接跳到 smoke
> 会得到 `FileNotFoundError` 或“unrecognized arguments”，这属于正确的前置检查，不是
> 训练失败。本文把实现顺序、接口和验收条件写全，便于你逐项落地。

## 0. 项目结论先行

### 0.1 项目名称

**RLVR 训练动态审计：奖励、长度偏置与模型能力的解耦分析**

英文名可写作：**Auditing RLVR Training Dynamics: Reward, Length Bias, and Capability Decoupling**。

### 0.2 核心问题

在一个有确定答案的数学任务上，GRPO 训练后出现的 reward 上升，是否真的代表正确率上升？还是模型主要学会了：

- 输出固定格式；
- 延长回答长度；
- 利用 verifier 或 reward model 的漏洞；
- 在组内奖励退化时重复已有行为。

项目不预设正结果。以下三种结论都有效，但证据要求不同：

| 结果 | 可以得出的结论 | 不能得出的结论 |
|---|---|---|
| 正确率上升，长度稳定 | 当前设置下 RLVR 有真实任务收益 | 不能说具备新推理能力 |
| reward/格式率上升，正确率不变 | 发现 reward hacking 或能力瓶颈 | 不能说 RLVR 失败于所有模型 |
| 所有指标都不动 | 需要排查数据、verifier、吞吐和超参 | 不能直接归因于模型容量 |

### 0.3 推荐技术路线

采用两层结构：

1. **白盒层：MiniMind-64M**。保留现有从零实现，用于理解张量流、验证公式、快速做 verifier 和退化机制实验。
2. **工业近似层：Qwen2.5-1.5B-Instruct + TRL/Open-R1 或 verl**。用真实基座、vLLM rollout、FSDP/Ray 组件复现同一套数据和 verifier。

不要在 MiniMind 上从零预训练 0.5B，也不要一开始同时做 PPO、DPO、GRPO、Agent、VLM。主线只做一个问题：**可靠 verifier + GRPO + 训练动态诊断**。

### 0.4 完成标准（Definition of Done）

项目完成必须满足：

- [ ] verifier 单元测试通过，且有人工抽查记录；
- [ ] 训练集和 held-out 集按表达式/题目 key 去重，无交叠；
- [ ] MiniMind rule-only smoke test 至少跑通 2 个 optimizer step；
- [ ] 至少完成 SFT 冷启动和一个 rule-based GRPO run；
- [ ] 至少一个 reward 对照（RM-only 或 calibrated mix）；
- [ ] 有 greedy、pass@1、pass@16（资源允许时 pass@32）评估；
- [ ] 保存每次运行的 config、命令、seed、硬件、日志和 checkpoint；
- [ ] 至少分析 20 条原始 rollout，包含错误类别；
- [ ] README 有限制、失败结果和数据来源说明；
- [ ] 简历只写已经真实跑完的数字。

### 0.5 实现门槛（先完成，再运行后续命令）

按下面顺序提交小而可审查的变更；每一行“验收命令”都应在进入下一阶段前通过：

| 阶段 | 必须新增/修改 | 验收命令/结果 |
|---|---|---|
| I1 | `trainer/math_verifier.py`、`tests/test_math_verifier.py` | `python -m unittest discover -s tests -p 'test*verifier*.py'` 全绿 |
| I2 | `scripts/clean_math_data.py`、`dataset/manifests/` | `python scripts/clean_math_data.py --synthetic-only ...` 生成三件套与 manifest |
| I3 | `dataset/math_rl_dataset.py`、SFT schema 补齐 | 对一条样本打印 prompt/gt；`python -m compileall model trainer dataset scripts` |
| I4 | `trainer/train_grpo.py` 的 reward mode、GT、日志补丁 | `cd trainer && python train_grpo.py --help` 能看到新参数；`rule` 分支不初始化 RM |
| I5 | `scripts/eval_heldout.py` | 用随机/小 checkpoint 跑 1 条 held-out，生成 JSON；再做 MPS 2-step smoke |
| I6 | Qwen/TRL 或 verl 迁移 | 先锁 commit/环境，完成 10--20 step smoke，才允许写主实验结果 |

实现门槛未通过时，只运行 verifier、数据清洗和静态检查；不要把下面的示例命令复制成
“现成可用”的训练结果。

---

## 1. 资源和分支策略

### 1.1 硬件分层

| 环境 | 用途 | 推荐模型/框架 | 允许的结论 |
|---|---|---|---|
| macOS CPU/MPS | 代码、verifier、数据、最小 smoke | MiniMind-64M 原生脚本 | 仅代码通路，不作效果结论 |
| 24GB CUDA | 小规模完整实验 | MiniMind-64M；Qwen 0.5B；Qwen 1.5B 仅作待验证 LoRA smoke（优先 TRL） | 可做受控小实验，记录显存和吞吐；不能承诺 1.5B GRPO 可收敛 |
| 48-80GB CUDA | 主研究 run | Qwen2.5-1.5B + TRL/verl | 可做多 seed 和完整消融 |
| 多卡 | 工业栈/规模敏感性 | Qwen 1.5B-7B + verl | 只在前面结论稳定后做 |

`verl` 官方设备调优页列出了 Qwen2.5-0.5B/1.5B 的 GRPO-LoRA 配置，但其最小示例使用 H100；因此“24GB 可运行”必须通过实际 smoke test 证明，不能从参数量推断。[verl 设备调优](https://verl.readthedocs.io/en/latest/perf/device_tuning.html)

### 1.2 分支和目录约定

建议在自己的 fork 中创建分支：

```bash
git switch -c rlvr-dynamics-audit
```

如果当前工作树没有 `.git`，不要在原目录强行初始化并覆盖用户文件；先复制到自己的 fork 或重新 clone。新增目录：

```text
minimind/
├── configs/
│   ├── mini_rule.yaml
│   ├── mini_rm.yaml
│   └── qwen_grpo.yaml
├── dataset/
│   ├── math_sft_train.jsonl
│   ├── math_rlvr_train.jsonl
│   ├── math_heldout.jsonl
│   └── manifests/
├── trainer/
│   ├── math_verifier.py
│   └── math_rl_dataset.py
├── scripts/
│   ├── clean_math_data.py
│   ├── eval_heldout.py
│   └── summarize_runs.py
├── rewards/
│   └── math_reward.py
├── tests/
│   └── test_math_verifier.py
├── logs/
├── runs/
└── docs/posttraining-rlvr-sop.md
```

每个 run 使用唯一的 `run_tag` 和 `save_weight`，例如：

```text
mini_sft_seed42
mini_rule_seed42
mini_rule_dynamic_seed42
qwen_rule_seed42
```

不要复用同名 checkpoint；MiniMind 的 resume 文件会按前缀复用，重名可能悄悄接着错误的实验训练。

---

## 2. 环境准备

### 2.1 本地静态检查

在仓库根目录执行：

```bash
mkdir -p logs runs out dataset/manifests external configs
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m compileall model trainer dataset scripts eval_llm.py
```

当前 `requirements.txt` 中的 PyTorch 行被注释，需按照机器的 CUDA/MPS 情况安装匹配版本。
MiniMind 这套环境只用于白盒代码和本地 smoke；TRL/Open-R1 的 GRPO 主轨必须另建环境。
不要直接把任意 CUDA wheel 写进 requirements；先按机器选择安装入口，再记录：

```bash
# Apple Silicon：使用与你的 Python 版本匹配的官方 macOS wheel
python -m pip install torch torchvision

# NVIDIA CUDA：把 <cuda-wheel-index> 替换为机器对应的官方 PyTorch wheel 索引
# python -m pip install torch torchvision --index-url <cuda-wheel-index>
```

安装后必须通过 `import torch` 和设备探测；只完成 `pip install -r requirements.txt`
而没有 torch 时，下面的训练命令不能运行。

```bash
python - <<'PY'
import importlib.util
import platform
print("python:", platform.python_version())
for name in ("torch", "transformers", "datasets", "trl"):
    print(name, bool(importlib.util.find_spec(name)))
if importlib.util.find_spec("torch"):
    import torch
    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available())
    print("mps:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
PY
```

正式 CUDA 环境另存：

```bash
python -m pip freeze > runs/environment-pip-freeze.txt
nvidia-smi > runs/environment-nvidia-smi.txt
```

### 2.2 模型和数据下载原则

- 大模型、RM、数据集放在数据盘或仓库外的缓存目录；
- 记录来源 URL、版本/commit、下载日期和 SHA256；
- 不把 checkpoint 提交进 Git；
- 网络不可用时，先用 `--synthetic-only` 完成代码通路，不要把下载失败伪装成训练结果。

MiniMind 官方 README 要求从 ModelScope/Hugging Face 另行下载权重和数据；训练脚本还依赖当前工作目录。训练脚本固定从 `trainer/` 启动，评估脚本固定从仓库根目录启动。

示例布局：

```text
../models/minimind-3-pytorch/
../models/internlm2-1_8b-reward/
../models/Qwen2.5-1.5B-Instruct/
```

MiniMind 的官方权重/数据入口见 README 的 [模型下载](https://www.modelscope.cn/models/gongjy/minimind-3-pytorch)
和 [数据集文件页](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)。如果使用
ModelScope CLI，可以先查看当前 CLI 的参数，再执行类似：

```bash
modelscope download --model gongjy/minimind-3-pytorch \
  --local_dir ../models/minimind-3-pytorch
modelscope download --dataset gongjy/minimind_dataset \
  --local_dir dataset/raw
```

下载后先 `find` 检查文件名，不要假设 release 一定包含每个训练阶段：

```bash
find ../models/minimind-3-pytorch -maxdepth 2 -type f | sort
find dataset/raw -maxdepth 2 -type f | sort
```

MiniMind 训练脚本从 `trainer/` 运行，并按 `../out/<prefix>_768.pth` 查找权重；因此把
确认过的 `full_sft_768.pth`（以及需要的 `grpo_768.pth`）复制到 `out/` 后再训练。RM
按仓库 README 放在 MiniMind 同级目录，或始终传入绝对 `--reward_model_path`。数据集
文件的许可证、下载日期和 SHA256 写入 manifest，不能只保留一个未经说明的本地文件。

若要做“改造前后 RM 路径等价”的回归护栏，还要下载官方 `rlaif.jsonl`；它与本项目
的 `math_rlvr_train.jsonl` 是两套数据，不能混用或互相替代。网络不可达时跳过该回归，
用合成数据验证 rule 分支，并在报告中标明缺失项。

下载后先只验证加载，不立即训练：

```bash
python - <<'PY'
from pathlib import Path
for p in [Path("../models/minimind-3-pytorch"),
          Path("../models/Qwen2.5-1.5B-Instruct")]:
    print(p, "exists=", p.exists())
PY
```

### 2.3 运行深度选择

默认按下面顺序推进：

1. `smoke-test`：MPS/CPU + 合成数据；
2. `local-full-run`：24GB CUDA 上 MiniMind 完整小实验；
3. `remote-full-run`：Qwen + TRL/verl 主实验；
4. 只有主实验结论稳定，才做 3B/7B 单点复核。

---

## 3. 先理解现有 MiniMind 主链

### 3.1 现有模块和责任边界

| 模块 | 现状 | 本项目处理方式 |
|---|---|---|
| `trainer/train_full_sft.py` | 全参 SFT | 作为冷启动入口，复用 |
| `trainer/train_dpo.py` | vanilla DPO | 只作可选离线对照，不作为主线 |
| `trainer/train_ppo.py` | critic + GAE + PPO | 阅读和面试推导，不加入主实验 |
| `trainer/train_grpo.py` | GRPO/CISPO、组内 advantage、KL | 主改造文件 |
| `trainer/rollout_engine.py` | torch 与 SGLang rollout | 主实验固定 torch；SGLang 后续再审计 |
| `trainer/train_agent.py` | 固定 mock tools 的多轮 RL | 不并入数学主线 |
| `dataset/lm_dataset.py` | SFT/RLAIF 数据集 | 新增单轮带 `gt` 的 Dataset |
| `eval_llm.py` | 交互式聊天 | 不足以做指标，新写批量评估 |

### 3.2 关键数据流和张量形状

设 batch size 为 `B`，每个 prompt 采样 `G` 条，completion 长度为 `R`：

```text
prompt[B]
  -> tokenize[B, P]
  -> rollout[B*G, P+R]
  -> completions[B*G]
  -> gt_batch[B] -> repeat_interleave(G) -> gt_for_rollouts[B*G]
  -> reward[B*G]  (assert len(gt_for_rollouts) == len(completions))
  -> grouped_reward[B, G]
  -> advantage[B*G]
  -> policy/ref logprob[B*G, R]
  -> masked GRPO loss
```

MiniMind 当前 `train_grpo.py` 的重要事实：

- 默认 `G=6`；
- 默认 `loss_type=cispo`，正式实验必须显式 `--loss_type grpo`；
- reward 是格式/长度/重复惩罚加 1.8B RM，不是数学 GT verifier；
- `RLAIFDataset` 不返回 `gt`；
- 组内 reward 全相同时，advantage 的 policy-gradient 部分为零，但 KL 项仍可能有梯度；
- SGLang 只在保存间隔同步权重，存在 stale-policy 风险；
- RM 逐样本串行调用，容易成为训练热点。

这些事实决定了本项目的改造边界：**不重写 rollout 和 loss 主循环，只替换奖励来源、数据入口、指标和评估。**

---

## 4. 数据工程 SOP

### 4.1 数据 schema

SFT 样例：

```json
{"conversations":[
  {"role":"user","content":"请计算 17 * 23，并在最后一行以‘答案是 数字’输出。","reasoning_content":"","tools":"","tool_calls":""},
  {"role":"assistant","content":"答案是 391","reasoning_content":"先计算 17*20，再计算 17*3，最后相加。","tools":"","tool_calls":""}
]}
```

这五个 message 字段（`role/content/reasoning_content/tools/tool_calls`）是为了兼容仓库
`SFTDataset` 当前的显式 `datasets.Features`。assistant 的思考放在
`reasoning_content`，最终答案放在 `content`；不要把同一段 `<think>...</think>` 同时写进
两处。对不涉及思考或工具的消息，清洗器统一补空字符串。
RLVR 文件则只保留 user message 和顶层 `gt/source/difficulty/sample_id`，由新的
`MathRLDataset` 读取。

RLVR 样例：

```json
{"conversations":[
  {"role":"user","content":"请计算 17 * 23，并在最后一行以‘答案是 数字’输出。"}
],"gt":"391","source":"synthetic","difficulty":"easy","sample_id":"..."}
```

held-out 样例与 RLVR 样例相同，但只能放在 `math_heldout.jsonl`，不能被 SFT 或 RL 读取。

### 4.2 数据来源选择

优先级：

1. 仓库作者公开的 `agent_rl_math`，下载后先检查真实字段类型；
2. 自造确定性算术题，作为可控实验集；
3. 不要未经清洗直接把 GSM8K 翻译集、网上答案或聊天数据混进训练集。

`agent_rl_math` 的具体 schema 需要下载后确认，不能假设 `gt` 一定是字符串，也不能假设每条对话都是单轮。清洗器必须对坏行、缺字段、无法解析的答案显式统计。官方 `rlaif.jsonl` 常见的末尾占位 assistant 会被现有 `RLAIFDataset` 的 `conversations[:-1]` 丢弃；但自造 RLVR 数据若沿用“user + 占位 assistant”而新 `MathRLDataset` 不切片，占位内容会进入 prompt，并可能被 RM 的历史消息解析当成真实回答。因此自造的 `math_rlvr_train.jsonl` 一律只放 user 消息和 `gt`，不放占位 assistant。

### 4.3 清洗流程

实现 `scripts/clean_math_data.py`，按以下顺序处理：

1. 逐行 JSON 解析，记录坏行编号；
2. 保留包含 `gt` 和可解析 user 内容的样本；
3. 移除工具定义、tool response 和不可还原的多轮轨迹；
4. 统一题目格式，加入固定输出指令；
5. 只保留能被 `Decimal` 解析的阿拉伯数字答案；
6. 对表达式做规范化并生成 `sample_id`；
7. 先按 key 去重，再按 key 的 hash 分层切分 train/held-out；
8. 输出 manifest、来源占比、答案类型、难度桶和丢弃原因。

清洗器的最小 CLI 契约固定为：

```text
--input <raw-jsonl>          # 与 --synthetic-only 二选一
--out-dir dataset/
--synthetic-only
--train-size N --heldout-size N
--seed 42
--check-only                 # 只校验已有三件套和 schema，不重写文件
```

`--check-only` 至少验证：SFT 每条 message 有五个 canonical 字段、RLVR/held-out 每条
都有非空 user 和可归一化 `gt`、`sample_id` 唯一、train/held-out key 零交叠。把这些
检查放进清洗器本身，而不是依赖人工看几行 JSON。

在 `--synthetic-only` 模式下，默认输出 `math_sft_train.jsonl`、
`math_rlvr_train.jsonl`、`math_heldout.jsonl` 和 `manifests/math_manifest.json`；其中
SFT 文件包含带 `reasoning_content` 的 assistant 答案，RLVR/held-out 文件不包含占位
assistant。若另设文件名，必须同步修改后续命令和 manifest。

### 4.4 自造题生成器

只允许白名单表达式：整数、`+ - * / ** ( )`。不要对外部字符串直接调用 unrestricted `eval`，也不要仅靠 `compile(..., {"__builtins__": {}})` 作为安全边界；应使用 AST 白名单和递归求值，并限制指数与结果范围：

```python
import ast

MAX_EXPR_LEN = 128
MAX_NODES = 64
MAX_ABS = 10**12
MAX_EXP = 8

def safe_eval(expr: str) -> str:
    if not expr or len(expr) > MAX_EXPR_LEN:
        raise ValueError("expression too long")
    tree = ast.parse(expr, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ValueError("expression too deep")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            if abs(node.value) > MAX_ABS:
                raise ValueError("literal out of range")
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif isinstance(node.op, ast.Div):
                if right == 0 or left % right != 0:
                    raise ValueError("non-integral or zero division")
                value = left // right
            elif isinstance(node.op, ast.Pow):
                if right < 0 or right > MAX_EXP:
                    raise ValueError("exponent out of range")
                value = left ** right
            else:
                raise ValueError(type(node.op).__name__)
            if abs(value) > MAX_ABS:
                raise ValueError("result out of range")
            return value
        raise ValueError(type(node).__name__)

    return str(visit(tree))
```

生成三档难度：

| 难度 | 例子 | 目的 |
|---|---|---|
| easy | `17*23`、`94-35` | 先验证格式和基础答案学习 |
| medium | `(17+23)*4-9` | 检查两步组合泛化 |
| hard | `(4**3)+(20**2)-17` | 检查更长表达式和长度行为 |

不要把同一个表达式仅换标点后同时放进 train 和 held-out；按表达式规范化 key 去重。

### 4.5 数据验收门槛

清洗脚本完成后必须生成 `dataset/manifests/math_manifest.json`，至少包含：

```json
{
  "source_files": [],
  "source_sha256": {},
  "train_count": 0,
  "heldout_count": 0,
  "duplicate_count": 0,
  "discard_reasons": {},
  "difficulty_counts": {},
  "overlap_count": 0
}
```

退出条件：

- 200 条 verifier 合成用例全部有预期标签；
- 至少 50 条人工核对 `gt`；
- `overlap_count == 0`；
- held-out 至少 200 条，或在资源很紧时明确写出较小样本限制；
- 所有丢弃样本有原因，不允许静默过滤。

---

## 5. Verifier 实现 SOP

### 5.1 设计原则

Verifier 是项目最重要的 ownership。它必须：

- 在模型外部计算 reward；
- 先拆出 think 段，再读取最终答案；
- 只取最后一个答案 marker 后的内容；
- 采用精确数值归一化，不做危险的 substring 匹配；
- 将 `correct`、`incorrect`、`no_answer`、`invalid_gt` 分开记录；
- 不为了提高命中率而“猜答案”。

默认 marker 选择 `答案是`，也可以统一改成 `####`。一旦选择，必须同步修改：数据清洗、SFT gold、RL Dataset、verifier、eval 脚本和 README。

### 5.2 `trainer/math_verifier.py` 最小接口

建议先实现下面的接口，再接训练：

```python
from decimal import Decimal, InvalidOperation
import re

ANSWER_MARKER = "答案是"
MAX_ANSWER_LEN = 64

def split_think(response: str) -> tuple[str, str, bool]:
    """返回 thinking、final、think_closed。取最后一个闭合标签后的文本。"""
    text = response or ""
    if "<think>" in text and "</think>" not in text:
        # 出现未闭合思考标签时，不把其中的 marker 当成最终答案。
        return "", "", False
    if "</think>" not in text:
        return "", text.strip(), False
    thinking, final = text.rsplit("</think>", 1)
    thinking = thinking.replace("<think>", "", 1).strip()
    return thinking, final.strip(), True

def normalize_number(text: str) -> str | None:
    """只接受有限十进制；不解析中文数字、单位和分数。"""
    if text is None:
        return None
    text = (text.replace("\u3000", " ")
                .replace("，", ",")
                .replace("．", ".")
                .replace("０", "0").replace("１", "1")
                .replace("２", "2").replace("３", "3")
                .replace("４", "4").replace("５", "5")
                .replace("６", "6").replace("７", "7")
                .replace("８", "8").replace("９", "9"))
    text = text.strip()
    if "," in text:
        # 只接受标准千分位，避免把任意逗号文本拼成数字。
        if not re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
            return None
        text = text.replace(",", "")
    if not text or len(text) > MAX_ANSWER_LEN:
        return None
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return None
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    normalized = format(value, "f")
    # 只去掉小数部分的尾随零；不能把整数 1000 变成 1。
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in ("", "+0", "-0") else normalized

def extract_answer_text(response: str, marker: str = ANSWER_MARKER) -> str | None:
    _, final, _ = split_think(response)
    if marker not in final:
        return None
    tail = final.rsplit(marker, 1)[1].strip()
    # 只读取 marker 后的第一行，避免模型列出多个候选答案作弊。
    first_line = tail.splitlines()[0].strip() if tail else ""
    return first_line or None

def verdict(response: str, gt: str, marker: str = ANSWER_MARKER) -> dict:
    extracted = extract_answer_text(response, marker)
    pred = normalize_number(extracted)
    gold = normalize_number(gt)
    _, final, think_closed = split_think(response)
    return {
        "status": ("invalid_gt" if gold is None else
                   "no_answer" if pred is None else
                   "correct" if pred == gold else "incorrect"),
        "correct": bool(gold is not None and pred is not None and pred == gold),
        "marker_hit": marker in final,
        "think_closed": think_closed,
        "extracted": extracted,
        "normalized_pred": pred,
        "normalized_gt": gold,
    }

def reward_tier(v: dict, mode: str = "binary") -> float:
    if v["status"] == "correct":
        return 1.0
    if v["status"] == "no_answer":
        return -0.5 if mode == "binary" else -1.0
    if v["status"] == "invalid_gt":
        raise ValueError("invalid ground truth reached reward")
    return 0.0 if mode == "binary" else -0.5
```

实现时不要把上面代码原样当成完整生产解析器。需要补充：答案行边界、负数、全角字符、重复 marker、异常 JSON、日志和单测。

### 5.3 单元测试矩阵

至少覆盖：

```text
59 vs 59.0                    -> correct
1,000 vs 1000                 -> correct
全角数字 ３９１ vs 391         -> correct
-3 vs 3                       -> incorrect
缺少 marker                   -> no_answer
只有 <think> 没有 </think>     -> no_answer
marker 在 think 段，final 无 marker -> no_answer
答案是 1 / 答案是 2             -> 只取最后 marker 后第一行
答案是 1959, gt=59             -> incorrect，不允许 substring 命中
gt 含单位/中文数字              -> invalid_gt，清洗阶段丢弃
```

测试命令：

```bash
python -m unittest discover -s tests -p 'test*verifier*.py'
```

若希望支持单文件自检，再在 `math_verifier.py` 末尾添加 `if __name__ == "__main__":` 的最小断言入口；没有这段入口时不要把 `python -m trainer.math_verifier` 当成测试命令。

报告 verifier 质量时同时报告：

- parser 命中率；
- correct/incorrect/no_answer 数量；
- 假阳率（错误被判对）；
- 假阴率（正确被判错）；
- 人工抽查样本和原始输出。

---

## 6. MiniMind 改造实现

### 6.1 新增 `dataset/math_rl_dataset.py`

核心要求：数据没有占位 assistant，因此不要像 `RLAIFDataset` 一样无条件切 `conversations[:-1]`。

```python
import json
import random
from torch.utils.data import Dataset

class MathRLDataset(Dataset):
    def __init__(self, path, tokenizer, thinking_ratio=1.0, max_length=768):
        self.samples = [json.loads(line) for line in open(path, encoding="utf-8")]
        self.tokenizer = tokenizer
        self.thinking_ratio = thinking_ratio
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = sample["conversations"]
        use_thinking = random.random() < self.thinking_ratio
        prompt = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=use_thinking,
        )
        return {"prompt": prompt, "gt": str(sample["gt"])}
```

这里的 `max_length` 只是接口兼容字段；真正的 prompt 截断仍由训练循环里的 tokenizer 完成，建议在 `__getitem__` 中额外记录超长样本数，避免静默截断。第一版将 `thinking_ratio=1.0` 固定，避免训练中 prompt 模板随机变化。之后再将 `0.0` 作为独立消融，不要混在主实验里。

当前 `SFTDataset` 声明了嵌套 `Features`。如果你的 `math_sft_train.jsonl` 只包含 `role/content` 两个键而 `datasets` 报缺字段错误，就在清洗器中给每条消息补空的 `reasoning_content/tools/tool_calls`，或把 `SFTDataset` 的显式 `Features` 改为从首行推断；不要在训练前悄悄丢掉字段。

### 6.2 `train_grpo.py` 参数改造

三条项目实验臂（`rule`、`rm`、`mix`）都使用同一个 `MathRLDataset` 和同一个
`math_rlvr_train.jsonl`，这样 prompt、gt、随机顺序完全一致。原来的 `RLAIFDataset`
只保留给“官方 rlaif 回归护栏”，不能拿它读取 user-only 的数学文件。

新增参数：

```python
parser.add_argument("--reward_mode", choices=["rule", "rm", "mix"], default="rule")
parser.add_argument("--reward_marker", default="答案是")
parser.add_argument("--verdict_tier", choices=["binary", "tiered"], default="binary")
parser.add_argument("--dynamic_sampling", type=int, default=0)
parser.add_argument("--len_penalty_coef", type=float, default=0.0)
parser.add_argument("--length_cap", type=int, default=300)
parser.add_argument("--mix_alpha", type=float, default=0.5)
parser.add_argument("--mix_beta", type=float, default=0.5)
parser.add_argument("--run_tag", default="unnamed")
parser.add_argument("--log_dir", default="../logs")
parser.add_argument("--rlvr_data_path", default="../dataset/math_rlvr_train.jsonl")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--temperature", type=float, default=0.8)
```

同时把现有 `setup_seed(42 + rank)` 改为基于 `args.seed`，并把 epoch 内用于 shuffle 的
`setup_seed(42 + epoch)`、`DistributedSampler.set_epoch(...)` 和 DataLoader worker seed
都纳入同一套规则；否则 `--seed` 只控制初始化，不能复现数据顺序。推荐约定：

```python
rank = dist.get_rank() if dist.is_initialized() else 0
setup_seed(args.seed + rank)
# 在 train_ds 创建之后构造 sampler，显式传入同一个 base seed。
train_sampler = (DistributedSampler(train_ds, seed=args.seed)
                 if dist.is_initialized() else None)
for epoch in range(start_epoch, args.epochs):
    setup_seed(args.seed + epoch)
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
```

DataLoader 的 `worker_init_fn` 也用 `args.seed + epoch * 1000 + worker_id` 派生；第一轮
本地 smoke 直接使用 `num_workers=0`，避免多进程随机源遮蔽问题。

把 rollout 中硬编码的 `temperature=0.8` 改为 `args.temperature`。正式 run 一律显式写：

```bash
--loss_type grpo
--seed 42
--temperature 0.8
```

主函数顶部同时加入 `from dataset.math_rl_dataset import MathRLDataset`，数据入口至少要改成下面的形状（不要只加 argparse 而忘了接线）：

```python
if args.reward_mode in ("rule", "rm", "mix"):
    data_path = os.path.abspath(args.rlvr_data_path)
    train_ds = MathRLDataset(data_path, tokenizer,
                             thinking_ratio=args.thinking_ratio,
                             max_length=args.max_seq_len)
    assert len(train_ds) > 0
    assert all("gt" in row for row in train_ds.samples)
else:
    train_ds = RLAIFDataset(args.data_path, tokenizer,
                            max_length=lm_config.max_seq_len,
                            thinking_ratio=args.thinking_ratio)
```

随后在 `grpo_train_epoch` 的每个 step 取得 `gt_batch = batch["gt"]`，断言
`len(gt_batch) == len(prompts)`，再执行
`gt_for_rollouts = [gt for gt in gt_batch for _ in range(G)]`（等价于按 prompt 顺序
`repeat_interleave(G)`），并断言 `len(gt_for_rollouts) == len(completions)`。
不要把未经展开的 `B` 个 gt 直接和 `B*G` 个 completion zip。RM-only 项目对照也必须走 `MathRLDataset`；只有单独做官方 RLAIF
回归时才使用 `RLAIFDataset`，并明确标注它没有 GT accuracy。

推荐把 reward 壳子的调用写成显式参数，便于单测错位：

```python
gt_batch = batch["gt"]
assert len(gt_batch) == len(prompts)
gt_for_rollouts = [gt for gt in gt_batch for _ in range(args.num_generations)]
assert len(gt_for_rollouts) == len(completions)
rewards, verdict_stats = calculate_rewards(
    prompts=prompts,
    responses=completions,
    gt_for_rollouts=gt_for_rollouts,
    reward_model=reward_model,
)
```

### 6.3 条件加载 RM

将当前无条件 RM 初始化改成：

```python
reward_model = None
if args.reward_mode in ("rm", "mix"):
    rm_path = os.path.abspath(args.reward_model_path)
    if not os.path.isdir(rm_path):
        raise FileNotFoundError(f"reward model not found: {rm_path}")
    reward_model = LMForRewardModel(
        rm_path, device=args.device, dtype=torch.float16
    )
```

这样 `rule` 模式可以在 MPS/CPU smoke test 运行，也能量化 verifier-only 相比 RM 的显存和耗时差异。

### 6.4 奖励函数拆分

把原来的 `calculate_rewards` 拆成三个逻辑单元：

```text
rule_rewards(completions, gt_batch) -> reward, verdict_stats
official_rm_rewards(prompts, completions, reward_model) -> reward, rm_stats
pure_rm_rewards(prompts, completions, reward_model) -> reward, rm_stats
mix_rewards(rule_reward, pure_rm_reward) -> reward
```

薄壳函数的签名应显式包含 `gt_for_rollouts`，不要依赖全局变量：

```python
def calculate_rewards(prompts, responses, gt_for_rollouts=None, reward_model=None):
    # responses 与 gt_for_rollouts 都是 B*G；rule/mix 分支必须做长度断言。
    ...
```

重要规则：

1. `rule` 模式只用 GT verifier，不把原长度/think 启发式悄悄混进来；格式率只作为统计指标。
2. `rm` 模式在数学数据上复用官方启发式分 + RM 的计算逻辑，作为行为对照；真正的
   “改造前后逐位回归”另用官方 `rlaif.jsonl` 和 `RLAIFDataset`，并明确标注它包含
   官方启发式分和 RM。
3. `mix` 模式使用**纯 RM score**（只搬运原函数中的 `get_score` 项，去掉长度、think、重复启发式），避免混合臂同时改变三种变量；rule 和纯 RM 在组内分别标准化后再按显式的 `mix_alpha`/`mix_beta` 加权。

组内标准化示意：

```python
def normalize_group(x, eps=1e-6):
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, unbiased=False, keepdim=True)
    return (x - mean) / (std + eps)

rule_z = normalize_group(rule_reward.view(B, G)).view(-1)
rm_z = normalize_group(pure_rm_reward.view(B, G)).view(-1)
mixed = args.mix_alpha * rule_z + args.mix_beta * rm_z
```

如果某个分量组内全相同，标准化结果设为零并记录 `component_degenerate_groups`，不要产生 NaN。RM-only 和 mix 的绝对 reward 不直接比较，只比较正确率、格式率、长度、KL 和熵等无量纲指标。

### 6.5 长度惩罚消融

长度惩罚必须在组内 advantage 计算之前加入。先沿用训练主循环的 EOS 扫描逻辑，构造真实的 `completion_mask`；不要直接使用 `rollout_result.completion_mask`，因为 torch rollout 后端返回的那一份只是“全长占位 mask”。然后：

```python
lengths = completion_mask.sum(dim=1).float()
if args.len_penalty_coef > 0:
    excess = (lengths - args.length_cap).clamp(min=0)
    rewards = rewards - args.len_penalty_coef * excess / args.length_cap
```

如果当前代码在 reward 计算时还没有构造 `completion_mask`，就把 EOS 扫描代码前移到 reward 之前，或先计算 `lengths`、再计算 reward；不要用 decode 后字符串长度替代 token mask。这不是“让回答越短越好”，只惩罚超过预注册上限的 completion，必须同时报告正确率和长度效率，防止模型靠极短错误答案获得表面收益。

### 6.6 动态采样/退化组过滤

在每个 prompt 的 `G` 个样本中：

- 全部 `correct`：没有相对任务信号；
- 全部 `incorrect/no_answer`：同样没有相对任务信号；
- 至少一个正确、至少一个错误：有可学习的组内差异。

不要只用 `std > 0` 作为唯一语义判断，因为不同状态可能碰巧获得相同数值。保留：

```python
valid_group = has_correct & has_non_correct
```

如果开启 `--dynamic_sampling 1`，loss 聚合只使用 valid group，并记录：

```text
total_groups
valid_groups
all_correct_groups
all_wrong_groups
filtered_fraction
```

建议把过滤 mask 作用在 advantage 驱动的 policy-gradient 项，而不是悄悄改变 KL
正则项；这样可以区分“没有相对任务信号”和“没有 reference 约束”。实现时先把当前
合并的 `per_token_loss` 拆成两项：

```python
pg_token_loss = -torch.min(per_token_loss1, per_token_loss2)  # 或对应 CISPO 的 pg 项
kl_token_loss = args.beta * per_token_kl
per_sample_pg = (pg_token_loss * completion_mask).sum(1) / completion_mask.sum(1).clamp(min=1)
per_sample_kl = (kl_token_loss * completion_mask).sum(1) / completion_mask.sum(1).clamp(min=1)
valid_sample = valid_group.repeat_interleave(args.num_generations)
if args.dynamic_sampling:
    pg_loss = per_sample_pg[valid_sample].mean() if valid_sample.any() else per_sample_pg.sum() * 0.0
else:
    pg_loss = per_sample_pg.mean()
policy_loss = pg_loss + per_sample_kl.mean()
```

若决定连 KL 也过滤，必须作为单独的预注册消融并记录配置；不要把两种语义混在同一
个 `dynamic_sampling` 开关里。

小 batch 下过滤后可能没有有效组。遇到一个 step 的 `valid_groups == 0` 时跳过 policy-gradient 更新，但仍记录日志；不要用随机噪声“补” advantage。

### 6.7 ACR 指标

ACR 不写成“全对/全错组无梯度”这一绝对句式。严格说，advantage 驱动的 policy-gradient 部分为零，KL 项仍可能回拉 reference。建议记录两条曲线：

```python
import numpy as np

status = np.asarray(verdict_status).reshape(B, G)
all_correct = (status == "correct").all(axis=1)
all_wrong = np.isin(status, ["incorrect", "no_answer"]).all(axis=1)
acr_correct = all_correct.mean()
acr_wrong = all_wrong.mean()
```

解释顺序：

- `all_correct` 高：该难度可能已经被模型掌握；
- `all_wrong` 高：可能是模型容量、数据难度或冷启动问题；
- 对错混合但 reward 分辨率低：才更像奖励设计问题。

### 6.8 本地 CSV 和原始样本

每个 run 至少落盘两类文件：

```text
logs/{run_tag}_metrics.csv
logs/{run_tag}_samples.jsonl
```

CSV 推荐列：

```text
timestamp,step,seed,avg_reward,format_rate,correct_rate,
no_answer_rate,avg_completion_tokens,median_completion_tokens,
kl_ref,entropy,adv_std,policy_loss,valid_groups,
all_correct_groups,all_wrong_groups,filtered_fraction
```

`samples.jsonl` 每条包含：

```json
{"step":10,"prompt":"...","completion":"...","gt":"391",
 "reward":1.0,"status":"correct","extracted":"391",
 "difficulty":"easy","seed":42}
```

这份原始样本是后续人工抽查和面试 debug 的主要证据。

---

## 7. 冷启动 SFT

### 7.1 冷启动目标

SFT 不是为了直接提高数学榜分，而是教会模型：

- 遵守 think/answer 模板；
- 在最后一行输出 marker；
- 不在答案行列出多个候选数字；
- 为后续 verifier 提供稳定输入。

### 7.2 MiniMind 命令

训练脚本要求从 `trainer/` 启动。先确保官方 `full_sft_768.pth` 已放到脚本预期的 `../out/`。

```bash
cd trainer
python train_full_sft.py \
  --data_path ../dataset/math_sft_train.jsonl \
  --from_weight full_sft \
  --save_weight math_cold_seed42 \
  --epochs 2 \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --max_seq_len 512 \
  --num_workers 0 \
  --save_interval 1000 \
  --log_interval 20
```

MPS smoke 只验证代码，不产生效果结论：

```bash
cd trainer
python train_full_sft.py \
  --data_path ../dataset/math_sft_train.jsonl \
  --from_weight none \
  --save_weight smoke_sft \
  --epochs 1 \
  --batch_size 2 \
  --max_seq_len 256 \
  --num_workers 0 \
  --device mps \
  --save_interval 99999
```

### 7.3 SFT 退出标准

从 held-out 随机抽 20 条，检查：

- 是否输出 `</think>`；
- 是否命中 marker；
- marker 后是否只有一个答案行；
- verifier 是否能抽取；
- 答案正确率是多少。

如果格式率低于 80%，先修数据模板和 SFT label mask，不要直接开始 RL。MiniMind 的 SFT label 依赖 assistant 字符串锚点，模板渲染后必须打印一条样本的 token/mask 做人工核对。

---

## 8. MiniMind RLVR 主实验

本节是 **I1--I5 实现完成后的命令**。开始前必须通过：

```bash
python scripts/clean_math_data.py --help
python scripts/eval_heldout.py --help
(cd trainer && python train_grpo.py --help)
```

若任一命令提示文件不存在或参数不存在，回到第 0.5 节继续实现，不要修改命令来绕过
检查。

### 8.1 冒烟命令

先生成极小合成数据：

```bash
python scripts/clean_math_data.py \
  --synthetic-only \
  --train-size 30 \
  --heldout-size 10 \
  --out-dir dataset/
```

然后运行 rule-only GRPO。`--from_weight` 必须指向刚生成的 SFT 权重前缀：

```bash
cd trainer
python train_grpo.py \
  --reward_mode rule \
  --rlvr_data_path ../dataset/math_rlvr_train.jsonl \
  --from_weight smoke_sft \
  --save_weight smoke_rule \
  --run_tag smoke_rule \
  --loss_type grpo \
  --seed 42 \
  --temperature 0.8 \
  --batch_size 1 \
  --num_generations 4 \
  --max_seq_len 256 \
  --max_gen_len 64 \
  --epochs 1 \
  --num_workers 0 \
  --device mps \
  --save_interval 99999 \
  --log_interval 1 \
  --debug_mode \
  --debug_interval 1
```

退出标准：至少 2 个 step 完成，`logs/smoke_rule_metrics.csv` 有记录，`rule` 模式没有加载 RM，debug JSONL 有 prompt/completion/gt/status。

### 8.2 正式三臂

先跑 rule-only，再跑 RM-only，最后跑 calibrated mix。每个 run 使用相同 train/held-out 数据、相同 seed、相同 rollout 温度和最大长度。

**A. verifier-only**

```bash
cd trainer
python train_grpo.py \
  --reward_mode rule \
  --rlvr_data_path ../dataset/math_rlvr_train.jsonl \
  --from_weight math_cold_seed42 \
  --save_weight mini_rule_seed42 \
  --run_tag mini_rule_seed42 \
  --loss_type grpo \
  --seed 42 \
  --temperature 0.8 \
  --batch_size 2 \
  --num_generations 6 \
  --learning_rate 3e-7 \
  --beta 0.1 \
  --max_seq_len 512 \
  --max_gen_len 512 \
  --epochs 1 \
  --num_workers 0 \
  --log_interval 5 \
  --save_interval 50 \
  --debug_mode \
  --debug_interval 50
```

**B. RM-only**

```bash
cd trainer
python train_grpo.py \
  --reward_mode rm \
  --rlvr_data_path ../dataset/math_rlvr_train.jsonl \
  --from_weight math_cold_seed42 \
  --save_weight mini_rm_seed42 \
  --run_tag mini_rm_seed42 \
  --reward_model_path /absolute/path/to/internlm2-1_8b-reward \
  --loss_type grpo \
  --seed 42 \
  --temperature 0.8 \
  --batch_size 2 \
  --num_generations 6 \
  --learning_rate 3e-7 \
  --beta 0.1 \
  --max_seq_len 512 \
  --max_gen_len 512 \
  --epochs 1 \
  --num_workers 0 \
  --log_interval 5 \
  --save_interval 50
```

RM-only 在数学数据上没有 GT reward，只能作为“官方 RLAIF reward 行为”的对照。评估时仍用外部 verifier 计算正确率，但不要把 RM reward 当作数学准确率。

**C. calibrated mix**

```bash
cd trainer
python train_grpo.py \
  --reward_mode mix \
  --rlvr_data_path ../dataset/math_rlvr_train.jsonl \
  --from_weight math_cold_seed42 \
  --save_weight mini_mix_seed42 \
  --run_tag mini_mix_seed42 \
  --reward_model_path /absolute/path/to/internlm2-1_8b-reward \
  --mix_alpha 0.5 \
  --mix_beta 0.5 \
  --loss_type grpo \
  --seed 42 \
  --temperature 0.8 \
  --batch_size 2 \
  --num_generations 6 \
  --learning_rate 3e-7 \
  --beta 0.1 \
  --max_seq_len 512 \
  --max_gen_len 512 \
  --epochs 1 \
  --num_workers 0 \
  --log_interval 5 \
  --save_interval 50
```

### 8.3 消融顺序

资源有限时按以下顺序：

1. rule-only baseline；
2. rule + `--dynamic_sampling 1`；
3. rule + length penalty；
4. RM-only；
5. calibrated mix；
6. 第二个 seed；
7. 只有余力时尝试 REFT/Dr.GRPO/OPD 支线。

不要把动态采样和长度惩罚同时打开，否则无法解释贡献。每次只改一个因素。

### 8.4 预注册表

在正式训练前把表格写进 `configs/experiment_registry.csv`：

```text
run_tag,model,reward_mode,loss_type,seed,G,max_gen_len,beta,
dynamic_sampling,len_penalty_coef,alpha,beta_mix,hypothesis
```

示例假设：

```text
mini_rule_seed42,MiniMind-64M,rule,grpo,42,6,512,0.1,0,0,0.5,0.5,rule signal baseline
mini_rule_dynamic_seed42,MiniMind-64M,rule,grpo,42,6,512,0.1,1,0,0.5,0.5,remove degenerate groups
mini_rule_len_seed42,MiniMind-64M,rule,grpo,42,6,512,0.1,0,0.05,0.5,0.5,reduce overlong outputs
```

---

## 9. Qwen + TRL/Open-R1 主轨

### 9.1 为什么先用 TRL/Open-R1

`Open-R1` 已提供 SFT、GRPO、数据生成、代码执行奖励和评估入口；TRL 生态分别提供
`GRPOTrainer`、`DPOTrainer` 和 `RewardTrainer`，并可与 vLLM 集成。[Open-R1 README](https://github.com/huggingface/open-r1) · [TRL 文档](https://huggingface.co/docs/trl/grpo_trainer)

这条路线的价值不是“调一个 Trainer”，而是把你在 MiniMind 中写好的：

- 数据 manifest；
- verifier 单测；
- reward/format/length/accuracy 分离指标；
- ACR 和 pass@k 评估；

迁移到更真实的基座和 rollout 栈。

### 9.2 安装和官方 smoke

不要把 MiniMind 的 `requirements.txt` 和 Open-R1/TRL 主环境混装，也不要盲目追
`main`。仓库当前 `trl==0.13.0` 不提供本文所需的 `GRPOTrainer/GRPOConfig`；为
Open-R1 单独建 Linux/CUDA 环境，先选择一个 tag 或 commit，安装后立即
把 commit、Python/CUDA/PyTorch/vLLM/TRL 版本写入 `runs/open-r1-environment.txt`：

```bash
git clone https://github.com/huggingface/open-r1.git external/open-r1
cd external/open-r1
OPEN_R1_REF="<tested-tag-or-commit>"  # 执行前替换为真实 tag/commit
git fetch --tags origin
git cat-file -e "${OPEN_R1_REF}^{commit}" || { echo "invalid Open-R1 ref"; exit 2; }
git checkout "${OPEN_R1_REF}"
git rev-parse HEAD | tee ../../runs/open-r1-commit.txt
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python - <<'PY'
import trl
from trl import GRPOTrainer, GRPOConfig
print("trl:", trl.__version__)
print("GRPO import: PASS")
PY
python -m pip freeze > ../../runs/open-r1-environment.txt
accelerate config
```

`<tested-tag-or-commit>` 不能留在正式报告里；它是执行前必须填入并锁定的变量。若
import probe 失败，回退到包含 `GRPOTrainer` 的已发布 TRL 版本或 Open-R1 自带锁定版本，
并把最终版本写进 environment 文件；不要改 MiniMind 的 `trl==0.13.0` 来“碰运气”。
Open-R1 的依赖会随主分支变化，MiniMind 当前 `transformers/trl` 版本与 Open-R1
不一定兼容，分环境是可复现性的硬要求。

官方 README 的 GRPO 示例使用类似命令：

```bash
accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml \
  src/open_r1/grpo.py \
  --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml \
  --vllm_mode colocate
```

实际 config 文件名随 commit 变化，运行前用：

```bash
find recipes -iname '*grpo*' -o -iname '*1.5b*'
```

第一轮只做 10-20 step smoke，先证明：模型加载、chat template、vLLM 生成、reward callback、checkpoint 和日志都工作。

### 9.2.1 Qwen 模板迁移约定

MiniMind 的模板参数 `open_thinking` 不能直接传给 Qwen2.5；两者的消息字段、思考开关
和 assistant 起始串不是同一个接口。Qwen 主轨先选定一种并写入 manifest：

1. **稳定优先（推荐第一轮）**：不强制思考标签。user prompt 只要求最后一行输出
   `答案是 数字`，`accuracy_reward` 仍使用同一 verifier，`format_reward` 只检查 marker，
   不把 `</think>` 当成必需条件。
2. **思考格式实验**：确认所选 Qwen tokenizer 的 `chat_template` 支持的参数名（常见是
   `chat_template_kwargs` 或模型自定义字段），用一条样本打印渲染结果，确认确实出现
   成对的 `<think>...</think>`；只有确认后才把 `require_think=true` 写入 config。

迁移前必须做两条单测：同一条 user message 在 MiniMind 和 Qwen 各渲染一次；分别用
字符串 completion 和 assistant message-list 调用 reward callback。把模板版本、渲染文本
样例和 `require_think` 值保存到 `runs/{run_tag}/`。不要为了让 format rate 非零而在
verifier 中自动补 `</think>`。

### 9.3 自定义数学 reward

将 `math_verifier.py` 放到 Open-R1 fork 可导入的位置。先统一约定数据列名为
`ground_truth`；TRL 对 conversational 数据可能把 completion 传成 message list，
因此 reward callback 必须兼容字符串和消息列表，不能直接假设 `text` 有 `.strip()`：

```python
def completion_text(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("content", ""))
    if isinstance(item, list):
        # conversational completion 通常只有一个 assistant message。
        for message in reversed(item):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content", ""))
        return ""
    return ""

def accuracy_reward(completions, ground_truth=None, gt=None, **kwargs):
    golds = ground_truth if ground_truth is not None else gt
    if golds is None:
        golds = kwargs.get("ground_truth")
    if golds is None:
        raise KeyError("ground_truth column is required")
    if isinstance(golds, (str, int, float)):
        golds = [golds] * len(completions)
    rewards = []
    for completion, gold in zip(completions, golds):
        result = verdict(completion_text(completion), str(gold), marker="答案是")
        rewards.append(reward_tier(result, mode="binary"))
    return rewards

def format_reward(completions, require_think=False, **kwargs):
    return [
        1.0 if ("答案是" in completion_text(item)
                and (not require_think or "</think>" in completion_text(item))) else 0.0
        for item in completions
    ]
```

第一版建议只用 `accuracy_reward` 作为主 reward，`format_reward` 仅作为日志或独立消融；
若 Qwen 模板没有明确生成 think 标签，调用时保持 `require_think=False`。否则你无法判断
模型是在学格式还是学答案。

若选择 TRL 的纯字符串数据格式，也可以在数据转换时把 prompt 保持为渲染后的字符串；
但必须在转换 manifest 中写明格式，并用一条字符串 completion 和一条 message-list
completion 各跑一次 callback 单测。

TRL 配置中至少固定：

```text
model: Qwen2.5-1.5B-Instruct
seed: 42
num_generations: 4 or 8
max_prompt_length: 512
max_completion_length: 512
temperature: fixed
gradient_checkpointing: true
logging_steps: 1
save_strategy: steps
```

如果使用 LoRA，明确在报告中写“参数高效微调”，不要写成全参训练。若使用 TRL 新版本的 `scale_rewards`/`loss_type` 等选项，先查看当前文档和 config 的合法值，记录具体版本。

### 9.4 何时迁移到 verl

满足以下条件再迁移：

- MiniMind 和 TRL 版本的 verifier 结果一致；
- 至少一个 Qwen rule-only run 已稳定；
- 你能解释 rollout/train logprob、policy lag、FSDP 和 vLLM 的内存分工。

`verl` 适合展示更强的系统能力：它把 FSDP/Megatron、vLLM/SGLang、Ray 和 RL dataflow 组合起来。[verl README](https://github.com/volcengine/verl)

迁移原则：复制官方 `examples/grpo_trainer` 中与目标模型最接近的脚本，只替换：

1. train/validation 文件；
2. reward manager；
3. 采样长度、group size 和日志目录；
4. checkpoint/评估命令。

不要一开始重写 verl controller。面试价值来自你能定位一处真实问题（例如 reward batch、rollout policy version 或显存配置），而不是改动文件数。

### 9.5 verl 的版本、数据和 reward 接口

`verl` 版本变化很快。本文校验日观察到的 current-main ref 是
`89dad2d7a82dbd0e6740788fd755ecb09613c72f`，但执行前必须在网络可用的环境重新验证；它
要求的 CUDA、PyTorch、vLLM 和 Python 版本与旧 release 不同。实际执行时二选一并写进
`environment.txt`：

```bash
git clone https://github.com/volcengine/verl.git external/verl
cd external/verl
git fetch --tags origin
VERL_REF=89dad2d7a82dbd0e6740788fd755ecb09613c72f  # 或已验证存在的 release tag
git cat-file -e "${VERL_REF}^{commit}" || { echo "invalid verl ref: ${VERL_REF}"; exit 2; }
git checkout "${VERL_REF}"
git rev-parse HEAD | tee ../../runs/verl-commit.txt
uv run --frozen --all-packages --extra vllm --extra fsdp python -c \
  "import torch, verl; print(torch.__version__, verl.__file__)"
```

current-main 的安装说明见 [verl 安装文档](https://verl.readthedocs.io/en/latest/start/install.html)。
如果 `git cat-file` 检查失败，改用 `git ls-remote --tags origin` 选一个实际存在的 tag，
不要跳过检查或把未解析的占位 ref 写进报告。
不要把 current-main 的锁文件、`v0.7.1` 的依赖和另一台机器的 CUDA wheel 混用；如果采用
`v0.7.1`，把 tag、容器/镜像和完整 `pip freeze` 一起保存。MiniMind 所在的 Mac/MPS
环境不能运行这条 GPU 栈。

不要把“24GB 能跑 1.5B GRPO”写成保证。官方设备调优表中的 Qwen2.5-1.5B
GRPO-LoRA 最小示例是 1×H100；24GB 只作为本项目的 LoRA smoke 假设，先用 1--3 个
optimizer step 验证，OOM 时降 `G`、response 长度和 vLLM utilization，或退回
Qwen2.5-0.5B/MiniMind。相关约束见 [verl 设备调优表](https://verl.readthedocs.io/en/latest/perf/device_tuning.html)。

verl 的训练文件通常是 Parquet，每行至少包含：

```python
{
    "data_source": "custom_math",
    "prompt": [{"role": "user", "content": "..."}],
    "ability": "math",
    "reward_model": {"style": "rule", "ground_truth": "42"},
    "extra_info": {"split": "train", "index": 0}
}
```

`ground_truth` 是 reward metadata，不是另起一个 reward model。先把
`math_rlvr_train.jsonl`/`math_heldout.jsonl` 转成 Parquet，并在转换后重新计算
manifest hash；不要把 tokenizer 渲染后的整段 prompt 塞入 `prompt`，verl 会按消息列表
再次套 chat template。

自定义 verifier 通过四参数函数接入：

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    result = verdict(solution_str, str(ground_truth), marker="答案是")
    score = reward_tier(result, mode="binary")
    return {
        "score": score,
        "correct": result["correct"],
        "format": result["marker_hit"] and result["think_closed"],
        "parse_error": result["status"] in ("no_answer", "invalid_gt"),
    }
```

返回 dict 时必须有 `score`；其余字段用于日志和后续 ACR/格式分析。但不同 `verl` tag
的 reward manager 可能只接受 float：在所选 commit 的一条样本上先做接口探针；若不支持
dict，就返回 `float(score)`，把 `correct/format/parse_error` 通过独立 logger 写出，不要
把整个训练 run 作为第一次兼容性测试。把下面两个
override 追加到所选 commit 对应的实际 `main_ppo` 启动命令；这不是跨版本完整命令：

```bash
python -m verl.trainer.main_ppo reward.custom_reward_function.path=/absolute/path/to/rewards/math_reward.py reward.custom_reward_function.name=compute_score
```

### 9.6 verl GRPO 配置顺序

先固定最小标准 GRPO，再逐项打开优化。核心字段的含义是：

```text
algorithm.adv_estimator=grpo
actor_rollout_ref.rollout.n=G
actor_rollout_ref.actor.ppo_mini_batch_size=<global trajectory count>
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl
algorithm.use_kl_in_reward=False
actor_rollout_ref.actor.loss_agg_mode=token-mean
```

`data.train_batch_size` 是全局 prompt 数，`rollout.n=G` 后总 trajectory 数是
`train_batch_size * G`；`ppo_mini_batch_size` 必须能整除这个总数。带
`_per_gpu` 的字段是每卡 micro batch，不能拿它直接替代 global batch。第一轮保持
同步 colocated rollout；等 verifier 结果稳定后，再单独比较 FSDP2、offload、async
或 separate rollout。每次只改一个系统变量。

24GB smoke 的起始约束（工程假设，不是官方保证）如下：

```text
train_batch_size=2, rollout.n=2
max_prompt_length=256, max_response_length=256
ppo_mini_batch_size=4, actor/ref/rollout micro batch=1
LoRA rank=16, alpha=16, gradient_checkpointing=true
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.max_num_seqs=8
actor_rollout_ref.rollout.gpu_memory_utilization=0.10
```

这些是 verl 配置键名的概念起点，不是跨版本可直接粘贴的完整 config；先用
`python -m verl.trainer.main_ppo --cfg job`（或该 tag 对应的 config 帮助）确认键名。
先跑 1--3 step 并保存峰值显存、tokens/s、reward/correct/format/parse_error；确认后按
`response length -> G -> batch -> vLLM utilization` 的顺序逐项增加。若 vLLM 初始化
就 OOM，可临时用 HF rollout 做功能 smoke，但不能用该模式比较吞吐。

正式多卡 run 再加入 `use_dynamic_bsz=True`、`ppo_max_token_len_per_gpu`、FSDP
offload 和更大的 response length。checkpoint 合并后用同一份 `eval_heldout.py` 做
greedy/pass@k，不能只看 verl 的训练 reward。verl 的 checkpoint 合并入口可参考
[`verl.model_merger`](https://verl.readthedocs.io/en/latest/examples/config.html)。

---

## 10. Held-out 评估实现

### 10.1 `scripts/eval_heldout.py` 输入输出

命令接口建议：

```bash
python scripts/eval_heldout.py \
  --model_path /absolute/path/to/checkpoint \
  --data_path dataset/math_heldout.jsonl \
  --output logs/mini_rule_seed42_eval.json \
  --num_samples 16 \
  --max_new_tokens 512 \
  --temperature 0.8 \
  --top_p 0.85 --top_k 50 \
  --seed 42 \
  --device cuda \
  --marker '答案是'
```

参数 `--hidden_size`、`--num_hidden_layers`、`--device`、`--max_new_tokens`、`--top_p`
和 `--top_k` 也要在脚本中显式声明，且默认值写入输出 JSON；MiniMind `.pth` 权重的
结构参数必须与训练时一致。

输出 JSON 至少包含：

```json
{
  "model": "...",
  "data_manifest_sha256": "...",
  "seed": 42,
  "temperature": 0.8,
  "greedy_accuracy": 0.0,
  "sampled_pass_at_1": 0.0,
  "pass_at_4": 0.0,
  "pass_at_8": 0.0,
  "pass_at_16": 0.0,
  "format_rate": 0.0,
  "no_answer_rate": 0.0,
  "mean_completion_tokens": 0.0,
  "median_completion_tokens": 0.0,
  "by_difficulty": {},
  "samples": []
}
```

### 10.1.1 MiniMind evaluator 的实现骨架

`eval_heldout.py` 放在 `scripts/`，固定从仓库根目录启动。它不是交互式
`eval_llm.py` 的改名版，而是一个可批处理、可落盘的 harness。核心顺序如下：

```python
# 伪代码：实现时补齐 batch padding、异常处理和 jsonl 逐条写出。
import json
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.math_verifier import verdict

tokenizer = AutoTokenizer.from_pretrained(repo / "model")
config = MiniMindConfig(hidden_size=args.hidden_size,
                        num_hidden_layers=args.num_hidden_layers)
model = MiniMindForCausalLM(config).to(args.device).eval()
state = torch.load(args.model_path, map_location=args.device)
if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
    state = state["model"]
model.load_state_dict(state, strict=True)

for sample in heldout:
    prompt = tokenizer.apply_chat_template(
        sample["conversations"], tokenize=False,
        add_generation_prompt=True, open_thinking=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

    greedy_ids = model.generate(
        **inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)
    sampled_ids = model.generate(
        **inputs, do_sample=True, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k,
        num_return_sequences=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)

    greedy_text = tokenizer.decode(
        greedy_ids[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True)
    sampled_texts = [tokenizer.decode(
        row[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for row in sampled_ids]
    greedy_v = verdict(greedy_text, sample["gt"], marker=args.marker)
    sampled_vs = [verdict(t, sample["gt"], marker=args.marker)
                  for t in sampled_texts]
    # 写出 greedy_v、sampled_vs、长度和 difficulty；聚合时计算 I_k。
```

实现时必须注意：`num_return_sequences` 是每个 prompt 的采样数，不是 batch size；
`do_sample=False` 的 greedy 输出单独计入 `greedy_accuracy`，不能把它当成随机
`sampled_pass_at_1`。在正式运行前先加一条 schema/权重检查：

```bash
python scripts/eval_heldout.py \
  --model_path out/smoke_rule_768.pth \
  --data_path dataset/math_heldout.jsonl \
  --output logs/smoke_rule_eval.json \
  --num_samples 1 --max_new_tokens 64 --temperature 0.8 \
  --seed 42 --device mps --marker '答案是'
```

这条命令的验收只是“能读权重、生成一条 JSON、字段齐全”；随机初始化或 10 条合成
题的数字不能写进效果结论。

### 10.2 pass@k 口径

第一版最简单、最可解释的定义：对每个问题独立采样恰好 `k` 条，令
`I_k = 1[至少一条被 verifier 判对]`，最后对问题求 `mean(I_k)`。`--num_samples 16`
时保存同一批样本的前 1、4、8、16 个前缀，分别计算 `pass@1/4/8/16`；不要为了
每个 `k` 重新采样一批不同的输出。若要报告 `pass@32`，将 `--num_samples` 提高到
32，并在报告中注明它使用的 held-out 子集和额外采样成本。

固定：

- sampling temperature；
- top-p/top-k；
- prompt template；
- seed 生成规则；
- 最大 completion 长度；
- verifier 版本。

greedy 不是 pass@1 的同义词。报告中分别写 `greedy_accuracy` 和 `sampled_pass_at_1`。

### 10.3 评估矩阵

至少评估：

| 权重 | greedy | pass@1 | pass@16 | format | avg len | easy/medium/hard |
|---|---:|---:|---:|---:|---:|---:|
| 官方/冷启动 SFT | | | | | | |
| rule GRPO | | | | | | |
| dynamic sampling | | | | | | |
| length penalty | | | | | | |
| RM-only | | | | | | |
| calibrated mix | | | | | | |

如果出现小 k 上升、大 k 下降，记录为“pass@k inversion”现象，并讨论 base coverage；不要把它包装成模型获得了全新推理能力。相关方法学背景见 [NeurIPS 2025 Oral 论文](https://arxiv.org/abs/2504.13837)。

### 10.4 统计和人工审计

- 关键臂至少两个 seed；
- 报告均值和标准差，资源允许时加 bootstrap 置信区间；
- 每个权重抽查至少 20 条原始 completion；
- 错误分类至少包括：无答案、格式错误、算术错误、冗余候选、长度烧满、重复循环、verifier 误判。

---

## 11. 实验分析模板

每次实验完成后，按同一模板写 `runs/{run_tag}/analysis.md`：

```markdown
# Run: <run_tag>

## 1. 目的
- 只改变了什么？
- 预注册假设是什么？

## 2. 环境
- commit:
- model:
- GPU:
- torch/transformers/trl/verl:
- command:

## 3. 训练动态
- reward:
- format rate:
- correctness:
- length:
- KL/entropy:
- ACR correct/wrong:

## 4. Held-out
- greedy:
- pass@1:
- pass@16:
- difficulty breakdown:

## 5. Error analysis
- 20 条样本类别统计
- 典型样本路径

## 6. 结论边界
- 支持/不支持哪个假设？
- 哪些结论不能从本 run 推出？
- 下一步最小实验是什么？
```

### 11.1 推荐图表

最小图集：

1. reward、format rate、correctness、completion length 随 step；
2. ACR correct/wrong 随 step；
3. 各 reward arm 的 held-out 表；
4. 按难度的正确率变化；
5. 长度分布直方图；
6. reward-eval gap 散点图。

所有图表在标题或 caption 中写清：模型、数据集、seed、采样数、温度和长度口径。

### 11.2 结果解释决策树

```text
reward 上升？
  ├─ 否：查 reward 范围、verifier、学习率、rollout、NaN
  └─ 是
      ├─ correctness 也上升：真实任务信号候选，检查 held-out/seed
      └─ correctness 不上升
          ├─ format 上升：格式吸收 reward
          ├─ length 上升：长度偏置/过长 hack
          ├─ ACR_wrong 高：容量/难度/冷启动
          └─ ACR 混合组高但不涨：奖励分辨率或优化设置问题
```

---

## 12. 常见故障排查

| 症状 | 首先检查 | 修复顺序 |
|---|---|---|
| `FileNotFoundError` 权重/RM | 当前 cwd 和绝对路径 | 固定 `cd trainer`，打印 `Path.resolve()` |
| rule 模式仍加载 RM | reward mode 分支 | 在初始化处断言 `reward_model is None` |
| MPS 报 fp16/RM 算子错误 | 是否误跑 RM | MPS 只跑 rule-only，`num_workers=0` |
| SFT loss 不降/全是 -100 | assistant mask/template | 打印一条渲染文本和 label mask |
| completion 总是烧满上限 | EOS/chat template | 降低 `max_gen_len`，检查 eos id 和 prompt |
| reward 全相同 | verifier 抽取、数据 gt、模型能力 | 保存原文，先离线跑 verifier，不要马上调 lr |
| reward 上升但正确率不升 | 格式分/长度/重复 | 拆分 reward，画四曲线和 ACR |
| NaN | group std=0、混合 reward 未标准化 | `eps`、degenerate group 置零、检查 fp16 |
| OOM | G、completion 长度、vLLM utilization | `G: 8 -> 4`，长度 `512 -> 256`，再降 batch |
| RM 极慢 | 每样本串行前向 | 主结论先用 rule；RM 对照记录耗时，后续 batch 化 |
| SGLang 结果不稳定 | 权重同步间隔和 logprob 长度 | 主实验先固定 torch；再加 policy version 校验 |
| pass@16 太慢 | held-out 数量 × 采样数 | 先 100-200 条子集，明确报告口径 |

### 12.1 失败时的降级顺序

```text
verl 环境失败
  -> TRL/Open-R1 1.5B LoRA
  -> Qwen2.5-0.5B 小规模
  -> MiniMind rule-only 完整诊断
  -> 仅保留 verifier + 评估 + 代码审计
```

任何降级都要在 README 写明，不要用低配 smoke 的数字冒充主实验。

---

## 13. 复现和归档清单

每个 run 目录必须有：

```text
runs/{run_tag}/
├── command.txt
├── config.yaml
├── environment.txt
├── data_manifest.json
├── metrics.csv
├── eval.json
├── samples.jsonl
├── analysis.md
└── checkpoints.txt
```

`command.txt` 由 shell 保存：

```bash
set -o pipefail
your_command 2>&1 | tee runs/<run_tag>/train.log
```

归档前检查：

```bash
sha256sum dataset/math_*.jsonl > dataset/manifests/data.sha256
python -m compileall model trainer dataset scripts
git diff --stat
git status --short
```

不要提交：

- checkpoint 二进制；
- RM 私有/大体积目录；
- API key、云平台 token；
- 未核验的实验数字；
- 只有截图而没有原始 CSV/JSONL 的图表。

---

## 14. 四周执行计划

### 第 1 周：代码和数据地基

- 阅读并画出 `train_grpo.py` 的 rollout -> reward -> advantage -> loss 链；
- 实现 `math_verifier.py` 和 200 条单测；
- 实现清洗脚本、manifest 和 synthetic 数据；
- MPS/CPU smoke；
- 退出标准：verifier 和数据切分通过，代码至少跑 2 step。

### 第 2 周：冷启动和 MiniMind rule baseline

- 下载/准备官方 MiniMind 权重；
- 跑数学 SFT 冷启动；
- 评估 20 条格式和答案；
- 跑 rule-only GRPO；
- 退出标准：有第一条完整曲线和 held-out JSON。

### 第 3 周：对照和消融

- RM-only 对照；
- dynamic sampling；
- length penalty；
- 第二 seed；
- 退出标准：至少一张对照表、ACR 曲线和 20 条错误分析。

### 第 4 周：Qwen 迁移和面试材料

- 用 TRL/Open-R1 跑 Qwen 1.5B smoke；
- 迁移同一 verifier 和 held-out harness；
- 资源允许时再迁移 verl；
- 写 README、实验分析、简历和 Q&A；
- 退出标准：能从空目录解释环境、输入输出、训练动态和限制。

若只有两周：砍掉 mix、第二 seed 和 Qwen 迁移，保留 verifier、rule baseline、一个消融、held-out 和诚实报告。

---

## 15. 面试输出包

### 15.1 简历占位版（完成实验后再填数字）

中文：

> 基于 MiniMind-64M 与 Qwen2.5-1.5B 构建数学 RLVR（GRPO）实验链路，将启发式/LLM judge 奖励替换为带三态判定和人工审计的规则 verifier；实现数据去重、难度分层、held-out greedy/pass@k 评估和 reward-format-accuracy-length 监控。设计 verifier、RM、calibrated mix 及退化组过滤/长度惩罚对照，分析 reward 上升与真实正确率之间的偏差，并如实记录模型规模限制和负结果。

英文：

> Built a reproducible math RLVR pipeline on MiniMind-64M and Qwen2.5-1.5B, replacing heuristic/LLM-judge rewards with a three-state, audited rule verifier. Added leakage-aware data curation, difficulty-stratified held-out evaluation, greedy/pass@k metrics, and reward-format-accuracy-length diagnostics. Compared verifier, RM, calibrated mixed rewards and degenerate-group/length ablations, reporting both gains and negative findings with explicit resource and capability limits.

只有在真实完成后，才把“Qwen2.5-1.5B”“pass@16”“多少 seed”等写进简历。

### 15.2 必答追问

1. 为什么 MiniMind 是 toy，你为什么还用它？
2. 为什么 rule verifier 比 RM 更适合数学 RLVR？
3. 你的 reward 和 accuracy 为什么可能背离？
4. GRPO 的 group mean/std 在全对组、全错组时发生什么？
5. ACR 统计的是什么，为什么不能简单说“无梯度”？
6. 为什么正式 run 要显式 `loss_type=grpo`？
7. 你如何避免 `1` 被错误地从 `21` 中 substring 命中？
8. verifier 假阳和假阴哪个更危险？如何测试？
9. pass@1 和 greedy 有什么区别？pass@16 成本如何控制？
10. Qwen/TRL/verl 的 rollout、policy logprob、reference logprob 如何对齐？
11. 为什么 mix reward 要先标准化？
12. 如果 reward 涨了但 held-out 不涨，你会先查什么？
13. 为什么不直接做 OPD、PRM 或 Agentic RL？
14. 你的结果能否证明“RL 学会了新推理能力”？

### 15.3 口头解释主线

用下面这段 60 秒开场：

> MiniMind 的价值是让我能逐行理解 rollout、组内 advantage、KL 和 mask，但它的 64M 规模不能支持泛化能力结论。于是我把项目问题收窄为 reward 是否真的代表学习：用外部 GT verifier 替换原来的格式分和 RM，建立 train/held-out 无泄漏数据，记录正确率、格式率、长度、KL、熵和退化组比例，再把同一套 verifier 迁移到 Qwen + TRL/verl。最终重点不是一条漂亮 reward 曲线，而是能解释 reward hacking、长度偏置、verifier 错误和小模型能力边界。

---

## 16. 前沿扩展（主线完成后才做）

### 16.1 OPD -> RLVR

如果已经有可靠教师模型和远程 GPU，可增加三组：

```text
SFT -> RLVR
SFT -> OPD
SFT -> OPD -> RLVR
```

要记录 teacher/student token logprob gap、训练成本、held-out accuracy、pass@k 和长度。近期 OPD 研究强调教师必须带来学生未见过的新能力，不能把同族弱教师当成有效蒸馏。[OPD 研究](https://arxiv.org/abs/2604.13016)

64M 主线不强行做 OPD：没有合适 teacher 时，结论会被 teacher 数据质量和 API 噪声混淆。

### 16.2 Agentic RL

MiniMind 已有 `train_agent.py`，但工具环境是静态 mock，不能直接包装成真实 Agent RL。若要扩展，先替换为可记录版本的 sandbox 环境，再定义：

- 工具调用合法性；
- 每轮 observation/action；
- 终局 reward；
- 工具失败和超时；
- trajectory-level credit assignment。

这应是第二个独立项目，不要和数学 RLVR 同一轮简历里堆成“全都做过”。

---

## 17. 红线

- 不写“复现 DeepSeek-R1”除非严格复现论文设置，而且这不是本 SOP 的目标；
- 不把仓库已有的 DPO/PPO/GRPO 代码写成你从零实现；
- 不把 MiniMind-64M 的格式变化写成通用推理能力提升；
- 不把 reward model 分数当数学准确率；
- 不在数据没有下载/验证时写数据规模和效果数字；
- 不把 ACR、pass@k inversion、长度偏置写成你首创；它们是已有方法学语境，你的贡献是复现、实现和应用到受控实验；
- 不隐藏负结果、OOM、verifier bug 或 policy lag；这些往往是最能被面试官追问的部分。

---

## 18. 最终检查表

### 代码

- [ ] `python -m compileall` 通过；
- [ ] verifier 无 unrestricted parser；
- [ ] rule 模式不加载 RM；
- [ ] mix 分量先标准化；
- [ ] group mask、completion mask、EOS 逻辑有测试；
- [ ] 每个 run 的权重前缀唯一。

### 数据

- [ ] train/held-out key 零重叠；
- [ ] `gt` 可解析；
- [ ] 50 条人工核对；
- [ ] source、版本、SHA256 已记录；
- [ ] 没有占位 assistant 污染 RL prompt。

### 实验

- [ ] SFT baseline；
- [ ] rule-only GRPO；
- [ ] 至少一个对照；
- [ ] 至少一个单因素消融；
- [ ] 至少一个第二 seed 或明确资源限制；
- [ ] greedy/pass@1/pass@16；
- [ ] reward/format/accuracy/length/ACR 曲线。

### 求职

- [ ] 简历数字均能回到 JSON/CSV；
- [ ] 能解释一个失败案例；
- [ ] 能说清 MiniMind 与 Qwen/TRL/verl 的边界；
- [ ] 能在 5 分钟内画出数据流和 loss 流；
- [ ] README 能让别人从空环境复现最小路径。

---

## 参考资料

- [MiniMind 仓库](https://github.com/jingyaogong/minimind)
- [Hugging Face Open-R1](https://github.com/huggingface/open-r1)
- [Hugging Face TRL](https://github.com/huggingface/trl)
- [verl](https://github.com/volcengine/verl)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [Dr.GRPO](https://arxiv.org/abs/2503.20783)
- [RLVR 能力边界与 pass@k](https://arxiv.org/abs/2504.13837)
- [OPD 机制研究](https://arxiv.org/abs/2604.13016)
