# 预注册实验表（SOP §8.4）

> 第一次上服务器：照着 [`docs/server_runbook.md`](../docs/server_runbook.md) 走，
> 里面包含传代码、装环境、下载权重、`scripts/server_preflight.py` 自检、
> SFT 冷启动/GRPO/评估/归档/回传的完整命令与故障速查。

`experiment_registry.csv` 是**在正式训练之前**写死的实验登记表。它的作用不是记录，而是约束：
跑之前先把「改什么变量、预期看到什么、什么结果会推翻它」写下来，跑完就不能再改。

配套工具：

```bash
python scripts/experiment_registry.py --check                       # 校验表格
python scripts/experiment_registry.py --list                        # 一览
python scripts/experiment_registry.py --print-command mini_rule_seed42
python scripts/experiment_registry.py --save-command mini_rule_seed42   # 写 runs/<tag>/command.txt
```

`--check` 会把每一列映射到 `train_grpo.py` 的真实 argparse flag 上做交叉检查，并检查
`run_tag` 唯一、`G >= 2`、`beta > 0` 等硬约束。所以「表格里写了 `dynamic_sampling=1`
但命令里忘了传」这类漂移会在跑之前就报错。

## 列语义

| 列 | 含义 |
|---|---|
| `run_tag` | 唯一标识，同时用作 `--run_tag` 和 `--save_weight` |
| `model` | 仅标注用的模型名 |
| `reward_mode` | `--reward_mode`，`rule`/`rm`/`mix`/`official` |
| `loss_type` | `--loss_type`，正式 run 一律 `grpo` |
| `seed` | `--seed`，控制初始化、数据顺序、rollout 采样 |
| `G` | `--num_generations`，每 prompt 采样数 |
| `max_gen_len` | `--max_gen_len` |
| **`beta`** | **KL 惩罚系数**（`--beta`），不是 mix 的权重 |
| `dynamic_sampling` | `--dynamic_sampling`，是否过滤全对/全错组 |
| `len_penalty_coef` | `--len_penalty_coef`，超过 `--length_cap` 的惩罚 |
| `alpha` | `--mix_alpha`，mix 模式下 rule 分量权重 |
| `beta_mix` | `--mix_beta`，mix 模式下 RM 分量权重 |
| `hypothesis` | 该 run 的预注册假设（一句话） |

⚠️ `beta` 和 `beta_mix` 是两个不同的东西：前者是 KL 系数，后者是 mix 里 RM 分量的权重。
列名沿用 SOP §8.4 的原始定义，不要合并。

## 未注册但固定不变的配置

这些值对所有实验臂相同，写在 `scripts/experiment_registry.py` 的 `FIXED_ARGS` 里，
由 `--print-command` 显式注入命令，保证各臂之间不会悄悄漂移：

```text
batch_size=2  learning_rate=3e-7  max_seq_len=512  epochs=1
num_workers=0 log_interval=5      save_interval=50 temperature=0.8
thinking_ratio=1.0  verdict_tier=binary  reward_marker=答案是
device=cuda:0
from_weight=math_cold_cot_seed42   rlvr_data_path=../dataset/math_rlvr_train.jsonl
```

`thinking_ratio=1.0` 是刻意固定的：prompt 模板在训练中随机变化会让「长度/格式变化」
这件事无法归因。

## 前置 run（不在登记表内）

登记表只放**实验臂**。下面两个是前置条件/护栏，不属于实验设计：

1. **SFT 冷启动**（§7.2），产出 `math_cold_cot_seed42`，所有 GRPO 臂都从它出发：
   ```bash
   cd trainer && python train_full_sft.py \
     --data_path ../dataset/math_sft_train.jsonl \
     --from_weight full_sft --save_weight math_cold_cot_seed42 \
     --epochs 2 --batch_size 16 --learning_rate 1e-5 --max_seq_len 512 \
     --num_workers 0 --save_interval 1000 --log_interval 20
   ```

   > 为什么 tag 里有 `cot`：pilot 阶段用的是 `math_cold_seed42`（CoT 是模板句，
   > 不含中间步骤），实测格式率 1.00 但 held-out 正确率 0/20。修好 CoT 之后换新 tag 重跑，
   > 旧 tag 的产物原样保留在 `runs/math_cold_seed42/`，作为「CoT 不可学」的对照证据。
   退出标准（§7.3）：抽 20 条 held-out，格式率 ≥ 80%（输出 `</think>`、命中 `答案是`、
   marker 后只有一个答案行）才能开始 RL。

2. **改造前后回归护栏**：官方 `rlaif.jsonl` + `RLAIFDataset` + 原始奖励，用来证明
   I4 的奖励拆分没有改变原有行为。它没有 GT accuracy，不进入结果表。
   ```bash
   cd trainer && python train_grpo.py --reward_mode official \
     --data_path ../dataset/rlaif.jsonl --thinking_ratio 0.9 \
     --reward_model_path /abs/path/to/internlm2-1_8b-reward \
     --save_weight smoke_official --run_tag smoke_official --device cuda:0
   ```

## 每个 run 的预注册预测与推翻条件

「推翻条件」比「预期」重要：**如果结果落进推翻区间，就必须如实写成负结果，不能改口径。**

### 1. `mini_rule_seed42` — rule 信号基线
- 变量：无（其余五个 run 都相对它比较）
- 预测：held-out `greedy_accuracy` / `pass@1` 高于 SFT 冷启动基线；训练 `correct_rate` 与
  `avg_reward` 同向上升
- 推翻：reward 上升但 `correct_rate` 与 held-out 都不动（reward hacking 或能力瓶颈）；
  或四个指标全不动（先查数据、verifier、吞吐、超参，不要先怪模型）
- 主指标：held-out `greedy_accuracy`，辅以 `pass@16`

### 2. `mini_rule_dynamic_seed42` — 过滤退化组
- 变量：`dynamic_sampling` 0 → 1
- 预测：同样 step 数下 `correct_rate` 上升更快（随机权重下退化组占比接近 100%，
  过滤后剩余样本的梯度更干净）；held-out 不低于基线太多
- 推翻：held-out 明显下降（有效样本被砍太多或把有用信号也过滤了）；
  或 `valid_groups` 长期为 0 导致几乎不更新
- 必报：`filtered_fraction` 曲线，不报这个数就无法解释

### 3. `mini_rule_len_seed42` — 长度惩罚
- 变量：`len_penalty_coef` 0 → 0.05（cap 300）
- 预测：`avg_completion_tokens` 下降，held-out accuracy 不下降
- 推翻：长度下降但 accuracy 同步下降 —— 这说明模型原本是靠「写长」换正确率，
  是**有价值的负结果**，不许只报长度下降

### 4. `mini_rm_seed42` — RM-only 对照
- 变量：奖励来源 rule → 官方启发式 + RM
- 预测：`avg_reward` 上升（启发式里含长度/格式项，很容易被刷），
  但 held-out accuracy **不高于** rule 基线
- 推翻：RM 臂 accuracy 显著高于 rule 臂 —— 要么 RM 提供了 verifier 没有的信号，
  要么 verifier 太弱，两种都要查
- 注意：RM 臂与 rule 臂的**绝对 reward 不可比**，只能比 accuracy / format / 长度 / KL / 熵

### 5. `mini_mix_seed42` — 标准化后的混合奖励
- 变量：reward = `0.5·rule_z + 0.5·rm_z`（两个分量各自组内标准化后再加权）
- 预测：accuracy 介于 rule 与 rm 之间或略高于 rm；reward 曲线形状更接近 RM 臂
- 推翻：mix 与 rm 臂曲线几乎重合 —— 说明 rule 分量被淹没或标准化没生效，
  此时必须检查 `component_degenerate_groups`（如果是 rule 分量在多数组里退化，
  mix 就等价于 RM 臂，这个结论要写清楚）

### 6. `mini_rule_seed7` — 第二 seed
- 变量：`seed` 42 → 7
- 预测：方向一致（正确率变化符号相同），幅度可以不同
- 推翻：**符号相反** —— 那么基线结论是噪声，任何「提升 X%」都不能写进简历
- 报告：两个 seed 的均值与范围；只有一个 seed 时必须在 README 里写明资源限制

## 跑完之后：归档（SOP §13）

训练命令用 `runs/<run_tag>/train.log` 落日志，然后一条命令把归档补齐：

```bash
# 1) 训练（命令由登记表生成，不要手敲）
bash -c "$(python scripts/experiment_registry.py --print-command mini_rule_seed42)"

# 2) 评估 held-out，结果直接写进归档目录
python scripts/eval_heldout.py --model_path out/mini_rule_seed42_768.pth \
  --output runs/mini_rule_seed42/eval.json --num_samples 16 --max_new_tokens 512

# 3) 归档 + 自检
python scripts/archive_run.py --run-tag mini_rule_seed42 --force
python scripts/archive_run.py --run-tag mini_rule_seed42 --check --strict
```

`scripts/archive_run.py` 生成 `runs/<run_tag>/` 下 SOP §13 要求的九个文件：

| 文件 | 来源 |
|---|---|
| `command.txt` | 由登记表渲染，含 `set -o pipefail` 与 `tee` |
| `config.yaml` | `logs/<run_tag>_config.json` 原样转换 |
| `environment.txt` | 平台/python/包版本/CUDA/git commit（本机无 git 时写明） |
| `data_manifest.json` | 数据集 manifest + 实际使用文件的 sha256 复核 |
| `metrics.csv` | `logs/<run_tag>_metrics.csv` **逐字节复制** |
| `eval.json` | held-out 评估输出（未评估时记进 `missing`，不伪造） |
| `samples.jsonl` | `logs/<run_tag>_samples.jsonl` **逐字节复制** |
| `analysis.md` | §11 模板，自动填「只改了什么/训练动态/held-out」，结论留 TODO |
| `checkpoints.txt` | 权重文件索引（路径/size/sha256），**权重本体不入库** |
| `manifest.json` | 额外：每个文件的 sha256、`status`（complete/partial）、缺失项 |

两条设计约束：**原始证据（metrics/samples/eval）永不改写**，缺失项写进 `manifest.json`
的 `missing` 而不是省略；`--check --strict` 会要求 `eval.json` 和 `train.log` 都在位、
且所有文件哈希与归档时一致 —— 所以「手动改过一个数字」会被当场发现。
`--all` 可以一次性归档 `logs/` 里所有已登记且跑过的 run。

## 跨 run 汇总与图表（SOP §10.3 / §10.4 / §11）

归档完之后（每跑完一个臂就可以跑一次）：

```bash
python scripts/summarize_runs.py                       # 全部登记臂 + SFT 冷启动
python scripts/summarize_runs.py --require-eval        # 还缺 held-out 就返回非 0
python scripts/summarize_runs.py --run-tag mini_rule_seed42 --no-figures
```

产出在 `reports/`：

| 文件 | 内容 |
|---|---|
| `summary.csv` | 每个 run 一行：登记表超参 + 训练动态首末值 + ACR + held-out 全套指标；缺失=空，不填 0 |
| `summary.md` | §10.3 评估矩阵、§10.4 seed 均值/范围、按难度表、§11.2 决策树逐 run 判定、图表 caption 全文 |
| `figures/fig1..fig6*.png` | §11.1 的六张最小图集，每张图底部带 caption（模型/数据/seed/采样数/温度/长度口径） |
| `manifest.json` | 收录了哪些 run、证据等级、缺失项、每个产物的 sha256 |

口径与纪律：

- 训练侧数字只来自 `runs/<tag>/metrics.csv`，held-out 只来自 `runs/<tag>/eval.json`；
  **拿不到就留空**，并在 `summary.md` 第 6 节列出来。
- `evidence` 列区分 `archived`（有 `manifest.json`）和 `logs-only`：未归档的行在矩阵里带 `*`，
  引用前先 `archive_run.py`。
- `reward_eval_gap` 是「训练 reward 涨幅 − held-out 涨幅」，两者单位不同，**只能用来排序**，
  不能当物理量写进结论。
- 决策树提示只是把 §11.2 的分支按数字走一遍：`reward 涨 + held-out 不涨` 会写成
  reward-eval gap，而不是能力提升。
- 只有登记表里的臂 + `math_cold_cot_seed42` 会进汇总（`--run-tag` 可显式加别的 tag），
  所以 `logs/` 里的 smoke 不会被误当成结果。

## 纪律

1. 跑完任何 run 之后，**不要修改已登记行的预测或推翻条件**。需要新假设就新增一行。
2. 每个 `run_tag` 只跑一次。MiniMind 按前缀复用 checkpoint 和 `_resume.pth`，
   同名重跑会悄悄接着上一次训练。
3. 结果落进推翻区间时，照原样写进 `runs/<run_tag>/analysis.md` 的「结论边界」一节。
4. 简历里的每个数字都必须能回到 `logs/<run_tag>_metrics.csv`、`<run_tag>_eval.json`
   或 `runs/<run_tag>/` 下的产物。
