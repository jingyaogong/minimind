# MiniMind 数学 RLVR(GRPO)+规则 Verifier —— 实现级审计与规格文档

- 角色：coding 子 agent 代码审计输出（供 main/reviewer/实现者使用）
- 基线文档：`docs/post-training-guide.md`（下文简称 guide，行号锚点以 guide 文件行号引用）
- 审计对象：仓库 `/Users/qi/实习/项目/minimind`（commit 未知，无 git 历史；以下行号均为本仓库当前文件实测行号）
- 结论性质：**实现级设计 + 风险清单，不写完整实现**。文中伪代码为最小补丁骨架，行号锚点全部经过 read 核实。
- 环境事实（任务给定，不再复测）：macOS / python3 / torch 2.12.1 / 无 CUDA / 有 MPS；`dataset/` 与 `out/` 无真实数据与权重；正式训练在用户 24GB 单卡（CUDA）。

---

## §1 现状核实表（guide 疑点备忘逐条核实）

> 判定图例：✅=属实　⚠️=部分属实/需修正　❌=不属实。锚点行号均为本次审计实测。

### 1.1 奖励函数与主循环（任务清单 #1、guide §2 表 / §4.2 / §4.4）

| # | guide 疑点/说法 | 判定 | 代码锚点与说明 |
|---|---|---|---|
| 1.1a | "GRPO/CISPO：格式启发式奖励 + 1.8B LLM judge + token 级 KL + 组内归一 advantage(group=6)" | ✅ | `trainer/train_grpo.py` L226 `--num_generations 6`；L228 `--loss_type default=cispo`；L132–133 token 级 KL（`per_token_kl = exp(kl)-kl-1`）；L121–124 组内 mean/std 归一（`unbiased=False`，分母 `std_r+1e-4`）。 |
| 1.1b | "默认 loss 是 cispo" | ✅ | `train_grpo.py` L228 `default="cispo", choices=["grpo","cispo"]`。cispo 分支 L135–137（`clamp(ratio,max=epsilon_high).detach() * A * logp − β·KL`）；grpo 分支 L139–142（对称 clip `1±ε`）。guide §4.4"正式消融建议 loss_type=grpo"可执行：`--loss_type grpo` 即可。 |
| 1.1c | 重写 `calculate_rewards`（guide §4.2："约 150-300 行"） | ⚠️ | 实际 **L37–68 共 32 行**（+格式工具 `rep_penalty` L31–34 共 4 行）。guide 行数估计严重偏大，**改动面比 guide 预期更小**：只改 `calculate_rewards` + 数据集即可，主循环（rollout→advantage→KL→loss→save）全保留。 |
| 1.1d | 奖励函数名称/签名/调用点 | ✅ | 函数 `calculate_rewards(prompts, responses, reward_model)` L37–68；调用点 L95（每 step 一次，入参 `prompts, completions, reward_model`）；返回 `torch.zeros(len(responses))`，长度 = `B*num_generations`，已在 `args.device`（L38）。 |
| 1.1e | "格式启发式分 + RM 打分"的具体构成 | ✅ | 逐条（L44–66，按 i∈prompt、j∈generation 双层循环，每样本：`response_idx=i*G+j`）：① 整段长度分 `0.5 if 20≤len≤800 else -0.5`（L54）；② 若含 `</think>`：thinking 段长度分 `1.0/-0.5`（L57）、闭合分 `+0.25/-0.25`（`count('</think>')==1`，L58）、`answer` 取 `</think>` 之后内容（L56–59）；③ `answer` 去重惩罚 `−rep_penalty(answer)`（L60，重复度分 0~0.5）；④ RM 连续分：`reward_model.get_score(messages, answer)`（L62），messages 从 prompt 文本正则 `(system|user|assistant)\s+…<\|im_end\|>` 解析（L50–52）；最后 `rewards += reward_model_scores`（L66）。**总分不 clip**（与 train_agent.py 不同，agent 版 L218/L238 有 `max(min(…,3),-3)`）。 |
| 1.1f | 是否已有动态采样过滤/长度惩罚 | ✅ 无 | 全循环无 DAPO 式过滤：退化组（组内全对/全错）只是 std≈0 → advantage≈0 → 静默零梯度，无显式处理。长度仅启发式阈值（L54/L57）无惩罚项。`rep_penalty` 是重复度惩罚，非长度惩罚。→ 两处消融点需新加（见 §2.d）。 |
| 1.1g | 组内 advantage 归一细节 | ✅ | L121–124：`rewards.view(-1, G)` → `mean_r/std_r`（`unbiased=False`）→ `repeat_interleave(G)` → `(r−μ)/(σ+1e-4)`，`[B*G]` 无 unsqueeze 的序列级标量 A（L137/L142 里 `advantages.unsqueeze(1)` 广播到 token）。 |
| 1.1h | token 级 KL / β / ref 模型 | ✅ | `ref_model` 独立 init（L276–277，`eval().requires_grad_(False)`，与 policy 同权重同源）；每 step 对完整序列重算 `ref_per_token_logps`（L104，`torch.no_grad`）；KL per-token（L132–133）计入 loss（cispo L137 / grpo L142 中 `−β·per_token_kl`），按 `completion_mask` 长度归一后取 mean（L143）。β 默认 0.1（L227）。注意：**序列内所有 token 共享同一 A**（序列级 reward 归一），非 token 级 PG loss —— 与 DAPO "token 级 PG loss" 是两回事，train_grpo 并未实现后者。 |
| 1.1i | group size/采样超参 | ✅ | G=6 默认（L226）；temperature 硬编码 0.8（rollout 调用点 L85，非 arg）；`max_gen_len=1024`（L224）；`max_seq_len=768`（L223，prompt 截断上限，在 L76–78 对 input_ids 右截断）；lm_config `max_seq_len=max_seq_len+max_gen_len=1792`（L254，仅用于 RLAIFDataset 构造参数，**Dataset 内未实际使用**，见 1.2d）。 |
| 1.1j | 日志/断点/保存 | ✅ | 控制台 Logger（L154–167）；wandb 名存实亡——`--use_wandb` 时 `import swanlab as wandb`（L263–269）记录 L169–178（reward/kl_ref/advantages_std/…/avg_response_len 等，无格式率/正确率/长度直方图）；权重每 `save_interval=10` 步存 `{save_dir}/{save_weight}_{hidden}.pth`（L183，**覆盖式**、`.half().cpu()`）；resume 检查点 `../checkpoints/{save_weight}_{hidden}_resume.pth`（L188–189 + trainer_utils `lm_checkpoint` L63–116，含 optimizer/scheduler/epoch/step/wandb_id）。**训练循环内无任何 eval**。 |

### 1.2 数据流与 Dataset（任务清单 #2、guide §4.4）

| # | guide 说法 | 判定 | 锚点与说明 |
|---|---|---|---|
| 1.2a | RLAIFDataset 用 `conversations[:-1]` 切片构造 prompt | ✅ | `dataset/lm_dataset.py` L212（`create_chat_prompt` 内 `self.tokenizer.apply_chat_template(conversations[:-1], …add_generation_prompt=True, open_thinking=use_thinking)`）。官方 rlaif.jsonl 的 schema（README L988–997）：`{"conversations":[user(问题), assistant(占位,内容任意如"无")]}` —— 末条 assistant 被切片丢弃，**数据本身不含 gt**。 |
| 1.2b | input/answer 从哪来 / 返回结构 | ✅ | RLAIFDataset `__getitem__`（L217–224）只返回 `{'prompt': str, 'answer': ""}` —— **answer 恒空、无 gt 字段**；GRPO 侧 `batch['prompt']`（train_grpo L73）tokenize 后 rollout，答案完全由 rollout 生成。→ 数学 RLVR 必须**新增带 gt 的单轮 Dataset**（见 §2.c），或扩展本类返回 gt。 |
| 1.2c | SFT 与 RLAIF 的 eos 约定有差异（guide §4.4） | ⚠️ 需修正 | 常量确有差异：SFTDataset bos 锚=`f'{bos}assistant\n'`（L65）、eos 锚=`f'{eos}\n'`（L66，含 `\n`）；RLAIFDataset 的 bos/eos 锚 = `f'{bos}assistant'`（L202）、`f'{eos}'`（L203，**无 `\n`**）。但 RLAIFDataset 里这两个锚**是死代码**：该类只产 prompt 字符串，不做 label/mask（L217–224），GRPO 的 completion mask 实际由 `tokenizer.eos_token_id` 在 rollout 输出里检测得到（train_grpo L126–130）。→ "eos 约定差异"对当前 GRPO 流程**无实义**；真正有差异的是 **SFTDataset 的 label 机制依赖"assistant\n"锚（L92 起）+ eos 锚**，自造单轮 SFT 数据必须让模板渲染结果与这两个锚精确匹配（模板渲染见 1.3b）。 |
| 1.2d | thinking_ratio / prompt 随机性 | ✅ | RLAIFDataset 每次 `__getitem__` 按 `random.random()<thinking_ratio`（L210，train_grpo 传 `args.thinking_ratio=0.9`，L292）决定 `open_thinking`。SFT 侧 `pre_processing_chat`（L9–29）以 20% 概率在无 system 的消息前插随机 system（**数学格式稳定性建议在数据里加固定 system 或格式指令写进 user**，见 §2.c 风险）；`post_processing_chat`（L31–35）以 80% 概率删空 `<think>\n\n</think>\n\n` 片段。两个随机源都不受 `setup_seed` 控制（发生在 worker/dataset 内），影响可复现性（低危）。 |
| 1.2e | 官方数据 schema（agent_rl_math 带 gt） | ✅（README 层面） | README L1178："`agent_rl.jsonl`/`agent_rl_math.jsonl` 相比普通对话数据多了 `gt` 作为最终校验目标"；README L508–509（agent_rl_math=18MB 数学补充数据，适合 RLVR）。仓库侧唯一消费方：`AgentRLDataset` L249–252 返回 `{'messages','tools','gt': sample['gt']}`；`train_agent.py` 里 gt 只在 **tool_call 分支**参与奖励（L221–236，`validate_gt_in_text` L183–186，**字符串包含式匹配 + 浮点抽取，很弱**，见 §5）。`train_grpo.py` 全程不用 gt。⚠️ **文件级 schema（gt 是 str 还是 list、conversations 是否含 tools/system 等）本地无法确认**（数据在远端 ModelScope）→ §4 运行时验证项。 |
| 1.2f | 数据目录/字段约定 | ✅ | 所有脚本 `--data_path` 默认 `../dataset/*.jsonl`（train_grpo L225、train_full_sft L102、train_dpo L149、train_agent L394）→ **cwd=trainer/ 是硬约定**（README L671："所有训练脚本均 `cd ./trainer` 目录执行"）。SFT 消费 `{"conversations":[…]}`（SFTDataset 强制 features L63）；DPO 消费 `{chosen:[…], rejected:[…]}` 平铺对话数组（L130/L137–138，README L465–478）；RL 消费 `{"conversations":[…]}`（末条 assistant 占位）；agent 消费 `{"conversations":[带tools…], "gt":…}`。 |

### 1.3 模板、tokenizer、模型（审计新增关键事实）

| # | 事实 | 锚点与含义 |
|---|---|---|
| 1.3a | **chat_template 强制 think 包装** | `model/tokenizer_config.json` L333 chat_template：渲染任何 assistant 消息时一律输出 `<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}<|im_end|>\n`；若 content 内含 `</think>` 则自动切分 reasoning/content；`add_generation_prompt=True` 时结尾追加 `<|im_start|>assistant\n` + （`open_thinking=true` ? `<think>\n` : `<think>\n\n</think>\n\n`）。→ SFT 答案若写成 `<think>…</think>\n\n答案是 X` 会被模板规范化成同构格式，**格式锚点是仓库内建的**（等价"Qwen 式 think 盒子"）；`<think>`/`</think>` 是普通 token（id 25/26，`special:false`），decode 后保留在文本里 → verifier 在文本层解析即可，与分词无关。 |
| 1.3b | 特殊 token id | tokenizer：`<\|endoftext\|>`=0(pad)、`<\|im_start\|>`=1(bos/角色头)、`<\|im_end\|>`=2(eos)。模型 config 默认一致（model_minimind.py L18–20）。`MiniMindConfig.vocab_size=6400`。 |
| 1.3c | 模型规模/加载 | minimind-3=64M：vocab6400 / L8 / d768 / kv4 / q8 / max_pos 32768 / RoPEθ1e6 / tie embeddings（README L577；model_minimind.py L12–45）。`init_model`（trainer_utils L119–131）：tokenizer `AutoTokenizer.from_pretrained('../model')`、权重 `{save_dir}/{from_weight}_{hidden}.pth` 经 `torch.load(map_location=device)` + `load_state_dict(strict=False)`；`from_weight='none'` 跳过加载（L123）。**权重文件是裸 state_dict `.pth`（非 HF safetensors）**。 |
| 1.3d | 手写 generate | model_minimind.py L257–288：纯 python 逐 token 循环（`temperature` 除法 L267、top_k L271、top_p L273、`multinomial` L278、eos 后强制续 pad L279、全行 finish 才停 L283–285）；**默认参数 `do_sample=True, temperature=0.85, top_p=0.85, top_k=50`**，无 repetition_penalty 默认 1.0。→ rollout 的 `max_new_tokens` 是每行硬上限，模型不输出 eos 时会烧满 1024 token（64M 上很慢，RLVR runbook 建议压到 512，见 §2.g）。 |
| 1.3e | 注意力 mask | model_minimind.py L125–131：attention_mask 全 1 或 None 才走 sdpa 快速路径，否则手动路径加 `(1-mask)*-1e9`。→ 训练侧 `full_mask`（含 pad=0 的右侧 completion pad）保证 pad 不参与注意力；生成侧左侧 pad 属正常用法。 |

### 1.4 奖励模型加载与 cwd 约定（任务清单 #3/#4、guide §4.4）

| # | guide 说法 | 判定 | 锚点与说明 |
|---|---|---|---|
| 1.4a | "RM 无条件加载" | ✅ | train_grpo L279 `reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)` 无条件执行；train_agent L440 同。`LMForRewardModel`（trainer_utils L160–177）：`AutoModel.from_pretrained(model_path, trust_remote_code=True)`（**本地目录，不是 HF hub 名**）+ `.to(device).eval()`；`get_score(messages,response)`（L167–177）把历史拼成"以上是对话历史…新问题"文本后调远端模型的 `.get_score(tokenizer, eval_messages)`，分数 clip 到 [-3,3]（L177）。**路径不存在 → AutoModel.from_pretrained 直接报错**（无降级）。 |
| 1.4b | "train_grpo 跑通离不开 RM？verifier-only 能否绕过" | ✅ 必须改代码 | RM 加载在主进程模型初始化段（L279），与奖励计算（L62/L95）分离。verifier-only 需：① 跳过/条件化 L279（如 `--reward_mode rule` 时不实例化）；② 奖励函数不再接收 reward_model。改法见 §2.d。 |
| 1.4c | verifier-only 能省什么 | ✅（推算） | RM=InternLM2-1.8B fp16 ≈3.5GB 常驻显存 + 每样本一次 1.8B 前向（train_grpo L44–63 是 **per-sample 串行循环**，B×6=12 次/step，每次前向含长序列 softmax 打分）→ 省 ≈3.5GB 显存 + step 时长中占比可观（GPU 上实测见 §4）；CPU/MPS 冒烟下 RM fp16 前向几乎不可用，**verifier-only 是唯一可冒烟路径**。 |
| 1.4d | cwd 约定（guide §4.4 "训练脚本要 cd trainer"） | ✅ | README L671 明示；所有路径默认值都以 trainer/ 为基准（`../out`、`../checkpoints`、`../dataset`、`../model`）。eval_llm.py 则以**仓库根**为基准（`./out/{weight}_{hidden}.pth` L22–23、tokenizer `./model` L13/L34）——README L335–343 也从根目录跑 eval。→ runbook（§2.g）固定两套 cwd。 |
| 1.4e | "reward_model_path 相对仓库根（../../internlm2-1_8b-reward 实际生效方式需实测）" | ⚠️ 需修正 | 静态可定：**该字符串就是相对 cwd 的路径**（train_grpo L232 默认 `../../internlm2-1_8b-reward`）。cwd=trainer 时解析为 **仓库同级** `internlm2-1_8b-reward/`（README L1008–1019 要求的"与 minimind 项目同级目录"布局），guide"相对仓库根"表述不准确，但**从 trainer/ 运行时两说法等效**。若从仓库根运行则指向"上上级"，必错。→ runbook 统一 `cd trainer` + 冒烟第一步打印 `os.path.abspath(args.reward_model_path)` 并断言目录存在。 |
| 1.4f | 其它 cwd 事实 | ✅ | resume 检测 `../checkpoints`（L255）与保存（L188–189）一致；`lm_checkpoint`（trainer_utils L63–116）无后缀 `_resume.pth`，含 world_size 换算（L110–114）。**同一 (save_weight, hidden) 的 resume 文件会跨 run 复用** → 实验必须每臂独立 `--save_weight` 前缀（§2.g 风险）。 |

### 1.5 rollout（任务清单 #5、guide §2 表）

| # | guide 说法 | 判定 | 锚点与说明 |
|---|---|---|---|
| 1.5a | 接口与样本数 | ✅ | `rollout_engine.rollout(prompt_ids, attention_mask, num_generations, max_new_tokens, temperature=0.8)`（rollout_engine.py L55）；train_grpo 每 step：B 个 prompt × G=6 → 返回张量行数 B×6（torch 引擎 L76 `repeat_interleave`）。RolloutResult（L40–47）：output_ids/completion_ids/per_token_logps/completions(list[str])/prompt_lens/completion_mask。 |
| 1.5b | torch 后端生成参数 | ✅ | TorchRolloutEngine（L64–95）：`model.generate(..., do_sample=True, temperature=0.8, pad_token_id, eos_token_id)`（L75–84），**top_p/top_k 未传**（走 generate 默认 0.85/50，见 1.3d）；completions=`batch_decode(..., skip_special_tokens=True)`（L89）→ `<think>` 等非 special token 保留；completion_mask 全 1 占位（L92，真实 mask 在训练侧按 eos 检测）。old logps 由 rollout 后 `compute_per_token_logps` 前向得到（L88/L24–36）。 |
| 1.5c | torch 后端 MPS 支持 | ⚠️ 无显式 mps 支持但结构上可跑 | rollout/训练全是设备无关算子（generate L257–288 只有 cat/multinomial/sdpa 等）；障碍点：① 脚本默认 `--device cuda:0 else cpu`（L213）→ 必须显式 `--device mps`；② autocast 只在 cuda 生效（L258–260 `device_type=="cpu"→nullcontext`）→ MPS 走 fp32 全程（显存小但慢，可接受）；③ `setup_seed` 里 `torch.cuda.manual_seed*`（trainer_utils L58–59）在无 CUDA 机器上无害（no-op）；④ RM fp16 上 MPS 大概率失败/极慢 → 冒烟只跑 rule 模式。 |
| 1.5d | "sglang 权重滞后 save_interval 步（隐含 off-policy）" | ✅（仅 sglang 引擎） | `update_policy` 只在 `step % save_interval == 0 or step == iters` 被调用（train_grpo L193）；sglang 引擎的 `update_policy` 是全量落盘 + HTTP push（rollout_engine L175–194）→ 两个 save_interval 之间 sglang server 持旧权重 = **off-policy rollout**。**torch 引擎无此问题**：其 `update_policy` 只是指针重指（L94–95），且 rollout 全程引用同一 model 对象（L72），每步都是最新权重（同步 on-policy）。→ 主实验固定 torch 引擎与 guide 裁决一致（guide §4.3）。 |
| 1.5e | 引擎创建 | ✅ | factory L209–224；torch 引擎持有 policy_model/tokenizer/autocast_ctx（L65–69）；train_grpo 里 compile/DDP 后都重调 `update_policy`（L310–316）。 |

### 1.6 冷启动 SFT / 评估与日志 / 缺陷（任务清单 #6/#7/#8）

| # | guide 说法 / 审计发现 | 判定 | 锚点与说明 |
|---|---|---|---|
| 1.6a | SFT 数据格式：jsonl `{"conversations":[…]}` 单轮 QA 直接支持 | ✅ | `SFTDataset`（lm_dataset L58–119）吃任意轮次对话（强制 features L63）；单轮 `[user, assistant]` 即可。loss mask：以 `"<\|im_start|>assistant\n"` 为锚标记 assistant 段（L88–104），其余全 `-100`；模型侧 `F.cross_entropy(ignore_index=-100)`（model_minimind.py L252）。→ "冷启动数据=清洗后数学单轮 QA" 可直接喂 `train_full_sft.py --data_path ../dataset/math_sft_train.jsonl --from_weight full_sft`，**唯一约束**：assistant content 里含 `<think>…</think>` 时模板会切分（1.3a），内容与模板渲染结果需一致（§2.e 验收里有 diff 校验步骤）。 |
| 1.6b | 参考超参（README 锚点） | ✅ | 64M 单卡 3090：SFT≈1.1h/epoch（sft_t2t_mini）、RLAIF≈1.1h/epoch（README L623）；train_full_sft 默认 batch16/lr1e-5/epochs2/max_seq_len768/save_interval1000/from_weight=pretrain（L88–104）；train_grpo 默认 batch2/lr3e-7（L211–212）。冷启动自造数据体量 ~1000 条 → SFT 1–2 epoch 在 3090 上分钟级，与"≈1.1h"（那是 1.6GB 全量数据）不冲突。 |
| 1.6c | eval：现有入口 | ✅ | `eval_llm.py` 只是**交互式/固定问题聊天 CLI**（L51–91），从仓库根跑（`--load_from model` 默认值 L34 即 `./model`）；**无法批量打分、无 pass@1、不落盘** → heldout 评估需新写批处理脚本（§2.f/g）。`scripts/eval_toolcall.py` 是 agent 工具场景专用（README L1346 展示其输出格式，非数学 QA 批量评估）。 |
| 1.6d | 日志落盘 | ⚠️ 薄弱 | 全部脚本只 print + 可选 swanlab（L169–178 / L263–269）；**无本地曲线文件**。四曲线+heldout 的落盘是最小新增（§2.f）。 |
| 1.6e | 种子/可复现性 | ⚠️ | 每 epoch 前 `setup_seed(42+epoch)` + `torch.randperm` 洗牌（train_grpo L321、train_full_sft L160），但 Dataset 内随机（thinking 采样、SFT 随机 system、空 think 删除）不受控 → 同参重跑不完全逐位一致（RL 语境可接受，记录 seed 即可）。 |
| 1.6f | 数据泄漏防护现状 | ✅ 无内置 | 训练脚本不分 train/heldout；官方 rlaif/agent_rl 下载即训。→ 切分策略必须由新清洗脚本保证（§2.b）。 |
| 1.6g | 其它发现（缺陷/坑清单，详见 §5 展开） | — | ① 权重保存 `.half()` + `strict=False` 加载（trainer_utils L73/L127）：resume 后训练精度=加载进 fp32 参数再 autocast，行为可预期但**权重文件是 fp16**（eval 加载需注意）。② 保存路径按 `save_weight` 覆盖式写（L183），同前缀跨 run 互相覆盖 + resume 文件残留污染（§1.4f）。③ reward 计算是 python 双层循环 + RM 串行 12 次前向（L44–63），是 step 热点。④ 每 step 三次全序列前向（rollout old-logp L88、policy L99–101、ref L104），64M 无所谓，0.5B 起不可扩展（本项目不涉及）。⑤ `calculate_rewards` 里 heuristic 与 RM 相加后**不 clip**，与 README"reward≈4"曲线一致；但混合规则奖励时要小心尺度。⑥ DDP 分支 `args.device` 会被改写为 `cuda:{local_rank}`（L248），MPS 单机不受影响。⑦ 官方 grpo 权重存在（guide §2 表）但本地无 → D1 需下载，`from_weight=grpo` 时加载路径为 `../out/grpo_768.pth`。⑧ guide 提到的"rlaif 数据 assistant 占位"样例中 assistant content="无"，若有人把占位当真 assistant 历史则会污染 RM 的 messages 解析（L50–52 会把"无"当作历史 assistant 消息）→ 自造数据不放占位 assistant，见 §2.c。 |

---

## §2 最小改动实施清单

改动原则：**不动 train_agent.py；不动 rollout_engine.py（torch 引擎已够）；train_grpo.py 只做可开关的增量补丁；verifier/数据/评估全部新文件**，保证"官方脚本可随时恢复、diff 小而清晰"（简历口径：fork 内改造 + 自建数据/评估脚本）。

### a) 新 verifier 模块 —— `trainer/math_verifier.py`（纯 Python，无 torch 依赖）

**目标**：确定性三态判定（判对/判错/无答案）+ 可选分档奖励 + 格式/长度统计。与模型规模无关；全部在解码后的文本层操作。

```python
# trainer/math_verifier.py（新文件，~120 行 + 自测 main）
ANSWER_MARKER = "答案是"          # 建议与 SFT 教法一致的中文锚；guide 的 "####" 亦可，二选一并在清洗/SFT/eval 三处同步
MAX_EXTRACT_LEN = 64              # 答案行防呆上限

def split_think(response: str) -> tuple[str, str]:
    """返回 (thinking, final)。无 '</think>' 时 thinking=''，final=response.strip()；
    有则取最后一个 '</think>' 之后文本为 final（guide §4.2：扫全文取最后一个闭合标签）。"""

def extract_answer_text(response: str, marker: str = ANSWER_MARKER) -> str | None:
    """final 文本中取最后一个 marker 之后到行尾的内容（去首尾空白）；
    无 marker 或内容超 MAX_EXTRACT_LEN → None（判定为无答案，不猜）。"""

def normalize_number(text: str) -> str | None:
    """确定性规约：去空白/全角空格；全角数字/小数点→半角；去千分位 ',' '，'；
    Decimal(text) 数值化后输出规范串（'59'=='59.0'）；失败(含中文数字/分数/空)→ None。
    不引入容差、不解析中文数字（数据侧规避，见 b)。"""

def verdict(response: str, gt: str, marker: str = ANSWER_MARKER) -> dict:
    """返回 {'correct': bool, 'no_answer': bool, 'extracted': str|None,
              'think_closed': bool, 'marker_hit': bool}。判定：无 marker/提取失败→无答案；
    normalize_number(gt) is None → 数据错误哨兵(记 log，判 False)；否则精确串比较。"""

def reward_tier(v: dict, mode: str = "binary") -> float:
    """binary:  对=+1.0 错=0.0 无答案=-0.5
       tiered:  对=+1.0 错=-0.5 无答案=-1.0  （更陡，供消融）
       如需 guide §3.4 的"格式分档"作消融臂：+0.25 当 think_closed/marker_hit（仅 tiered_format 模式）。
    返回同一 reward 用于全部 G 个样本的公式必须只依赖 v，保证组内可比。"""

def reward_with_len_penalty(r: float, n_tokens: int, cap: int = 300, coef: float = 0.1) -> float:
    """长度惩罚（消融开关）：r -= coef * max(0, n_tokens - cap)/cap；n_tokens 来自 completion_mask.sum(dim=1)。"""
```

关键设计决策（写进注释与面试话术）：
- 不照抄 veRL 的"扫最后 300 字符"；也不照抄仓库 `validate_gt_in_text`（train_agent L183–186，**子串包含匹配会把 "1" 匹配进 "21"、'59' in '1959' 判对**——这就是奖励 hack 温床）。确定性精确匹配优先，抽取鲁棒性靠 SFT 教格式而不是解析器兜底。
- 单测：合成 200 条（对/错/无答案/边界：`59 vs 59.0`、负数、全角、千分位、`</think>` 缺失、marker 出现在 think 内等），验收：抽取命中 ≥99%（guide D3–4 退出标准）。

### b) 数学问答数据清洗 —— `dataset/scripts/clean_math_data.py`（新文件，仓库内 scripts/ 或 dataset/ 均可，建议 `scripts/` 与现有脚本同层）

**输入**：① 远端 `agent_rl_math.jsonl`（下载到 `dataset/`）；② 自造题库脚本（模板算术题生成器，见下）。**输出**（全部落到 `dataset/`）：
- `math_sft_train.jsonl`：`{"conversations":[{"role":"user","content":"<带格式指令的题目>"},{"role":"assistant","content":"<think>\n<简短推理>\n</think>\n\n答案是 <num>"}]}`
- `math_rlvr_train.jsonl`：`{"conversations":[{"role":"user","content":"<同一题目+同一格式指令>"}],"gt":"<num>","source":"agent_rl_math|synthetic"}`（注意：**无占位 assistant**，避免 1.6g ⑧ 的占位污染；RL Dataset 不切片即可）
- `math_heldout.jsonl`：同上但**只含 user + gt**（评估脚本用；也可带 gold assistant 供参考）
- `math_heldout_report.json`：切分统计（train/heldout 条数、source 占比、答案类型分布、难度桶计数）。

处理规则（agent_rl_math 条目）：
1. 保留含 `gt` 且 `conversations` 可解析的条目；剥离 system 里的 tools 定义、tool/tool_calls 消息，取**最后一条 user 文本**为题目（多轮工具链样本若无法还原为单轮问题则丢弃，不硬拆）。
2. **答案侧规避**（guide §4.2）：gt 必须能被 `normalize_number` 解析为有限十进制 → 全角/中文数字/分数/科学计数先转换，转换失败丢弃；含单位（"5元""3kg"）丢弃。
3. 题目文本统一：中文叙述 + `请计算` + 表达式（agent_rl_math 原题常见 `(94)-35`、`3**2`、`(4**3)+(20**2)` 这类 `**` 幂运算风格，README L1350–1395 佐证），转成自然中文问句模板 + **格式指令**（如"请先思考再回答，并在最后一行以『答案是 …』输出数字结果"）。
4. 自造补齐：确定性模板生成（加法/减法/乘法/两步混合/小整数幂/简单除法出整数），**gt 由受限 eval 计算**（参照 train_agent L58 的 `eval(…, {"__builtins__":{}})` 风格，白名单 `+ - * / ** ( )` 与整数，杜绝任意代码）；难度桶分 2–3 档；生成时记录每个表达式哈希。
5. **防泄漏切分**：以"表达式规范化哈希 + 数值"为 key 先去重，再按 key 的 hash 分层抽样 15–20% 作 heldout（200–400 条目标，guide §4.2）；**train 内 SFT/RLVR 共用同一 train 集，heldout 全程不参与任何训练/微调**；换模板族变体（措辞、加不加空格、标点）只出现在 heldout，防 verbatim 记忆。诚实口径：heldout 只能证明"格式泛化+指令遵循+小范围数值泛化"，不能证明"新颖推理"，文档与简历必须如实写。
6. 清洗后人工抽查 ≥50 条 gt 无错（guide D3–4 验收）。

### c) 单轮 RL Dataset —— `dataset/math_rl_dataset.py`（新文件 ~60 行）或直接扩 `RLAIFDataset`

```python
# ~55 行伪代码
class MathRLDataset(Dataset):
    """单轮数学 RLVR 数据。schema: {"conversations":[{user}], "gt": "123"}"""
    def __init__(self, jsonl_path, tokenizer, thinking_ratio=1.0, max_length=768):
        self.samples = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]  # 小文件直接读内存，避开 HF datasets 序列化
        self.tokenizer = tokenizer; self.thinking_ratio = thinking_ratio

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        conv = self.samples[i]["conversations"]           # [{"role":"user","content":Q}]（清洗器保证）
        use_thinking = random.random() < self.thinking_ratio
        prompt = self.tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True, open_thinking=use_thinking)
        return {"prompt": prompt, "gt": str(self.samples[i]["gt"])}
```

- 模板直接复用 `apply_chat_template`（含 1.3a 的 think 生成头），**与官方 RLAIF 路径逐字一致**；不手拼字符串。
- 两点与 RLAIFDataset 的差异是故意的：**不切片**（数据无占位 assistant）、**返回 gt**。
- `thinking_ratio` 建议 RLVR 主 run 固定 1.0（打开 think 盒，格式可判）；0.0 作消融（考验无 think 情况下模型仍输出"答案是 …"）。
- DataLoader：默认 collate 即可得 `batch['prompt']/['gt']`（list[str]）——与 train_grpo L73 兼容；**num_workers=0**（MPS 冒烟与 spawn 兼容问题，见 §5）；collate_fn 可留扩展口。

### d) `trainer/train_grpo.py` 最小补丁（改动点全列）

| # | 位置（现有行号） | 改什么 | 风险 |
|---|---|---|---|
| d1 | argparse 区 L206–244 末尾 | 新增：`--reward_mode {rule,rm,mix}`（默认 rule）、`--reward_marker`（默认"答案是"）、`--verdict_tier {binary,tiered}`、`--len_penalty_coef`（默认 0.0=关）、`--dynamic_sampling 0/1`、`--run_tag`（默认 ""，用于日志文件前缀）、`--log_dir`（默认 `../logs`）、`--rlvr_data_path`（单轮数据，默认 `../dataset/math_rlvr_train.jsonl`）。 | 参数面新增不影响官方用法；把"数学模式"整体挂在 `--reward_mode rule` 下，避免破坏原 rlaif 流程（对照 RM 臂仍走原代码）。 |
| d2 | L279 RM 加载 | 改为 `if args.reward_mode in ("rm","mix"): reward_model = LMForRewardModel(...)`；rule 模式不实例化，`calculate_rewards` 不接 RM。同时打印 `abs path(reward_model_path)` 存在性断言（§1.4e）。 | verifier-only 冒烟与 24GB 主 run 共用一条路径；RM 臂显存/时间成本如实记录（guide §4.4"曲线对比不比 reward 绝对值"）。 |
| d3 | L292 Dataset | `if args.reward_mode == "rule": train_ds = MathRLDataset(args.rlvr_data_path, tokenizer, thinking_ratio=…)`，否则保持 RLAIFDataset。 | 数据集与奖励模式绑定会隐含"rule 只跑数学单轮"约定——runbook 文档化，不留隐式状态。 |
| d4 | `calculate_rewards` L37–68 | 拆成三个纯函数 + 薄壳：
`def rewards_rule(completions, gt_batch, G, mode, marker, tier)`：逐样本 `v=verdict(...)` → `r=reward_tier(v)`（+可选 `reward_with_len_penalty`，长度取自训练侧 mask，所以薄壳返回 `(r, v)`，长度项在调用点算）；`def rewards_rm(completions, prompts, rm)` = 原 L44–66 内容原样搬入（**RM 臂与官方行为逐位一致**）；`def rewards_mix(...)=α·r_rule+β·r_rm`（α/β 默认 1.0，arg 化）。返回 `(rewards_tensor, stats)`，`stats={'correct':np.bool_, 'no_answer':…, 'format_ok':…, 'marker_hit':…}`（纯 CPU 计数，无 torch）。 | 原启发式格式分在 rule 模式下**整体弃用**（guide 裁决：规则臂只留"对/错/无答案"，格式率走统计曲线）→ 避免"格式分先于正确率收敛"污染主臂；`rewards_rm` 保持原样保证对照臂公平。 |
| d5 | L95 调用点 | `rewards, stats = calculate_rewards(prompts, completions, gt_list=batch.get('gt'), …)`；`gt_list` 来自 batch（d3 的 Dataset 提供）。**rule/mix 需要 `batch['gt']`，rlaif 无 gt → 传 None，内部守卫**。 | gt 与 prompt 的行序必须一一对应（同 DataLoader batch）；断言 len(gt)==B 防错位。 |
| d6 | 动态采样过滤（新接入点，消融 A） | 在 L121–124（group 归一）后：`group_std = grouped_rewards.std(dim=1)`；`if args.dynamic_sampling: keep = group_std > 0`（组内 reward 全同=全对或全错 → 整组无信息）；loss 聚合 L143 处按 `keep.repeat_interleave(G)` 掩码再 mean（仿 train_agent L330–331 的 `valid_rows` 写法）；日志记 `filtered_groups/total`。 | DAPO Dynamic Sampling 的语义=过滤全对/全错组（避免把"已经全会"的组继续推向格式 hack）；小 B 下过滤后有效样本少 → 统计噪声如实记录（guide §4.4）。默认关，消融开。 |
| d7 | 长度惩罚（新接入点，消融 B） | 在 d5 后：`if args.len_penalty_coef > 0: rewards -= coef * (mask_len − cap).clamp(min=0)/cap`（cap=300 建议，与 max_gen_len=512 配合）；mask_len=`completion_mask.sum(dim=1)`（已有 L158 计算处）。 | 惩罚进 reward 会在组内归一后改变相对排序 → 只能整组用同一 cap；与 §2.a `reward_with_len_penalty` 保持一致实现，二选一（建议在 rewards 层实现，便于日志归因）。 |
| d8 | 日志块 L154–167 + wandb L169–178 | 追加 rule 模式的**四曲线**：`format_rate = mean(stats.format_ok)`、`correct_rate = mean(stats.correct)`、`no_answer_rate`、`avg_len`（已有 L158）+ `filtered_groups`；本地落盘 CSV（§2.f）。 | 与 RM 臂对比时 RM 臂不产出 stats → CSV 列写 NA 并注释"RM 臂无 verifier 口径"（guide §4.4 只比无量纲曲线）。 |
| d9 | save 区 L180–191 | 无改动；但 runbook 每 run 用唯一 `--save_weight`（如 `rlvr_rule_bin`/`rlvr_rm`/`rlvr_mix`/`sft_mix_ablation_len`…），杜绝 1.4f 覆盖/污染。 | — |
| d10 | debug 采样 L106–119 | debug 分支同步打印 `gt / verdict / extracted`（1 行），并每 `--debug_interval` 把**原始样本（prompt+completion+gt+reward+verdict）追加进 `logs/{run_tag}_samples.jsonl`**（人工抽查 20 条的素材源，guide D9–10）。 | 落盘文件增长可控（仅 debug 模式 & 有间隔）。 |

**改动面合计**：train_grpo.py ≈ 10 处小补丁（约 +120 行/−20 行），全部开关化；官方 rlaif/RM 路径在 `--reward_mode rm` 下与改造前等价（可用官方数据回归验证 5 步）。**不改**：rollout_engine.py、train_agent.py、trainer_utils.py、model/*。

### e) 冷启动 SFT 跑法

- 数据：§2.b 的 `math_sft_train.jsonl`（含 `<think>` 格式的 gold 答案，模板 1.3a 会规范化）。
- 命令（cwd=`trainer/`，24GB GPU）：
  ```
  python train_full_sft.py --data_path ../dataset/math_sft_train.jsonl \
      --from_weight full_sft --save_weight math_cold --epochs 2 \
      --batch_size 16 --learning_rate 1e-5 --max_seq_len 512 \
      --num_workers 4 --save_interval 500 --log_interval 50
  ```
  说明：`from_weight` 需先在 `out/` 备好官方 `full_sft_768.pth`（ModelScope `gongjy/minimind-3-pytorch`）。若想从 zero 路线（官方 `pretrain_768.pth` 或 `full_sft_zero_768.pth`）起，改 `--from_weight` 即可——**RL 阶段 ref 必须与 init 同源**（都用 `math_cold_768.pth`），所以 base 的对话能力直接影响冷启动质量，推荐 `full_sft_768` 起步（guide D5 语义）。
- 1–2 epoch 后验收：greedy 生成 heldout 20 条人工全过"格式+答案锚"（guide D5 退出标准）；同时建议跑 50 条统计 `normalize_number` 抽取命中 ≥99%（见 §2.f 的 eval 脚本）。若批量 greedy 全空/格式崩 → 先查 SFT 数据模板与 `post_processing_chat` 的交互（§5 风险 6）。

### f) 四曲线统计与日志落盘（最小实现）

原则：**不动 swanlab/wandb 通路**（远端不可用也不阻塞），新增纯本地 CSV。

1. 复用 train_grpo 既有日志块（d8）计算的标量，在 `grpo_train_epoch` 内追加：
   ```python
   if is_main_process() and (step % args.log_interval == 0 or step == iters):
       row = f"{time.time():.3f},{step},{avg_reward_val:.4f},{format_rate:.4f},{correct_rate:.4f},{no_answer_rate:.4f},{avg_len_val:.2f},{kl_ref_val:.4f},{advantages_std_val:.4f},{policy_loss_val:.4f},{filtered:.0f}\n"
       with open(f"{args.log_dir}/{args.run_tag}_metrics.csv", "a") as f: f.write(row)
   ```
   （首次写入时先写表头；`os.makedirs(args.log_dir, exist_ok=True)`。）
2. heldout 评估脚本 `eval_heldout.py`（新文件，仓库根可跑或 trainer/ 下跑，二选一固定）：加载 `math_heldout.jsonl` → 用与训练**完全一致的模板**（同一 `apply_chat_template` 调用、同 marker）→ 对每个 checkpoint 权重（`out/math_cold_768.pth`、`out/rlvr_*_768.pth`、官方对照权重）跑 **greedy + pass@1@16（temp 0.8）**（guide §7 第 18 题口径：固定温度报方差）→ 输出 `logs/{run_tag}_eval.json`：`{acc_greedy, acc_pass1, mean_len, format_rate, per_difficulty:{…}, samples:[20 条人工抽查存档]}`。评估 decode 用与 rollout 相同引擎（torch 引擎直调 `model.generate`，eos/pad 对齐），**保证训练/评估格式一致**。
3. 曲线图（可选）：matplotlib 已注释（requirements L8），用 CSV 现算即可；不强加依赖。

### g) 实验 runbook（含 cwd 要求）

约定：**训练一律 cwd=`trainer/`**（路径相对基准），**eval 脚本可按其设计 cwd 任意但文档固定**；权重全部放 `out/`（trainer 视角 `../out`），resume 放 `checkpoints/`，指标放 `logs/`（新目录）。

**Phase 0 冒烟（本机 MPS，验证代码路径，不产实验结论）**
```
cd trainer
# 0a. 最小合成数据（30 条 SFT 式 + 30 条 RLVR 式 + 10 heldout），仅代码通路
python ../scripts/clean_math_data.py --synthetic-only --out ../dataset/  # 新脚本自带 --synthetic-only
# 0b. SFT 冒烟：--device mps --num_workers 0 --dtype bfloat16（无效，mps 走 fp32 默认即可）
python train_full_sft.py --data_path ../dataset/math_sft_train.jsonl --from_weight none --save_weight math_smoke \
    --epochs 1 --batch_size 2 --max_seq_len 256 --num_workers 0 --device mps --save_interval 99999
# 0c. RLVR 冒烟：rule 模式、极小规模（无 RM！）
python train_grpo.py --reward_mode rule --data_path ../dataset/math_rlvr_train.jsonl --rlvr_data_path ../dataset/math_rlvr_train.jsonl \
    --from_weight math_smoke --save_weight grpo_smoke --batch_size 1 --num_generations 4 --max_gen_len 128 \
    --max_seq_len 256 --epochs 1 --num_workers 0 --device mps --save_interval 99999 --log_interval 1 \
    --run_tag smoke --loss_type grpo --beta 0.1
# 0d. verifier 单测：python -m trainer.math_verifier（合成 200 条断言 ≥99%）
```
冒烟验收：脚本无异常走完 ≥2 步、`logs/smoke_metrics.csv` 有行、debug 采样 jsonl 有内容、无 RM 加载日志（rule 模式）、MPS 无算子报错。若 MPS 太慢（python 逐 token generate 12 行×128 token），把 `--max_gen_len 64` 再降。

**Phase 1 正式（24GB CUDA 单卡）** —— 每 run 独立 `--run_tag`/`--save_weight`：
```
# S1 冷启动（真实数据清洗后；from_weight 需官方 full_sft_768.pth 已在 ../out/）
cd trainer && python train_full_sft.py --data_path ../dataset/math_sft_train.jsonl \
    --from_weight full_sft --save_weight math_cold --epochs 2 --batch_size 16 \
    --learning_rate 1e-5 --max_seq_len 512 --num_workers 4 --save_interval 1000
# S2 基线评估（SFT 权重 greedy/heldout）
cd .. && python eval_heldout.py --weight math_cold --out_dir out --hidden 768   # 脚本按根目录 cwd 设计，见 f2
# S3 主实验 RLVR（三臂）——务必不同 save_weight，避免互覆盖
cd trainer
python train_grpo.py --reward_mode rule  --from_weight math_cold --save_weight rlvr_rule  --run_tag rlvr_rule  \
    --data_path ../dataset/math_rlvr_train.jsonl --rlvr_data_path ../dataset/math_rlvr_train.jsonl \
    --batch_size 2 --num_generations 6 --loss_type grpo --learning_rate 3e-7 --beta 0.1 \
    --max_gen_len 512 --epochs 1 --num_workers 4 --save_interval 50 --log_interval 5 --debug_mode --debug_interval 50
#（lr 扫 3e-7 / 1e-6：再跑一份 --run_tag rlvr_rule_lr1e6 --learning_rate 1e-6 --save_weight rlvr_rule_lr1e6）
# 对照臂 RM-only 与混合臂（此时才需要 RM，--reward_mode rm / mix；先 S1 之外把 RM 放到仓库同级并冒烟断言路径）
python train_grpo.py --reward_mode rm  --from_weight math_cold --save_weight rlvr_rm  --run_tag rlvr_rm \
    --data_path ../dataset/math_rlvr_train.jsonl --from_resume 0 ...（同 S3 超参）
# S4 消融（对 rule 臂）：--dynamic_sampling 1；--len_penalty_coef 0.05；thinking_ratio 0（关闭 think 盒）
# S5 heldout pass@1：对 math_cold / rlvr_rule / rlvr_rm / rlvr_mix / 官方 grpo 768 权重逐个跑 eval_heldout.py
# S6 人工抽查 20 条 + 四曲线 CSV 出图 + 结论段
```
超参依据：SFT 组 batch16/lr1e-5（train_full_sft 默认，README 锚点）；RLVR 组 batch2/lr3e-7（train_grpo 默认，guide D6 试 3e-7/1e-6）、β=0.1 默认、G=6 默认；`max_gen_len` 从默认 1024 收到 512（64M rollout 时间与"不输出 eos 烧满上限"风险折中）；`save_interval` 从 10 提到 50 减少 sglang 无关的落盘开销（torch 引擎 update_policy 无成本，但权重落盘 12×~50 步一次足够）。

---

## §3 实施顺序与每步验收标准（guide D1–D14 细化为 D1–D7 实现里程碑）

| 里程碑 | 内容（新文件/改动） | 验收标准（可执行检查） |
|---|---|---|
| D1 骨架落地 | `trainer/math_verifier.py` + 单测；`scripts/clean_math_data.py`（先 `--synthetic-only`） | 单测 200 条合成用例：抽取命中 ≥99%、无答案/判错三态无歧义；清洗脚本在空数据/坏行上不崩 |
| D2 数据与 Dataset | 下载 agent_rl_math.jsonl 并清洗出三件套 + report；`dataset/math_rl_dataset.py` | 人工抽查 50 条 gt 无误；train/heldout 无 key 重叠（脚本断言）；heldout 200–400 条；`MathRLDataset` 单条输出与 RLAIF 官方 prompt 渲染格式 diff 为空（同一 template） |
| D3 冒烟（MPS） | 全部 §2.g Phase 0 命令 | ≥2 step 跑通、CSV/样本 jsonl 落盘、rule 模式无 RM 加载；记录 MPS step 耗时 |
| D4 SFT 冷启动（GPU） | S1 + S2 | loss 下降；heldout greedy：人工 20 条**全过** think 闭合+`答案是` 锚（guide D5 口径）；若 <80% 回查 §5 风险 6 模板问题 |
| D5 主实验三臂 | S3（rule 先跑通 1/4 量 50 步）+ RM 冒烟断言 | rule 臂 reward/格式率/正确率/长度四曲线形态可解释（正确率从 SFT 基线附近起步而非 0）；RM 臂加载成功日志 + 显存峰值记录 |
| D6 全量 run + 消融 | S3 全量 + S4 + S5 | 三臂 ×（greedy/pass@1@16）对照表；动态采样/长度惩罚两消融表（**无效也是结果**，guide 红线：诚实记录） |
| D7 归档 | 四曲线图、20 条抽查存档 jsonl、`repo diff` 清单、命令 runbook 回放一遍 | 从空 `out/` 到三臂全曲线可复现（脚本幂等）；文档 §5 风险全部核销或转 §4 遗留 |

---

## §4 开放问题清单（静态无法确认，必须运行时验证）

1. **数据真实 schema**：`agent_rl_math.jsonl`（以及 rlaif.jsonl）首行字段、`gt` 类型（str/list[int]/str[]）、conversations 是否带 system+tools、题目语言/风格、样本条数与长度分布 —— 下载后 `head -5` + `python -c "… json 解析统计"` 确认后再写清洗规则；**清洗器对未知字段一律显式报错**，不静默丢弃。
2. **RM 包加载**：`internlm2-1_8b-reward` 目录布局与 `trust_remote_code` 的 `get_score` 签名/行为（trainer_utils L176 调用点）；本地目录缺失时 AutoModel 报错类型；24GB 卡上 fp16 加载耗时与每 step RM 前向耗时（用于 §1.4c 的节省量实测量化）。
3. **官方权重可用性**：ModelScope `gongjy/minimind-3-pytorch` 的 768 档文件清单（guide 说含 pretrain/full_sft/full_sft_zero/**grpo**/ppo_actor/agent，无 dpo）——逐名核对后写入 runbook 下载段。
4. **MPS 兼容性**：`MiniMindForCausalLM.generate`（逐 token 循环 + multinomial + scatter）在 fp32/MPS 上的算子覆盖与耗时；`apply_chat_template(..., open_thinking=…)` 在本机 transformers 4.57.6 是否正常（模板参数透传）；DataLoader `pin_memory=True`+`num_workers=0` 无副作用确认（代码 L324 固定 pin_memory=True）。
5. **模板渲染一致性的实证**：对同一条 SFT 数据，`SFTDataset.generate_labels` 的 mask 是否恰好覆盖 `<think>…</think>\n\n答案是 …` 全段（含 eos `\n`）——打印一条 label 序列人工核对（lm_dataset L114–118 有现成调试注释）。
6. **reward 尺度**：rule 三态奖励（1/0/-0.5）下组内 std=0 的退化组占比（早期可能 >50%）——D3 冒烟即统计，决定是否需要 `dynamic_sampling` 默认开。
7. **swanlab/wandb 联网**：本机/GPU 环境若不可联网，`--use_wandb` 会卡在哪一步；本地 CSV 是否已够（建议主流程不开 wandb）。
8. **GRPO loss 数值自检**：改造前后同一 rlaif 数据 `--reward_mode rm` 跑 5 步，loss/reward 曲线与改造前基线 diff≈0（回归护栏，防 d4 拆函数引入行为漂移）。

---

## §5 风险清单（含 guide 未覆盖的新发现）

**guide 已覆盖（执行要点）**
1. 小模型能力门槛/稀疏奖励：冷启动 SFT 是地基，verifier 正确率若长期 ~0 属预期诊断（guide §4.2），记录而不是修数据。
2. 奖励 hack：主臂不用格式分作奖励（格式只入统计）；verifier 强正则 + 确定性精确匹配；数据侧规避中文数字/单位/分数（guide §4.2）。
3. KL/β 与熵塌缩：β=0 绝不跑（guide §3.4 第 5 条）；β=0.1 起步，观测 `kl_ref` 曲线若暴增先降 lr。
4. sglang 权重滞后：主实验固定 torch 引擎（guide §4.3），已在 §1.5d 静态证实。

**本审计新发现（guide 未列）**
5. **save_weight 覆盖 + resume 残留**（§1.4f/1.6g①）：`save_weight` 同名即互覆盖且 `_resume.pth` 跨 run 复用 → runbook 强制每 run 唯一前缀；`from_resume` 只在明确续跑时开。
6. **SFT 模板/随机预处理交互**：`pre_processing_chat` 20% 插随机 system + `post_processing_chat` 80% 删空 think（lm_dataset L9–35）会**随机扰动数学格式样本** → 冷启动数据把格式指令写进 user content（固定）、并在 system 字段预置（触发 L26 分支则不再随机插）；空 think 删除对含内容 think 无影响，但仍需 D4 抽查确认。
7. **verifier 语义陷阱**：a) `answer` 若用"答案是 59 元"这类带单位文本 → normalize 失败判无答案（数据侧规避，§2.b）；b) 多答案（模型列出 2 个候选数）→ 提取器只取最后一个 marker 后的第一个数，规则简单可测；c) decode `skip_special_tokens=True` 会吃掉 `<|im_end|>` 但保留 `<think>`（§1.3a）——**不要**用 `skip_special_tokens=False` 做 verifier 输入，防特殊 token 混淆。
8. **MPS 冒烟的无效结论风险**：随机权重(from_weight none)+超小数据只能验证代码通路，所有"指标"一律不对外；冒烟产物用 `smoke` 前缀隔离，防混入正式 logs。
9. **eval 一致性风险**：heldout 评估若模板/采样超参与训练 rollout 不一致（如 top_p 默认差异），格式率会失真 → eval 脚本与 rollout 共用同一 `build_prompt` + generate 参数常量（§2.f 已要求）。
10. **数据许可证/溯源**：agent_rl_math 出自仓库作者的 ModelScope 数据集（仓库内无独立 LICENSE 说明，README L429 提到数据来源协议约束）→ fork 内使用与展示需随 README 数据章节的协议口径走，自造部分在 report 里标注确定性模板生成方式，简历不夸大数据量。
11. **长度统计口径**：avg_len 用 completion_mask（eos 前）而非 decode 文本长度（L158），与 verifier 文本长度可能差一个 eos——曲线注释写清口径，避免自相矛盾。
12. **DPO B 支线数据依赖**：guide §4.3 B 支线要"RL rollout 存档做错对配对"，依赖 d10 的 `samples.jsonl` 从第一步就含 gt/verdict —— 已纳入 d10 设计；若 B 支线不做，该 jsonl 仍服务人工抽查。

---

*本规格由 coding 子 agent 于 2025-09 基于仓库实测行号产出；与 guide 不一致处已在 §1 表格"判定"列显式标注（1.1c 行数估计、1.2c eos 差异实义、1.4e RM 路径语义）。*
