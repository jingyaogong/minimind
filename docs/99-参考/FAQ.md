# 99-参考 / FAQ 常见问题

> **TL;DR**：按"现象 → 原因 → 解决"组织，覆盖环境、训练、数据、推理、模型 5 大类高频问题。
> 排错顺序：先看本篇 → [术语表](./术语表.md) → [主 README](../../README.md) → [Issues](https://github.com/jingyaogong/minimind/issues)。

---

## 📦 一、环境 / 安装类

### Q1 `ImportError: cannot import name X from 'transformers'`

- **现象**：`from transformers.modeling_outputs import MoeCausalLMOutputWithPast` 等导入失败。
- **原因**：transformers 版本过低（minimind-3 需要 ≥ 4.57，对齐 Qwen3 结构）。
- **解决**：
  ```bash
  pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
  # 或单独升级
  pip install transformers==4.57.6
  ```
  见 `requirements.txt:21`，版本已被锁定。

### Q2 `RuntimeError: CUDA out of memory`（训练时）

- **现象**：训练几步后报 OOM。
- **原因**：默认 `batch_size` 对小显存卡偏大；或 `max_seq_len` 过长。
- **解决**：按优先级试：
  ```bash
  # 1. 缩小 batch_size，配合梯度累积保持有效 batch 不变
  python train_pretrain.py --batch_size 8 --accumulation_steps 4
  # 2. 缩短序列长度
  python train_pretrain.py --max_seq_len 256
  # 3. MoE 比 dense 显存略高，单卡可改 dense：--use_moe 0
  ```
  另：训练默认 `dtype=bfloat16`（AMP 混合精度），无需手动开 amp。

### Q3 swanlab 连不上 / 训练卡在 `swanlab.init`

- **现象**：加 `--use_wandb` 后训练卡住或报网络错误。
- **原因**：未登录 swanlab，或网络不通。
- **解决**：
  ```bash
  swanlab login     # 注册账号拿 API key，免费
  # 实在连不上就关掉，不影响训练
  python train_pretrain.py   # 不加 --use_wandb 即可
  ```
  swanlab 国内可直连，无需梯子。见 `requirements.txt:29`。

### Q4 Mac MPS 部分算子不支持

- **现象**：MPS 上跑 `train_dpo.py` / `train_grpo.py` 报 `RuntimeError: ... not implemented for MPS`。
- **原因**：MPS 后端对部分 PyTorch 算子（如某些 scatter/index 操作、Reward Model 计算）支持不全。
- **解决**：
  - pretrain / SFT / 推理：MPS 一般能跑，慢但可用；
  - DPO / PPO / GRPO / Agent：**强烈建议用 CUDA GPU**，远程租 3090 即可。
  - 临时方案：`--device cpu`（极慢，仅作冒烟测试）。

### Q5 `torch.cuda.is_available()` 返回 `False`

- **现象**：装完 PyTorch 检测不到 GPU。
- **原因**：torch 版本与 CUDA 驱动不匹配。`requirements.txt` 中 torch 是注释掉的（需按 CUDA 版本手动装）。
- **解决**：
  ```bash
  nvidia-smi       # 看 CUDA Driver 版本
  # 例：CUDA 12.1 → 装对应 torch
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
  验证：`python -c "import torch; print(torch.cuda.is_available())"` 应输出 `True`。

---

## 🏋️ 二、训练类

### Q6 loss 不下降 / 出现 NaN

- **现象**：pretrain loss 卡在 10+ 不动，或突然变 NaN。
- **原因**：
  1. 学习率过大（pretrain 默认 `5e-4`，对某些初始化偏激进）；
  2. 没有从 pretrain 权重起步直接跑 SFT/RL；
  3. MoE 路由崩溃导致 aux_loss 爆炸；
  4. 数据中混入超长样本导致数值溢出。
- **解决**：
  ```bash
  # 1. 调小 lr
  python train_pretrain.py --learning_rate 2e-4
  # 2. SFT 必须基于 pretrain 权重：train_full_sft.py 默认 from_weight=pretrain，勿改成 none
  # 3. NaN 时加 grad_clip（默认已 1.0），可调更小
  python train_pretrain.py --grad_clip 0.5
  ```
  见 `trainer/train_pretrain.py` 默认 lr=5e-4，`trainer/trainer_utils.py:40` cosine 调度。

### Q7 训练很慢（低于预期 step/s）

- **现象**：单卡 3090 跑 pretrain，预期 ~2h，实际远超。
- **原因**：
  1. Flash Attention 没启用（seq_len=1 时不走 flash 分支，见 `model/model_minimind.py:125`）；
  2. `num_workers=0`（默认 8，但 Windows 上多进程可能失效）；
  3. 数据在 CPU 上做了过多处理；
  4. MoE 配置（`--use_moe 1`）原生实现比 dense 慢约 50%（README 已说明）。
- **解决**：
  ```bash
  # 1. 确认 PyTorch ≥ 2.0（sdpa 可用）
  python -c "import torch; print(torch.__version__)"
  # 2. Linux 上保持 --num_workers 8；Windows 改 0
  python train_pretrain.py --num_workers 8
  # 3. 试 torch.compile（实验性）
  python train_pretrain.py --use_compile 1
  ```

### Q8 显存不足（OOM 之外的"快满"状态）

- **现象**：nvidia-smi 显示显存接近上限，训练偶发卡顿。
- **原因**：KV cache、optimizer state、activation 都占显存。
- **解决**：
  ```bash
  # 1. 减 batch_size + 加 accumulation_steps（保持等效 batch）
  python train_pretrain.py --batch_size 16 --accumulation_steps 2
  # 2. 短 max_seq_len
  # 3. RL 阶段（PPO/GRPO）显存占用最大，因为要同时放 actor + ref + reward model
  #    GRPO 单网络更省，PPO 双网络（actor+critic）显存约 1.5~2x
  ```

### Q9 DDP 多卡报错 `RuntimeError: ... NCCL ...`

- **现象**：`torchrun --nproc_per_node 4 train_pretrain.py` 报 NCCL 错误。
- **原因**：
  1. 多卡之间 CUDA 版本不一致（很少见）；
  2. 端口被占用；
  3. 没有用 `torchrun` 而是用 `python` 直接启动多卡。
- **解决**：
  ```bash
  # 正确启动方式（nproc_per_node = GPU 数）
  torchrun --nproc_per_node 4 trainer/train_pretrain.py
  # 排查端口
  export NCCL_DEBUG=INFO
  # 单机多卡无需 master_port；如遇冲突：
  torchrun --nproc_per_node 4 --master_port 29501 trainer/train_pretrain.py
  ```
  DDP 初始化见 `trainer/trainer_utils.py:44`。

### Q10 断点续训怎么用

- **现象**：训练中断后想接着训。
- **解决**：所有训练脚本都支持 `--from_resume 1`：
  ```bash
  python train_pretrain.py --from_resume 1
  python train_full_sft.py --from_resume 1
  python train_dpo.py --from_resume 1
  ```
  检查点保存在 `./checkpoints/<weight>_<dim>_resume.pth`，含模型/优化器/epoch/step/wandb_id。**支持跨 GPU 数量恢复**（自动按比例调整 step）。见 `trainer/trainer_utils.py:63` 的 `lm_checkpoint`。

### Q11 checkpoint 损坏 / 加载报错

- **现象**：`RuntimeError: ... size mismatch ...` 或 `KeyError`。
- **原因**：
  1. 改了 `hidden_size` / `num_hidden_layers` 后用旧 checkpoint；
  2. `.pth` 写入时被中断产生半截文件；
  3. 切换 MoE ↔ dense 后权重名不匹配。
- **解决**：
  ```bash
  # 1. 删除对应 resume 文件，从头开始（或换 weight 名）
  rm checkpoints/full_sft_768_resume.pth
  # 2. 写入是原子操作（os.replace，见 trainer_utils.py:76,104），若仍损坏多半是改了结构
  # 3. MoE 权重文件名带 _moe 后缀，dense 不带，勿混用
  ```
  另：`init_model` 用 `strict=False` 加载（`trainer_utils.py:127`），缺失键不报错但会随机初始化。

---

## 📂 三、数据类

### Q12 数据下载慢 / 失败

- **现象**：从 HuggingFace 拉数据集极慢或超时。
- **解决**：用 ModelScope 镜像（国内友好）：
  ```bash
  # 安装
  pip install modelscope
  # 下载（推荐）
  modelscope download --dataset gongjy/minimind_dataset --local_dir ./dataset
  # 或在 HF 上单独下需要的文件（无需全 clone）
  ```
  最小复现只需 `pretrain_t2t_mini.jsonl` + `sft_t2t_mini.jsonl`。

### Q13 `jsonl` 格式错误 / Dataset 加载报错

- **现象**：`load_dataset` 报 `Features` 不匹配或 JSON parse error。
- **原因**：自己造数据时字段缺失或类型不一致。
- **解决**：
  - Pretrain 数据必须每行 `{"text": "..."}`
  - SFT 数据必须每行 `{"conversations": [{"role": "...", "content": "..."}]}`
  - DPO 数据必须每行 `{"chosen": [...], "rejected": [...]}`
  - agent_rl 多一个 `gt` 字段
  ```bash
  # 验证 jsonl 格式
  python -c "
  import json
  for i, line in enumerate(open('your.jsonl')):
      try: json.loads(line)
      except Exception as e: print(f'Line {i}: {e}')
  "
  ```
  字段 schema 见 `dataset/lm_dataset.py:63`（SFT 用 Features 显式声明）。

### Q14 `max_seq_len` 怎么选

- **经验值**（来自 README）：

| 数据集 | 推荐 max_seq_len |
|---|---|
| `pretrain_t2t_mini.jsonl` | 768 |
| `pretrain_t2t.jsonl` | 380 |
| `sft_t2t_mini.jsonl` | 768 |
| `sft_t2t.jsonl` | 1024+ |
| `dpo.jsonl` | 1024（默认） |
| `rlaif.jsonl` | 768（GRPO 默认） |

- **取舍**：太长 → padding 浪费算力；太短 → 截断丢语义。中文约 1.5~1.7 字符/token。
- 训练参数 `--max_seq_len` 指 token 数（不是字符数）。

### Q15 自己的数据怎么转成 minimind 格式

- **Pretrain**：纯文本即可
  ```jsonl
  {"text": "你的预训练文本1"}
  {"text": "你的预训练文本2"}
  ```
- **SFT**：多轮对话（最少 1 轮）
  ```jsonl
  {"conversations": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]}
  ```
- **LoRA 垂域**：和 SFT 同格式，只是文件名 `lora_xxx.jsonl`，放到 `dataset/` 下。
- **DPO**：成对的 chosen/rejected
  ```jsonl
  {"chosen": [{"role":"user","content":"Q"}, {"role":"assistant","content":"好回答"}],
   "rejected": [{"role":"user","content":"Q"}, {"role":"assistant","content":"差回答"}]}
  ```
- 数据加载逻辑见 `dataset/lm_dataset.py:37`（Pretrain）/ `:58`（SFT）/ `:122`（DPO）/ `:195`（RLAIF）。

---

## 🗣️ 四、推理类

### Q16 输出乱码 / 全是空格

- **现象**：`eval_llm.py` 跑出来全是 `<unk>` 或乱字符。
- **原因**：
  1. 加载了 `pretrain` 权重当对话模型用（pretrain 只会接龙，不会对话）；
  2. 没经过 SFT 直接推理；
  3. tokenizer 路径不对。
- **解决**：
  ```bash
  # 对话必须用 full_sft 之后的权重
  python eval_llm.py --weight full_sft
  # 不要这样（pretrain 只会接龙）
  python eval_llm.py --weight pretrain
  ```

### Q17 推理慢

- **现象**：单条回答要十几秒。
- **原因**：原生 `generate` 是 PyTorch eager 模式，未做 kernel fusion。
- **解决**：
  ```bash
  # 1. 用 vllm（最快）
  vllm serve /path/to/minimind-3 --served-model-name minimind
  # 2. 用 ollama
  ollama run jingyaogong/minimind-3
  # 3. 限长：--max_new_tokens 512
  python eval_llm.py --max_new_tokens 512
  ```
  推理时 KV cache 自动启用（`model/model_minimind.py:120-122`），无需手动。

### Q18 重复输出 / 死循环同一句话

- **现象**：模型不断重复同一短语直到 max_new_tokens。
- **原因**：温度过低 / `repetition_penalty=1.0`（不惩罚）/ 小模型本身能力弱。
- **解决**：
  ```bash
  # 调温度 + 加重复惩罚
  python eval_llm.py --temperature 0.85 --top_p 0.95
  # 代码里默认 repetition_penalty=1.0，可改源码（model_minimind.py:257）
  # 或在 eval_llm.py 调用时覆盖
  ```
  采样参数说明见 `model/model_minimind.py:257-288`。

### Q19 `--open_thinking 1` 没效果

- **现象**：开了思考开关但模型直接出答案，没有思考过程。
- **原因**：
  1. 用的是 `pretrain` 权重（未学过思考标签）；
  2. 用的是早期版本权重（thinking 能力是 2026-03 后并入主线 SFT 的）；
  3. 同时开启 Tool Call + thinking 时模型不稳定（README 明确说明）。
- **解决**：
  ```bash
  # 必须用 minimind-3 主线 full_sft 之后的权重
  python eval_llm.py --load_from ./minimind-3 --open_thinking 1
  ```
  thinking 标签是 chat_template 注入的（`model/tokenizer_config.json` 的 chat_template 字段），训练时按 `thinking_ratio` 采样。

### Q20 多轮对话上下文丢失

- **现象**：第二轮问"那它呢"，模型不知道"它"指什么。
- **原因**：默认 `--historys 0`，不携带历史。
- **解决**：
  ```bash
  # 携带最近 2 轮（必须偶数）
  python eval_llm.py --historys 2
  python eval_llm.py --historys 4
  ```
  `historys` 是 token 数为单位的轮数（偶数），见 `eval_llm.py:46`。太大会爆显存。

---

## 🧠 五、模型类

### Q21 怎么换模型大小

- **修改 `model/model_minimind.py:10` 的 `MiniMindConfig`**，或训练时传参：
  ```bash
  # 更小（26M 风格）
  python train_pretrain.py --hidden_size 512 --num_hidden_layers 8
  # 更大（104M 风格）
  python train_pretrain.py --hidden_size 768 --num_hidden_layers 16
  ```
- README 的"模型配置"章节有详细对照表（dim/layers 取舍）。
- 小模型经验：`d_model < 512` 容易模式崩溃；`d_model > 1536` 加层比加宽划算。

### Q22 MoE 比 dense 还慢

- **现象**：`--use_moe 1` 后训练 step/s 明显下降。
- **原因**：minimind 用原生 PyTorch 实现 MoE（无 kernel fusion），token 按专家分桶后 forward 串行。
- **解决**：
  - 这是已知设计取舍（README "结构"章节有说明）；
  - 4 experts / top-1 比 dense 慢约 50%，已是甜点配置；
  - 想更快：用 DeepSpeed-MoE / Triton 自定义 kernel，或换 dense；
  - 路由 + aux_loss 见 `model/model_minimind.py:148-176`。

### Q23 词表能不能改

- **能改但不建议**。改词表会引发连锁：
  - embedding 层 / lm_head 参数变化（`model_minimind.py:201,241`）；
  - 已有 `.pth` 权重无法直接加载（vocab_size 不匹配）；
  - 与 transformers / vllm / ollama 生态对齐失效。
- 当前 6400 词表是有意为之的取舍（README "Tokenizer" 章节）：压缩 embedding 占比。
- 训练 tokenizer 的脚本在 `trainer/train_tokenizer.py`，仅供学习参考。

### Q24 怎么导出到 HuggingFace 格式

- **用 `scripts/convert_model.py`**：
  ```bash
  cd scripts && python convert_model.py
  ```
- 该脚本还支持：
  - **LoRA 权重合并**：`convert_merge_base_lora`，把基模 + LoRA 合成新完整权重；
  - 输出的 transformers 格式可被 `vllm serve` / `ollama run` / `llama.cpp` 直接加载。
- 评估 `--load_from ./minimind-3` 时即加载 transformers 格式（见 `eval_llm.py:34`）。

---

## 🔧 通用排查清单

遇到任何问题先按这 5 步走：

1. **路径**：在 `trainer/` 还是 `scripts/` 下执行？相对路径是否正确？
2. **权重**：`out/` 下有对应 `.pth` 文件吗？文件名是否与 `--weight` 一致？
3. **数据**：`dataset/` 下有对应 `.jsonl` 吗？路径与 `--data_path` 一致？
4. **GPU**：`nvidia-smi` 看显存；`torch.cuda.is_available()` 看 torch 是否识别。
5. **版本**：`pip list | grep -E "torch|transformers|swanlab"` 看是否满足 `requirements.txt`。

> 卡 30 分钟没解决？去 [minimind Issues](https://github.com/jingyaogong/minimind/issues) 搜关键词，社区很活跃。
