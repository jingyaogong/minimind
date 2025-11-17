## 目标
- 在现有 DPO/PPO 基础上，新增 GRPO 与 SPO 训练类型的完整前后端流程。
- 选定对应训练类型后，仅显示该类型的特有参数；隐藏不需要的参数。
- 参数默认值严格对齐 `trainer/train_grpo.py` 与 `trainer/train_spo.py`。
- 保持原有 `trainer_web` 逻辑与交互不受影响。

## 代码改动
### 前端模板（index.html）
- 在训练类型选择框中新增：`RL - GRPO`（value=`grpo`）、`RL - SPO`（value=`spo`）。
- 新增两块参数卡片：
  - GRPO 参数卡（class=`parameter-card grpo`）：
    - `beta`（浮点输入，placeholder=0.02）
    - `num_generations`（整数输入，placeholder=8）
    - `reasoning`（下拉 0/1）
    - `reward_model_path`（文本输入，placeholder=`../../internlm2-1_8b-reward`）
  - SPO 参数卡（class=`parameter-card spo`）：
    - `beta`（浮点输入，placeholder=0.02）
    - `reasoning`（下拉 0/1）
    - `reward_model_path`（文本输入，placeholder=`../../internlm2-1_8b-reward`）
- 保持现有 DPO/PPO 参数卡结构不变。

### 前端交互（static/js/train/form.js）
- 在 `onTrainTypeChange` 中：
  - 为 GRPO/SPO 添加显示/隐藏逻辑：
    - `.grpo` 与 `.spo` 卡片只在对应训练类型显示。
    - `pretrain-sft` 区域显示范围扩展到 `grpo` 与 `spo`（需要 `save_weight` 字段）。
    - `from-weight` 区域在 `ppo/grpo/spo` 下隐藏（这三者均不使用 `from_weight`）。
  - 新增默认参数设定：
    - GRPO 默认值（严格对齐 `trainer/train_grpo.py`）：
      - `save_dir=../out`, `save_weight=grpo`, `epochs=1`, `batch_size=2`, `learning_rate=8e-8`
      - `data_path=../dataset/rlaif-mini.jsonl`, `log_interval=1`, `save_interval=10`
      - `hidden_size=512`, `num_hidden_layers=8`, `max_seq_len=66`, `use_moe=0`
      - 特有：`beta=0.02`, `num_generations=8`, `reasoning=1`, `reward_model_path=../../internlm2-1_8b-reward`
    - SPO 默认值（严格对齐 `trainer/train_spo.py`）：
      - `save_dir=../out`, `save_weight=spo`, `epochs=1`, `batch_size=2`, `learning_rate=1e-7`
      - `data_path=../dataset/rlaif-mini.jsonl`, `log_interval=1`, `save_interval=10`
      - `hidden_size=512`, `num_hidden_layers=8`, `max_seq_len=66`, `use_moe=0`
      - 特有：`beta=0.02`, `reasoning=1`, `reward_model_path=../../internlm2-1_8b-reward`
- 保持 GPU 选择与提交逻辑不变。

### 后端启动器（trainer_web/train_web_ui.py）
- 在 `start_training_process` 中新增 `train_type=='grpo'` 与 `train_type=='spo'` 分支：
  - `grpo` 使用脚本 `../trainer/train_grpo.py`；显式拼接：`--beta`、`--num_generations`、`--reasoning`、`--reward_model_path`（若前端提供）。
  - `spo` 使用脚本 `../trainer/train_spo.py`；显式拼接：`--beta`、`--reasoning`、`--reward_model_path`（若前端提供）。
- 在通用参数拼接的跳过列表中新增：`'num_generations','beta','reasoning','reward_model_path'`（防止重复添加），并与 DPO/PPO 现有跳过策略一致；同时对 `grpo/spo` 也跳过 `from_weight`。
- 其余逻辑（多卡、日志、监控）保持不变。

### 进程显示名称
- `static/js/processes/list.js` 的名称映射中补充：`ppo: 'PPO', grpo: 'GRPO', spo: 'SPO'`（`logfiles/list.js` 已有，无需改动）。

## 验证步骤
- 前端：切换训练类型时仅显示对应参数卡；`from_weight` 在 `ppo/grpo/spo` 下隐藏；默认值填充正确。
- 后端：提交后查看日志首行的命令拼接，确认包含对应特有参数：
  - GRPO：`--beta 0.02 --num_generations 8 --reasoning 1 --reward_model_path ../../internlm2-1_8b-reward`
  - SPO：`--beta 0.02 --reasoning 1 --reward_model_path ../../internlm2-1_8b-reward`
- 保证 DPO/PPO 流程未受影响：分别发起一次 DPO/PPO 以回归测试。
- 多卡/单卡/CPU 选择与日志查看、进程管理均保持工作。

## 兼容性与风险控制
- 所有改动均为增量更新，不修改既有字段与 API 形态。
- 默认值与训练脚本对齐；占位符仅用于引导，实际以默认填充值为准。
- 若未提供特有参数，后端不拼接该 flag，训练脚本使用自身默认值。

## 交付
- 更新上述 4 处文件：`templates/index.html`、`static/js/train/form.js`、`trainer_web/train_web_ui.py`、`static/js/processes/list.js`。
- 提供端到端验证记录与示例日志片段，确保功能闭环。