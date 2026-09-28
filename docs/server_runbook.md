# 服务器跑通手册（SOP 第 3 步：真实数据链路）

目标：在租来的单卡机器上跑出**两个真实 run + 两次 held-out 评估 + 两个归档目录 + 一份汇总报告**，
全程可复现、可审计。本文档每一步都给出「命令 / 期望输出 / 失败怎么办」。

配套：`scripts/server_preflight.py`（开机器自检）、`scripts/archive_run.py`（§13 归档）、
`scripts/summarize_runs.py`（§10.3/§11 汇总）。数据与登记表在本地已经做好，不需要在服务器上重做。

---

## 0. 这一步的产出清单与时间预算

| 产出 | 路径 | 大约耗时（24GB 单卡） |
|---|---|---|
| SFT 冷启动权重（CoT v2） | `out/math_cold_cot_seed42_768.pth` | 5–15 分钟（250 step） |
| SFT 格式抽检 | `runs/math_cold_cot_seed42/sft_format_check.json` | 2–5 分钟（20 题 greedy） |
| SFT held-out 抽检（看有没有算术提升） | `runs/math_cold_cot_seed42/eval.json` | 10–30 分钟（400×16） |
| 难度标定复测（决定 RL 数据难度） | `runs/probe/calibration_cot_v2.json` | 3–8 分钟（120 题 × 8 采样） |
| rule 臂权重 | `out/mini_rule_seed42_768.pth` | 1.5–2.5 小时（1000 step） |
| rule 臂 held-out 评估 | `runs/mini_rule_seed42/eval.json` | 10–30 分钟（400×17） |
| 归档目录 | `runs/math_cold_cot_seed42/`、`runs/mini_rule_seed42/` | 秒级 |
| 汇总报告 | `reports/summary.md` + `figures/*.png` | 秒级 |

先跑通这条链路（约 2–3 GPU 小时），确认数字合理，再一次性把剩下 5 个臂排队跑完。
**§6.2 的难度标定复测是进入 GRPO 前的一道闸门**：全档 pass@1 < 0.05 就别开 RL，先停下来讨论。
**中途换机器/断线都不怕**：`out/` 里有权重、`runs/` 里有归档，接着跑就行。

---

## 1. 机器规格（别省钱省在这里）

- 单卡显存 ≥ 24GB：RTX 4090 / A10 / L40S / A100 都可以；**T4 16GB 别用来跑 GRPO**（512 长度 × G=6 容易 OOM 且慢）。
- CUDA ≥ 12.1，Ubuntu 20.04/22.04，**磁盘空闲 ≥ 20GB**（本项目实测总占用约 10GB，预算明细见 §5.1；
  AutoDL 数据盘 50GB 足够，别用系统盘）。
- 不需要多卡：脚本支持 DDP，但 64M 模型单卡足够，多卡只会增加变量。

---

### 1.1 租机器时怎么选镜像（框架/版本/Python/CUDA）

| 选项 | 选择 | 判据 |
|---|---|---|
| 框架名称 | PyTorch | 全链路只依赖 torch，不需要 TF/JAX |
| 框架版本 | **2.12.1**（首选，就是我们本地验证过的版本）或 **2.8.0**（最接近 MiniMind 注释里的 2.6.0） | 2.5.1 及更旧别选；1.10/1.11 直接排除（与 `datasets 3.6 / transformers 4.57` 组合容易出 ABI 问题）。已确认 `torch.cuda.amp.autocast`（训练脚本在用）在 2.12 上仍然可用，只是有弃用告警 |
| Python 版本 | **3.11**（3.12 也可以） | 仓库锁了 `numpy==1.26.4`，它只提供 3.9–3.12 的 wheel；**3.13 上 pip 会去源码编译 numpy 并失败**（本机 3.13 能用是因为 Anaconda 自带打包版） |
| CUDA 版本 | **12.4**；平台若只给 12.1 / 12.6 / 12.8，就跟该 torch 镜像配套的那个 | cu12x 覆盖 sm_86/89（3090/4090/A10/L40S）与 sm_90（A100/H100）；只有 Blackwell（5090 / RTX PRO 6000）才必须 12.8 |

两条平台经验：

1. **镜像里的 torch 和 CUDA 是配套的，别跨配**，装依赖时也不要再 `pip install torch`（会重装/换版本），只装 runbook §3 里那四个包。
2. 镜像的 Python 版本经常只有 3.10–3.13 可选；如果只能选到 3.13，用逃生方案在镜像内部建一个干净环境：

```bash
conda create -n minimind python=3.11 -y && conda activate minimind
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "transformers==4.57.6" "datasets==3.6.0" "numpy==1.26.4" matplotlib pyyaml modelscope
nvidia-smi                       # 驱动版本要 ≥ 该 CUDA runtime 的要求（平台镜像一般已配好）
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### 1.2 路径约定（AutoDL）与「无卡模式」

- **代码与数据统一放数据盘**：下文写的 `/data/minimind`，在 AutoDL 上就是 `/root/autodl-tmp/minimind`
  （数据盘 50GB、关机不丢、I/O 快；系统盘只有 30GB，还要留给环境）。
- **`nvidia-smi` 报 `No devices were found` 不是故障**：如果实例信息写着 `0.5 核心 / 2GB / GPU: No devices were found`，
  说明现在开的是**无卡模式**（AutoDL 的廉价 CPU 模式）。它专门用来传代码、装依赖、下权重；
  **训练前必须关机 → 用「有卡模式」重开**，重开后 `/root` 与 `/root/autodl-tmp` 里的东西都还在。
- 无卡模式只有 0.5 核 / 2GB，`pip install` 会慢一些，属正常；若 pip 中途 OOM，就先切有卡模式再装依赖。

---

## 2. 把本地仓库传上去

**先记住：服务器是另一台机器，本地路径在它那里不存在。** 在服务器上 `cd /Users/qi/…` 必然报
`No such file or directory`——这不是权限问题，是**文件还没传过去**。上传 = 用 SSH 把本地文件复制一份到服务器；
传完后是**两份独立副本**，本地改代码不会自动同步（见 §2.4）。

### 2.0 拿到登录信息，先建一个 ssh 别名（只做一次）

AutoDL 控制台 → 你的实例 → 「登录指令」，形如 `ssh -p 17233 root@connect.nmb1.seetacloud.com`。

**AutoDL 每次关机重开，端口号和主机名都会变**，如果每条命令都手敲 `-p <端口> root@<主机>`，
改一次就要全文替换一次。所以先在本地 `~/.ssh/config` 里写一个固定别名 `autodl`：

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
cat >> ~/.ssh/config <<'EOF'

Host autodl
  HostName connect.nmb1.seetacloud.com   # ← 换成控制台当前的主机名
  Port 17233                             # ← 换成控制台当前的端口
  User root
  ServerAliveInterval 30
EOF
chmod 600 ~/.ssh/config
ssh autodl 'pwd; df -h /root/autodl-tmp'
```

- 以后**机器重开只需要改这个文件里的两行**，本文档下面所有命令都不用动。
- `ServerAliveInterval 30` 让长命令（训练、评估）在 ssh 静默时也保活，少掉线。
- 期望输出：`/root` + 数据盘容量（约 50G）。**这一步通了才谈得上上传。**
- 密码方式登录：首次会问 `yes` 和密码。想免密可以再跑 `ssh-copy-id autodl`（可选）。

> 如果 `~/.ssh/config` 里已经有一个 `Host autodl`，先删掉旧的那段再追加，否则 SSH 用第一个匹配到的。

### 2.1 上传：`tar | ssh`（macOS 首选）

macOS 15 起自带的 `rsync` 是 openrsync（`--progress` 等参数会报错），所以默认用 `tar` 管道。本地执行：

```bash
cd /Users/qi/实习/项目/minimind
tar czf - --no-xattrs \
  --exclude='*.pth' --exclude='__pycache__' --exclude='.DS_Store' \
  --exclude='.ipynb_checkpoints' --exclude='logs' --exclude='checkpoints' \
  --exclude='out' --exclude='reports' --exclude='website' --exclude='images' \
  . | ssh autodl \
      'mkdir -p /root/autodl-tmp/minimind && tar xzf - -C /root/autodl-tmp/minimind'
```

- 整包压缩后约 **0.6MB / 89 个文件**，几秒传完；`dataset/ docs/ scripts/ trainer/ tests/ configs/ planning/` 都会上去。
  （含 `dataset/probe/calibration_probe.jsonl`，§6.2 的难度标定要用。）
- 排除项是权重与运行时产物（`*.pth`、`out/`、`logs/`、`checkpoints/`、`reports/`、`website/`、`images/`）。
- `*.pth` 不传：官方权重在服务器上直接下载（§4），你自己的权重在服务器上产生。
- 数据必须在上传这一刻就带过去，**不要在服务器上重新生成**（会改 sha256，和本地 manifest 对不上）。
- `--no-xattrs` 用来关掉 macOS 扩展属性；不加也能传成功，只是服务器端 GNU tar 会为每个文件刷一行
  `Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.quarantine'`——**那是警告不是错误**。

### 2.2 备选：GNU rsync

只有 `rsync --version | head -1` 显示 `rsync version 3.x`（而不是 `openrsync`）时才用：

```bash
cd /Users/qi/实习/项目/minimind
rsync -avz --progress -e ssh \
  --exclude '__pycache__' --exclude '.ipynb_checkpoints' --exclude '.DS_Store' \
  --exclude '*.pth' --exclude 'logs/' --exclude 'checkpoints/' --exclude 'out/' \
  --exclude 'reports/' --exclude 'website/' --exclude 'images/' \
  ./ autodl:/root/autodl-tmp/minimind/
```

### 2.3 上传后立刻校验（必做）

```bash
ssh autodl \
  'cd /root/autodl-tmp/minimind && du -sh . && sha256sum dataset/math_*.jsonl'
```

期望三个数据文件的哈希与本地一致：
`math_sft_train` → `1740c0d038f4…`、`math_rlvr_train` → `f55c0b6954ea…`、`math_heldout` → `f829eb084a3e…`。
（`a9771691882e…` 是 pilot 那一版模板 CoT 的 `math_sft_train` 哈希，已经作废但保留在文档里备查。）
对不上就重传，别在服务器上重新生成数据——评估脚本的 `data_matches_manifest` 就是靠这个哈希兜底。

### 2.4 以后本地改了代码怎么同步

再跑一遍 §2.1 的同一条命令即可（0.5MB，秒级）。tar 只覆盖包里有的文件，
服务器上已经生成的 `out/*.pth`、`runs/`、`logs/`、`checkpoints/` 不在包里，不会被删掉。

---

## 3. 环境（5 分钟）

```bash
python -V                                                   # 期望 3.12.x，来自 /root/miniconda3/bin/python
python -c "import torch;print(torch.__version__, torch.version.cuda)"
python -c "import numpy;print(numpy.__version__)"
pip list | grep -iE "^(torch|numpy|matplotlib|transformers|datasets|modelscope|pyyaml)"
```

AutoDL 的 PyTorch 镜像已经把 torch / numpy / matplotlib 装好了。本项目在 `PyTorch 2.8.0 / Python 3.12 / CUDA 12.8`
镜像实测为 `torch 2.8.0+cu128`、`numpy 2.3.2`、`matplotlib 3.10.5`——**这些统统不要动，也不要再 `pip install torch`**
（重装会换掉与驱动配套的 CUDA 版本）。只补缺的那几个：

```bash
pip install "transformers==4.57.6" "datasets==3.6.0" pyyaml modelscope
```

为什么不用全量 `requirements.txt`：它包含 streamlit / ngrok / sentence_transformers 等与训练无关的重包，
装它们只会白等几分钟并引入版本冲突。

验证（无卡模式下 `is_available()` 必然是 `False`，这里只确认 import 与版本；真正的 GPU 检查在 §5）：

```bash
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
# 无卡模式期望：2.8.0+cu128 False 12.8
```

### 3.1 关于 numpy：**不要降级到 1.26.4**（与 requirements.txt 不一致，是有意为之）

`requirements.txt` 锁的 `numpy==1.26.4` 是上游为老环境写的，本链路不需要：

- 全仓库对 numpy 的使用只有 `np.random.seed(seed)`（`trainer/trainer_utils.py:56`），
  没有任何 numpy 2.0 已删除的 API（`np.float_` / `np.NaN` / `np.in1d` / `.ptp()` / `newbyteorder` 全查过：无）。
- 镜像里的 `torch 2.8.0+cu128` 是**按 numpy 2 编译**的，把 numpy 降到 1.x 才是风险方向。
- `datasets 3.6.0` / `transformers 4.57.6` / `matplotlib 3.10.5` 都支持 numpy 2。

万一后面真的撞上 numpy 相关的 ABI 报错，再降级重试（那时告诉我，不要自己猜）：

```bash
pip install "numpy==1.26.4"    # 仅在确认需要时执行
```

### 3.2 逃生方案：镜像自带的 torch 不可用时

只有 §3 第一步里 torch 缺失、或版本 < 2.5 时才建独立环境（别在镜像正常时多此一举）：

```bash
conda create -n minimind python=3.11 -y && conda activate minimind

# 1) 先装 torch（按服务器的 CUDA 版本选索引）
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2) 再装本链路真正需要的包（不要装全量 requirements.txt）
pip install "transformers==4.57.6" "datasets==3.6.0" matplotlib pyyaml modelscope
```

如果 home 盘小（HF 的 `datasets` 缓存默认写在 `~/.cache/huggingface`）：

```bash
mkdir -p /root/autodl-tmp/hf && echo 'export HF_HOME=/root/autodl-tmp/hf' >> ~/.bashrc && source ~/.bashrc
```

`dataset/lm_dataset.py` 通过 `datasets.load_dataset("json", ...)` 读 SFT 数据，
所以这条缓存路径必须可写、且有空间。

### 3.3 在服务器上跑单测（零 GPU 成本的环境体检，推荐）

```bash
pip install pytest
cd /root/autodl-tmp/minimind && python -m pytest tests/ -q
```

| 环境 | 期望输出 | 说明 |
|---|---|---|
| 本地 macOS（用户 `qi`） | `251 passed` | 全绿 |
| 服务器（**root**） | `250 passed, 1 skipped` | 跳过的是 `test_server_preflight.py::test_read_only_directory_fails` |

为什么服务器上会 skip 一个：那个用例用 `chmod 0o500` 造「只读目录」再期待写入失败，
而 root 有 `CAP_DAC_OVERRIDE`，会**绕过文件权限位**照样写进去。这是同一份代码在不同 uid 下的合法差异，
**不是环境坏了**。`check_writable` 本身没有因此变弱：它真去建一个探针文件再删掉，
真正的只读挂载（root 也写不进去）照样会 FAIL，而 FAIL 分支由与 uid 无关的
`test_writable_fails_when_parent_is_a_file` 覆盖。

---

## 4. 下载官方权重（唯一需要联网的重活）

训练脚本 `init_model()` 按 `../out/{from_weight}_{hidden_size}.pth` 找权重，所以最终文件必须叫
`out/full_sft_768.pth`。流程是：下载 → `find` 看清文件名 → 复制进 `out/` → **校验 key 能对上**（第 4.3 步，别跳）。

### 4.1 下载

```bash
pip install -U modelscope                 # 服务器上没装才需要
mkdir -p /root/autodl-tmp/models && cd /root/autodl-tmp/models

# --local_dir 一定要给：不给会写进 ~/.cache/modelscope（SOP §2.2 要求权重放数据盘）
modelscope download --model gongjy/minimind-3-pytorch --local_dir ./minimind-3-pytorch
```

常用可选参数（`modelscope download --help` 可查）：

| 参数 | 用途 |
|---|---|
| `--revision v1.0` | 锁版本；报告里要写清用的是哪个 revision |
| `--include "*.pth" "*.json"` | 只拉需要的文件，省时间和磁盘 |
| `--max-workers 4` | 公司网络限速时降并发 |
| 重复执行 `--local_dir` | 已存在且完整的文件会跳过，**断线重跑就是续传** |

如果报 repo 不存在，换名字再试（MiniMind 的 release 命名变过，两个都试一遍）：

```bash
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
# 也可以先开网页看文件列表再决定：
# https://www.modelscope.cn/models/gongjy/minimind-3-pytorch/files
```

HF 镜像备选：

```bash
pip install -U "huggingface_hub[cli]"
HF_ENDPOINT=https://hf-mirror.com hf download jingyaogong/minimind-3-pytorch \
  --local-dir ./minimind-3-pytorch
# 或 git-lfs（先跳过 smudge，避免一次拉全部大文件）
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/jingyaogong/minimind-3-pytorch
```

这一步只有 `full_sft_768.pth` 是必需的；官方 `rlaif.jsonl` / `sft_t2t_mini.jsonl`
是「改造前后回归护栏」用的，可选，不影响主链路。

### 4.2 find：把文件列表看清楚

```bash
# 1) 全部文件（先确认目录结构）
find ./minimind-3-pytorch -maxdepth 2 -type f | sort

# 2) 只列大文件，按体积排序（权重文件一眼可见）——这两条在 Linux/macOS 都能跑
find ./minimind-3-pytorch -maxdepth 2 -type f -size +50M -exec du -h {} + | sort -h

# 3) 只找我们要的那几个名字
find ./minimind-3-pytorch \( -name 'full_sft*.pth' -o -name 'pretrain*.pth' \) | sort
```

> Linux 上还可以用 `-printf '%12s  %p\n'` 直接拿字节数；macOS 的 `find` 没有这个参数
> （本地排查时会报 `unknown primary or operator`），用上面的 `-exec du -h {} +` 即可。

期望看到 `.../full_sft_768.pth`（约 130MB，fp16 的 63.91M 参数模型）。
**如果后缀不是 `_768`（例如 `_104M`），先停下把 `find` 输出发我**，不要改名硬凑——
下一节的校验会说明为什么不能凑。

### 4.3 复制进 out/ 并校验（防静默错配）

```bash
cd /root/autodl-tmp/minimind && mkdir -p out
src=$(find /root/autodl-tmp/models/minimind-3-pytorch -name 'full_sft_768.pth' -print -quit)
[ -n "$src" ] && cp -v "$src" out/ || echo "没找到 full_sft_768.pth，先看 4.2 的输出"
ls -lh out/
sha256sum out/full_sft_768.pth | tee out/full_sft_768.pth.sha256   # 记录来源哈希
```

`init_model()` 用的是 `load_state_dict(..., strict=False)`：**文件缺失会直接报错，但文件不匹配是静默的**。
所以复制完必须自己校一次 key：

```bash
cd /root/autodl-tmp/minimind
python - <<'PY'
import sys, torch
sys.path.insert(0, '.')
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

cfg = MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=1024, use_moe=False)
model = MiniMindForCausalLM(cfg)
weights = torch.load('out/full_sft_768.pth', map_location='cpu')
missing, unexpected = model.load_state_dict(weights, strict=False)
print('file keys:', len(weights), '| missing:', len(missing), '| unexpected:', len(unexpected))
print('missing sample:', missing[:3])
print('unexpected sample:', unexpected[:3])
PY
```

判定标准（用一个同结构文件在本机验证过，正确文件输出 `missing: 0 | unexpected: 0`）：

| 输出 | 含义 | 处理 |
|---|---|---|
| `missing: 0 | unexpected: 0` | 权重与架构完全匹配 | 可以开始训练 |
| `missing` 或 `unexpected` > 0 | 文件来自别的尺寸/结构 | 把输出发我，换正确的起点 |
| 抛 `size mismatch` | 形状不一致（实测：换成 512 维模型会打印几十行 size mismatch） | 同上，不要跳过 |

⚠️ **这个检查只能证明「结构对」，不能证明「阶段对」**：
`pretrain_768.pth`、`full_sft_768.pth`、`grpo_768.pth` 三个文件的形状完全一样，
都会打印 `missing: 0 | unexpected: 0`。所以还要确认文件名对应的训练阶段，
并用 §6.1 的格式率门槛兜底：如果冷启动后格式率上不去，先怀疑起点权重拿错了。

## 5. 开机器自检（每次新 session 都跑一遍）

```bash
cd /root/autodl-tmp/minimind
python scripts/server_preflight.py --expect-gpu --min-disk-gb 20
```

期望最后一行是 `0 blockers`。它会检查：python/torch/CUDA/显存/bf16、仓库文件是否齐全、
**数据 sha256 是否仍与 `data.sha256` 一致**、`out/full_sft_768.pth` 是否存在、
磁盘与写权限（`out/ logs/ runs/ reports/`）。

任何 FAIL 都先解决再训练；WARN 要知道自己在接受什么（比如 bf16 不支持 → 用 `--dtype float16`）。

### 5.1 磁盘预算（为什么 20GB 就够，而不是 50GB）

| 项目 | 单份 | 份数 | 小计 |
|---|---|---|---|
| 官方权重 `models/minimind-3-pytorch/` | 14 个文件 | 1 套 | 2.6 GB |
| `out/<tag>_768.pth`（fp16 权重） | 132 MB | 7 个 run | 0.9 GB |
| `checkpoints/<tag>_768_resume.pth`（fp16 权重 + AdamW 两个动量 + scaler） | ~0.7 GB | 7 个 run | ~5 GB |
| `checkpoints/<tag>_768.pth`（fp16 快照） | 132 MB | 7 个 run | 0.9 GB |
| HF `datasets` 缓存 + `logs/` + `runs/` + `reports/` | — | — | < 0.5 GB |
| **合计** | | | **≈ 10 GB** |

`save_interval` 每次保存都是**同名覆盖**（`lm_checkpoint` 先写 `.tmp` 再 `os.replace`），
所以步数跑再多也不会堆出几百个 checkpoint，磁盘占用是个常数。

### 5.2 `OMP_NUM_THREADS` 警告（每次启动 python 都刷一行）

```
libgomp: Invalid value for environment variable OMP_NUM_THREADS
```

先看它被谁设成了什么：

```bash
env | grep -i omp
```

如果是空串或小数（AutoDL 无卡模式 0.5 核时镜像可能这么设），libgomp 会退回默认值——**不致命**，
但 DataLoader 的线程数会变得不可控。修掉它：

```bash
export OMP_NUM_THREADS=8
echo 'export OMP_NUM_THREADS=8' >> ~/.bashrc
python -c "import torch;print('clean')"    # 不再刷那行警告 = 修好了
```

`8` 按机器核数调（`nproc` 可查）；写进 `~/.bashrc` 是为了让以后每个 session 都生效。

---

## 6. SFT 冷启动（前置 run，不在登记表里）

> **为什么 tag 里有 `cot`**：pilot 用的 `math_cold_seed42`（CoT 是「按运算顺序…结果是 X」这种
> 只宣告结果、不含中间步骤的模板句）实测格式率 1.00、held-out 正确率 **0/20**，
> 错误形态是「模板句 + 随机数」（`66*23`→2888、`97*61`→4378）。
> 修好 CoT（列竖式进位/借位、凑十破十、幂展开）后换新 tag 重跑，
> **旧 tag 的产物原样保留在 `runs/math_cold_seed42/`，作为「CoT 不可学」的对照证据，别删别覆盖。**
> 这是一次干净的对照：`math_rlvr_train.jsonl` 与 `math_heldout.jsonl` 逐字节未变
> （`f55c0b6954ea` / `f829eb084a3e`），唯一变量就是 SFT 的 CoT 内容。

```bash
cd /root/autodl-tmp/minimind
mkdir -p runs/math_cold_cot_seed42
cd trainer
python -u train_full_sft.py \
  --data_path ../dataset/math_sft_train.jsonl \
  --from_weight full_sft --save_weight math_cold_cot_seed42 \
  --epochs 2 --batch_size 16 --learning_rate 1e-5 --max_seq_len 512 \
  --num_workers 0 --save_interval 1000 --log_interval 20 \
  2>&1 | tee ../runs/math_cold_cot_seed42/train.log
```

- 期望看到 `Model Params: 63.91M`（对得上 64M 配置）和 250 个 step 日志。
- **注意**：`train_full_sft.py` 是 MiniMind 原版脚本，**不写 `logs/*_metrics.csv` 和 `samples.jsonl`**，
  证据就是 `train.log` + 权重 + 下面的评估。所以它的归档天然是 `partial`，不要去手工造 CSV。

### 6.1 退出标准（SOP §7.3：格式率 ≥ 80%）

```bash
cd /root/autodl-tmp/minimind
python scripts/eval_heldout.py \
  --model_path out/math_cold_cot_seed42_768.pth \
  --output runs/math_cold_cot_seed42/sft_format_check.json \
  --limit 20 --num_samples 1 --max_new_tokens 512
```

读输出的 `format_rate`（应 ≥ 0.8）和 `marker_hit_rate`。注意 `--limit 20` 取的是文件前 20 条（都是 easy 桶），
**只看格式，不要拿这 20 题的数字谈能力**。

格式率 < 80%：先别进 RL。检查 `math_sft_train.jsonl` 的模板（`marker` 与 `--reward_marker` 都是 `答案是`）
以及 SFT 的 label mask，把问题发我再决定。

---

### 6.2 难度标定复测（决定 GRPO 数据该多难）

`runs/probe/calibration.json` 是**起点策略的能力体检**：6 档 × 20 题 × 8 采样，出题模板与主数据集
完全相同（`synthetic_question`），差异只在数字范围。pilot 阶段在旧冷启动权重
（`out/math_cold_seed42_768.pth`，CoT 是「按运算顺序…结果是 X」的模板句）上实测：

| 档位 | 形态 | greedy | pass@1 | pass@8 |
|---|---|---|---|---|
| `probe-sub-1x1` | 18-12 | 0.250 | 0.206 | 0.450 |
| `probe-add-1x1` | 7+8 | 0.150 | 0.119 | 0.400 |
| `probe-mul-2x1` | 23*7 | 0.050 | 0.019 | 0.100 |
| `probe-add-2x2` | 43+38（≈当前 easy 的量级） | 0.000 | 0.013 | 0.100 |
| `probe-add-2x1` | 43+8 | 0.000 | 0.000 | 0.000 |
| `probe-mul-1x1` | 7*8（九九表） | 0.000 | 0.000 | 0.000 |

**结论**：SOP §「easy」桶（`17*23`、`94-35` 这种两位数运算）远在 64M 起点策略的能力边界之外
（pass@1 ≈ 0.01–0.13）。直接拿它跑 GRPO ⇒ 组内奖励恒等 ⇒ 优势 A=0 ⇒ 梯度为 0，即 MiniMind
README 里写的「奖励稀疏」，白烧 1.5–2.5 小时 GPU。这不丢人，是**要写进报告的现象**（SOP 把
「64M 能力边界」列为项目卖点）。

CoT v2 冷启动之后**必须复测一次**（同一份探针、同一个 seed，唯一变量是权重）：

```bash
cd /root/autodl-tmp/minimind
python scripts/eval_heldout.py \
  --data_path dataset/probe/calibration_probe.jsonl \
  --model_path out/math_cold_cot_seed42_768.pth \
  --output runs/probe/calibration_cot_v2.json \
  --num_samples 8 --max_new_tokens 256
```

- 探针文件**不进** `data.sha256`，所以输出的 `data_matches_manifest: false` 与那行 WARNING 是**预期行为**，
  不是数据被改过（探针只是诊断工具，见 `scripts/calibration_probe.py` 文件头）。
- 决策规则：正式 GRPO 数据按 **pass@1 落在 [0.2, 0.6]** 的档位配比；若复测后全档仍 < 0.05，
  就不跑 GRPO，改走「能力边界」对照实验路线——**改数据之前先跟我确认**（要新增登记表行和新 tag）。

---

## 7. rule 臂 GRPO

命令**由登记表生成**，不要手敲（手敲必然漏掉某个消融开关）：

```bash
cd /root/autodl-tmp/minimind
python scripts/experiment_registry.py --print-command mini_rule_seed42
```

渲染出来的就是一段可直接执行的 bash：`set -o pipefail` → `mkdir -p runs/<tag>` → `cd trainer`
→ `python -u train_grpo.py … 2>&1 | tee ../runs/<tag>/train.log`。
`-u` 是必需的：stdout 进管道会被缓冲，没有它日志文件会几分钟不刷新。

正式跑（`nohup` 保证 ssh 断开不中断；日志由命令自己 tee 到 `runs/<tag>/train.log`）：

```bash
cd /root/autodl-tmp/minimind
nohup bash -c "$(python scripts/experiment_registry.py --print-command mini_rule_seed42)" >/dev/null 2>&1 &
echo $! > /tmp/mini_rule_seed42.pid
sleep 60 && tail -5 runs/mini_rule_seed42/train.log
```

### 7.1 先测速，再决定要不要跑满

```bash
cd /root/autodl-tmp/minimind && sleep 300 && python - <<'PY'
import csv, datetime as dt
rows = list(csv.DictReader(open('logs/mini_rule_seed42_metrics.csv')))
t = [dt.datetime.fromisoformat(r['timestamp']) for r in rows]
s = int(rows[-1]['step']) - int(rows[0]['step'])
d = (t[-1] - t[0]).total_seconds()
print(f"{s} steps in {d:.0f}s -> {d/s:.2f}s/step -> 1000 steps ≈ {1000*d/s/3600:.1f} h")
PY
```

- 总步数 = `2000 行 / batch_size 2 = 1000`（1 epoch）。
- 如果估算 > 4 小时，先停下来告诉我：可以加 `--max_gen_len 256` 或换更小的数据子集，
  但那**就是另一个实验条件**了，要新增一行登记表，不能悄悄改。

### 7.2 训练期要看的三个数

```bash
tail -f runs/mini_rule_seed42/train.log
```

1. `Reward` 是否上升（rule 模式在 0 附近起步，正常会往正数走）；
2. `correct_rate` 是否上升——**这是真信号，reward 不是**；
3. `valid_groups: x/2` 与 `all_wrong: n`——如果长期全错，说明模型还没学会格式/太弱，
   这正是 ACR 指标要记录的现象（不要把它当成训练失败）。

### 7.3 中断与续训

- 断线不会中断 `nohup` 的任务；重连后用 `tail -f runs/<tag>/train.log` 看进度。
- 真要停：`kill -INT $(cat /tmp/mini_rule_seed42.pid)`。
- 续训：加 `--from_resume 1`（读 `checkpoints/<tag>_768_resume.pth`）。
- ⚠️ **MiniMind 按前缀复用 checkpoint**：同名 run 重跑会悄悄接着上次训练。
  要重跑就换 `--run_tag`/`--save_weight`（也就是新登记一行），或先删 `checkpoints/<tag>_*`。

---

## 8. held-out 评估

```bash
cd /root/autodl-tmp/minimind
python scripts/eval_heldout.py \
  --model_path out/mini_rule_seed42_768.pth \
  --data_path dataset/math_heldout.jsonl \
  --output runs/mini_rule_seed42/eval.json \
  --num_samples 16 --max_new_tokens 512 --temperature 0.8 \
  --top_p 0.85 --top_k 50 --seed 42 --device cuda \
  --marker '答案是'
```

- 先 `--limit 20` 快速确认脚本跑得动，再全量 400 题。
- 输出里的 `data_matches_manifest: true` 说明评估用的 held-out 文件与训练/登记时是同一份；
  如果是 `false`，那次评估作废，先查数据是不是被改过。
- 同时会写 `<output>.jsonl`（每条 completion 一行），图 5 的长度直方图就靠它。

---

## 9. 归档 + 汇总（本地也能做，服务器上做更快）

```bash
cd /root/autodl-tmp/minimind
python scripts/archive_run.py --run-tag math_cold_cot_seed42 --force
python scripts/archive_run.py --run-tag mini_rule_seed42 --force
python scripts/archive_run.py --check --strict --all  # 全量自检：哈希 + 必需文件
python scripts/summarize_runs.py --require-eval
```

- **`--force` 是必需的**：`eval_heldout.py` 会先往 `runs/<tag>/eval.json` 写评估结果，等再去归档时
  目录已经有文件，`archive_run.py` 默认会拒绝（`FileExistsError`）。`--force` 只刷新派生文件
  （`command.txt`/`environment.txt`/`data_manifest.json`/`checkpoints.txt`/`analysis.md`），
  **不会覆盖已有的 `eval.json`/`eval.jsonl`/`train.log`**。想先归档再评估就把它去掉。
- `--check --strict` 必须带 `--run-tag TAG` 或 `--all`，单独用会报 `pass --run-tag TAG or --all`。
  `--all` 只覆盖「登记表里有、且 `logs/` 里有 metrics」的臂，所以未登记的冷启动要按 tag 单独检。
- SFT 那个 run 会是 `[partial]`（缺 metrics/samples/config，原因见 §6），这是**如实记录**而不是错误。
  `--check --strict` 对**未登记**的 run（冷启动就属此类）用一份更短的必需清单：MiniMind 的
  `train_full_sft.py` 根本不写那三个文件，所以它们不该被当成证据缺口。`manifest.json` 里
  `prerequisite: true` 就是标记。**登记表里的 6 个臂没有这个豁免**，缺 metrics 仍然 FAIL。
- `--require-eval` 在还有臂没评估时返回非 0，正好当「能不能开始写简历」的闸门。
- 结果核对：`reports/summary.md` 第 2 节就是 SOP §10.3 的评估矩阵雏形。

### 9.1 逐题配对比较（回答「这点提升是不是噪声」）

`summarize_runs.py` 给的是**两个百分比相减**，那看不出统计显著性。评估跑完之后再补一条：

```bash
cd /root/autodl-tmp/minimind
python scripts/paired_compare.py \
  --baseline math_cold_cot_seed42 \
  --arms mini_rule_seed42 mini_rule_seed7 mini_rule_dynamic_seed42 mini_rule_len_seed42
```

它按 `sample_id` 把每个臂和冷启动**逐题配对**，用 McNemar 精确检验回答三个问题：

1. `greedy 净变` 与 `p`：净多对了几题，这个净额在零假设下有多罕见（**p < 0.05 才算显著**）；
2. `能力净变` 与 `p`：用 **pass@16 的逐题版本**（这题 16 次采样里至少对过一次）当"解题能力存在吗"的代理。
   它的净变不显著、而 greedy 显著，就是「**分布锐化而非能力扩展**」的直接证据（SOP 参考文献 arXiv:2504.13837）；
3. 分档 greedy 对比：提升是不是只来自某一档（对上 §6.2 的 ACR 解释链）。

输出 `reports/paired_compare.md` + 同名 `.json`，每一格都能回到 `runs/<tag>/eval.json`。
两个侧面的 `sample_id` 不一致时会显式列出来，**不会静默丢题**。

---

## 10. 把结果拉回本地

```bash
# 本地执行：结果三件套（反向 tar 管道，服务器打包 → 本地解包）
cd /Users/qi/实习/项目/minimind
ssh autodl \
  'cd /root/autodl-tmp/minimind && tar czf - runs reports logs' | tar xzf - -C .

# 权重可选（每个 ~130MB）
mkdir -p out
ssh autodl \
  'cd /root/autodl-tmp/minimind/out && tar czf - *.pth' | tar xzf - -C out
```

若服务器上没有某个目录（例如 `logs/` 被清理过），把它从上面的列表里去掉再执行，否则 tar 会报错退出。

拉回来后本地再跑一次 `archive_run.py --check --strict` 与 `summarize_runs.py`，
确认哈希一致——这就是「简历里的每个数字都能回到 JSON/CSV」的证据链。

---

## 11. 故障速查

| 症状 | 先查 | 处理 |
|---|---|---|
| CUDA OOM | `nvidia-smi` 显存占用 | 降 `--batch_size`→1、`--num_generations`→4、`--max_gen_len`→256 |
| 启动就报权重找不到 | `ls out/` | 文件名必须是 `full_sft_768.pth`（见 §4） |
| `loss: nan` | dtype / lr | 加 `--dtype float16`；确认没改 lr |
| reward 一直不涨、`Adv Std: 0.0` | `all_wrong/valid_groups` | 全对或全错组：调难度、开 dynamic sampling、或先修 SFT |
| `format_rate: 0.0` | marker 是否一致 | 数据模板、`--reward_marker`、verifier 三者的 `答案是` 必须相同 |
| 慢得离谱 | 日志里的 `device` | 确认是 `cuda:0` 而不是 `cpu`；CPU 回退通常意味着 torch 装成了 CPU 版 |
| `nvidia-smi` 报 `No devices were found` | 实例信息里的 CPU/内存 | 无卡模式，属正常；训练前关机 → 有卡模式重开（§1.2） |
| 服务器上 `cd /Users/…` 报 No such file | 文件是否传过 | 本地路径在服务器上不存在，先做 §2 上传 |
| `ssh 主机 '命令'` 报 `python: command not found` | `command -v python` | 一次性 ssh 是**非交互 shell**，不加载 conda 的 PATH：改成 `bash -lc '命令'`，或直接 `ssh` 登录进去在会话里操作（更推荐，`nohup`/`screen` 都在会话里）|
| `datasets` 缓存写不进去 | `HF_HOME` | `export HF_HOME=/root/autodl-tmp/hf` |
| 每步都刷 `libgomp: Invalid value for environment variable OMP_NUM_THREADS` | `env \| grep -i omp` | 变量被设成了非法值（空串/小数，无卡模式 0.5 核时容易出现）：`export OMP_NUM_THREADS=8` 并写进 `~/.bashrc`（见 §5.2） |
| 评估 `data_matches_manifest: false` | 数据 sha256 | 先看是不是探针文件（脚本会打 `NOTE:`，属预期）；若是 held-out 则数据被改过，重传或重新生成，别用这次结果 |

---

## 12. 本地/服务器分工

| 事情 | 在哪做 |
|---|---|
| 写代码、跑单测、`clean_math_data.py --synthetic-only`、`archive_run --check`、`summarize_runs`（无 GPU 部分） | 本地 |
| SFT、GRPO、held-out 评估、任何 `--device cuda` | 服务器 |
| 数据生成 | 本地一次，服务器只校验 sha256 |
| 归档与汇总 | 两边都行，两边结果必须一致 |
