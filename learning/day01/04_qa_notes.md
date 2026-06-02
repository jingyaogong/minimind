# Day 1 问答笔记（实操记录）

> 本文件记录第一天实际操作中遇到的、有价值的问题与结论。区别于 `01_prereading`/`02_deep_dive`（课程材料），这里是「踩坑 + 解决」的真实记录。

---

## Q1. 没有写权限装不了包：`EnvironmentNotWritableError`

**现象：** `conda install torch` 报 `current user does not have write permissions to /opt/anaconda3`。

> **笔记：** 系统级 base 环境（`/opt/anaconda3`）通常没有写权限，**不要往 base 里装包**。正确做法是建自己的虚拟环境（装在用户目录，有写权限）。

## Q2. 虚拟环境是什么？`python -m venv .venv` 在干嘛？

> **笔记：** 虚拟环境 = 一个「隔离的小盒子」，里面有独立的 python 和 pip，装的包只属于这个项目，不污染系统、不和别的项目冲突。
> - `python -m venv 名字` → 创建（只需一次）。`.venv` 是惯例名（开头 `.` 会隐藏）。
> - 环境名可自定义，用英文/数字/`-`/`_`，别用空格中文。

## Q3. venv 和 conda 的区别？怎么激活/退出？

> **笔记：** 两种建环境方式，激活命令不同：
> | 方式 | 创建 | 激活 | 退出 |
> |------|------|------|------|
> | venv | `python -m venv minimind-env` | `source minimind-env/bin/activate` | `deactivate` |
> | conda | `conda create -n minimind python=3.10` | `conda activate minimind` | `conda deactivate` |
>
> 退出命令是 `deactivate`（不是 `inactivate`）。conda 新建的环境默认装到 `~/.conda/envs`（用户目录，有写权限），所以能绕开 base 没权限的问题。

## Q4. 怎么判断包装进了虚拟环境，还是误装到 home？

> **笔记：核心判断方法——看路径，不要凭感觉。**
> ```bash
> which python   # 带 .venv/bin/ 才对；显示 /opt/anaconda3/bin 说明没激活
> which pip      # 同上
> pip show 包名   # 看 Location 字段在不在 .venv 里
> ```
> **关键习惯：** 每次新开终端，venv 都不会自动激活，必须先 `source .venv/bin/activate`，看到命令行前面有 `(.venv)` 再操作。没激活就 `pip install`，包会装到 `~/.local`（用户级），不在虚拟环境里。

## Q5. `pip install requirements` 报错？

> **笔记：** 装依赖文件要加 `-r` 和完整文件名：`pip install -r requirements.txt`。
> 不加 `-r` 时，pip 以为你要装一个名叫 `requirements` 的包，报 `No matching distribution`。
> minimind 的 `requirements.txt` 里 `torch`/`torchvision` 被注释掉了，要单独装 GPU 版：
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

## Q6. modelscope 数据集下到哪了？怎么下对地方？

> **笔记：**
> - 默认下到缓存 `~/.cache/modelscope`（和 Python 环境无关，是缓存目录）。
> - 下载中断时，大文件会卡在 `._____temp` 临时目录里（没下完）。
> - **下对地方的技巧：** 用 `--local_dir` 直接下到项目的 dataset 目录：
>   ```bash
>   modelscope download --dataset gongjy/minimind_dataset pretrain_t2t_mini.jsonl --local_dir ./dataset
>   ```
> - 只想快速跑通，下 mini 版即可（`pretrain_t2t_mini` + `sft_t2t_mini`），不用下全量（pretrain_t2t 10GB / sft_t2t 14GB）。

---

## Q7. 项目的数据流是怎么走的？怎么「打点」调试？

**预训练数据流路线图：**
```
jsonl ({"text":...})
  → PretrainDataset.__getitem__   [dataset/lm_dataset.py:47]  文字→编码→加bos/eos→pad
  → DataLoader                    [train_pretrain.py:162]      拼成batch input_ids[B,L]
  → .to(device)                   [train_pretrain.py:28]
  → model.forward(input_ids,labels) [model_minimind.py:443]
        embed → Transformer → hidden[B,L,768] → lm_head → logits[B,L,6400]
        → logits[:-1] 预测 labels[1:] → cross_entropy → loss
  → backward + optimizer.step     [train_pretrain.py:37-49]
  → 日志/保存                      [train_pretrain.py:51-69]
```

> **笔记：打点 = 在关键位置加 print 看数据 shape 和内容。** 推荐顺序：
> 1. **先单独验证数据**（不启动训练）：写个 `debug_data.py`，建 Dataset → 取 `ds[0]` 看单条 → 取一个 batch 看 shape。
> 2. **再在训练里打点**：`__getitem__` 里看编码对不对（解码回去对比原文 + labels 里 -100 是不是 pad 位）；batch 处看 `input_ids.shape`；forward 后看 `logits.shape` 和 `loss`。
> 3. 打点加 `if index < 2` / `if step <= 2` 限制，避免刷屏。
> labels 里 `-100` = 该位置不计算 loss（pad 位置）。

---

## Q8. 现代大模型的数据是怎么处理和清洗的？

> **笔记：分两大阶段，逻辑完全不同。**
>
> **预训练数据**（海量、求量和广，TB~PB 级，全自动清洗）经典流水线：
> 1. **抽取**：从网页 HTML 抽正文（trafilatura），去标签/广告。
> 2. **语种过滤**：fastText 识别语言，留目标语言。
> 3. **质量过滤**（核心）：启发式规则（Gopher/C4：太短、符号比例异常、重复行多→删）+ 模型打分（拿维基当正例训分类器打分）。
> 4. **去重**（关键！）：精确去重（哈希）+ 近似去重——**MinHash+LSH（`datasketch` 库）** 估 Jaccard 相似度删相似文档；**SimHash（`simhash` 库）** 指纹+汉明距离。⚠️ 这两个库就在 minimind 的 requirements.txt 里。去重防止模型死记重复内容、浪费算力。
> 5. **去隐私(PII)/有害内容**：去邮箱手机号、过滤色情暴力。
> 6. **去污染(Decontamination)**：把评测题(MMLU/CEval)从训练集删掉，否则评分虚高=作弊。
> 7. **格式统一**：→ `{"text": "干净文字"}`。
>
> **后训练数据**（少量、求质和准）：
> - **SFT**（`sft_t2t.jsonl`）：构造/筛选高质量问答对，来源含人工标注、公开数据、**模型蒸馏合成**（作者用 qwen3-4b 合成约10w条tool call）。筛选靠 LLM-as-a-judge 打分 + 多样性去重。格式 `{"conversations":[{role,content}]}`。
> - **RL/DPO**（`dpo.jsonl`）：每条是「同一问题的好/坏回答对」(chosen/rejected)，用于偏好对齐。
>
> **一句话总结：** 预训练清洗重在「质量过滤+去重+去污染」，SFT 重在「高质量问答对+格式统一」，DPO 重在「构造好坏对比对」。
>
> minimind 的数据集是作者已清洗好的成品，直接训练即可。

---

## Q9. 怎么看 GPU 利用率？nvidia-smi 表盘完整解读

**命令：** `nvidia-smi`（看一次）；`watch -n 1 nvidia-smi`（每秒刷新实时盯，Ctrl+C 退出，不影响训练）。

**表盘第一区（整卡状态）：**
```
| 59%  83C  P1  300W / 300W |  7807MiB / 97887MiB |  97%  Default |
   ↑     ↑   ↑      ↑              ↑                   ↑
  风扇  温度 性能 功耗/上限      显存用量/总量        算力利用率
```
> **笔记：关键看两个数，且它俩是两回事：**
> - **Memory-Usage（显存用量）**：`7807/97887 MiB` = 用了 7.8G / 共 96G。显存够不够「装得下」。
> - **GPU-Util（算力利用率）**：`97%` = 计算单元忙不忙。卡「算得快不快/有没有闲」。
> - 其他：风扇% / 温度(83℃偏高但安全) / Perf状态(P0最高~P8最低) / 功耗(300W已拉满=Max-Q功耗墙)。
> - ECC、MIG 一般用不到。

**表盘第二区（进程）：**
```
PID 42514   C   python   7284MiB   ← 你的训练进程占了 7.3G
PID  2898   G   Xorg      175MiB   ← 桌面系统占的，忽略
```
> **笔记：** `Type` 列 `C`=计算进程(你的训练)，`G`=图形进程(桌面)。能看出显存被谁吃了。

## Q10. 加大 batch_size 为什么没更快？为什么要同步调大学习率？

> **笔记：踩坑认知——「显存没满 ≠ 可以无脑加 batch 提速」。**
> 判断卡有没有「吃饱」要看 **GPU-Util**，不是显存：
> - 显存 8% 但 GPU-Util 已 97% → **算力已满载**，加大 batch **不会提速**（吞吐量/每秒token数已固定）。
> - 只有 GPU-Util 偏低（卡在等数据/batch太小喂不饱）时，加 batch 才提速。
>
> **实测：** batch 32(39695步/epoch) → 128(9924步/epoch)，总时间基本不变甚至略增。因为每步算4倍活、总步数变1/4，乘起来≈不变。早期 epoch_time(ETA) 偏高是因为启动开销摊在少量步数上，不准，跑久了会降。
>
> **大 batch 的真正收益是「训练质量」不是「速度」：** 梯度噪声更小、更稳，可配更大LR、用更少epoch收敛。
>
> **为什么 batch 和 LR 要一起放大：**
> 1. batch 大 → 梯度估计更准 → 敢迈更大步(大LR)，不怕走偏。
> 2. batch 大 → 总步数变少 → 每步不迈大点会欠训练，需大LR补偿。
>
> **缩放法则**（batch 放大 k 倍）：
> | 法则 | LR 调整 | 适用 |
> |------|--------|------|
> | 线性缩放 | ×k | SGD |
> | 平方根缩放 | ×√k | **Adam/AdamW(minimind用这个)** |
> batch 32→128(×4)：AdamW 推荐 √4=×2，即 `5e-4 → 1e-3`。
>
> **真正提速的手段（不是加batch）：** `--use_compile 1`(torch.compile) / 减少 `--epochs`。

---

## 今日待办状态

- [x] 建立 `.venv` 虚拟环境（`~/Project/minimind/.venv`）
- [x] modelscope 装进 `.venv`
- [x] 清理误装到 home 的包和缓存
- [x] 下载数据集（全量已下到 ~/.cache，再 mv 进 ./dataset/；下载时漏了 `--local_dir` 是坑）
- [x] 安装 PyTorch GPU 版（torch 2.11.0+cu128，Blackwell 必须 cu128）+ requirements.txt
- [x] **成功跑通第一次预训练**（loss 从 ~7.3 正常下降）✅
- [ ] 配置 VSCode 调试（已建 .vscode/launch.json，需装 Python 扩展，用 F5 而非 `python xxx.py`）
- [ ] 跑完预训练 → 用 eval_llm.py 测试生成效果
