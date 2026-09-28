# 02 · 材料审计

## Materials Summary

| 材料 | 位置 | 状态 |
|---|---|---|
| 目标 JD | `/Users/qi/实习/materials/target_jd.txt` | 语音语言交互大模型方向 |
| 简历 | `/Users/qi/实习/项目和简历/简历 (1).pdf` | 1 页；两个项目（中医古籍标点、NMT）+ 教育 + 技能 |
| 本项目代码 | `/Users/qi/实习/项目/minimind` | MiniMind 上游改造：verifier / 数据 / GRPO 改造 / 评估 / 归档 / 测试 |
| 实验报告 | `reports/report.md`（v1.0） | 每个数字可指回原始文件 |
| 项目说明 | `README-rlvr-audit.md` | 复现入口 + 失败清单 + 已知坑 |

## Strong Evidence（能直接支撑简历措辞）

- `reports/report.md` + `README-rlvr-audit.md`：结论表、六臂矩阵、局限清单。
- `runs/<tag>/eval.json` / `eval.jsonl` / `manifest.json`：8 个 run 的评测原始数据与归档哈希。
- `reports/paired_compare.json`：逐题配对统计（净变、p 值、分档）。
- `reports/error_analysis_*.md`、`reports/audit_*.md`：400 条全量错误分类 + 每类 20 条人工审计原文。
- `logs/<tag>_metrics.csv` / `_samples.jsonl` / `_config.json`：训练动态与 rollout 原始记录。
- `configs/experiment_registry.csv`：预注册（含假设列），防事后编故事。
- `dataset/manifests/data.sha256` + `human_audit_sample.tsv`：数据指纹与人工抽样。
- `tests/`：313 个测试全绿（`python -m pytest -q`）。
- `scripts/compare_runs.py` 产物：同命令同种子重跑的逐 step 对照（35/200 一致）。

## Medium Evidence（需解释才成立）

- `planning/*.md`：内部规划与评审记录，可证明流程意识，不是结果证据。
- `docs/server_runbook.md`：服务器手册（环境、坑），面试可当「工程规范」谈。
- `reports/figures/`：6 张图，均为派生文件，引用时要说明来源文件。

## Weak / Missing Evidence

- **简历上旧的两个项目**（GuwenBERT 中医古籍标点 94.12%/84.16%、NMT BLEU 35.62）：本机仓库里没有对应代码、数据或日志 → 属于「无法判断」，面试前你必须自己想清楚评估集与口径，否则被追问会崩。
- **无实习经历**、无 GitHub 公开链接（本项目目前只在本地）。
- **无语音 / 音频相关任何产物**。
- **无 Qwen2.5-1.5B / TRL / verl 实操**（SOP 的「工业近似层」未做，是主动范围裁剪）。
- **无 C/C++/perl 证据**。

## Ownership Risks

- 项目基于 MiniMind 上游（Apache-2.0）改造：措辞必须是「基于 MiniMind 上游改造 / 在 MiniMind 上实现」，**不能**写成「从零训练了一个大模型」。
- 全部实验为个人独立完成（单卡 4090 / 服务器），无团队归属歧义 → 可放心用「实现 / 设计 / 定位」，但别用「主导平台化」这类大词。
- 强度排序（H2 视角）：评测与负结果证据 > 训练链路改造 > 单臂正收益。

## Best Resume Sources

1. `README-rlvr-audit.md` §0 结论表 → 项目 bullet 的事实来源。
2. `reports/paired_compare.md` → 配对净变与 p 值。
3. `reports/error_analysis_mini_rm_seed42.md` + `reports/report.md` §4.9 → 错误类别与形态学。
4. `reports/report.md` §4.7 → 噪声地板与不可复现。

## Questions To Ask（需要你确认的事实）

1. 你有没有 C / C++ / perl 的实际经历（课程、竞赛、项目）？有就补进技能栏。
2. 旧两个项目现在还能找到代码 / 日志吗？找不到就要提前准备「口径自述」版本（什么数据、什么划分、指标怎么算）。
3. 你想投的是不是真的偏语音方向？如果是，`09` 的半天补课必做；如果只是海投，把开场白里的语音兴趣换成通用后训练兴趣。
4. 有没有可以公开的 GitHub 账号？没有的话，面试前把本项目整理成不带头文件/大文件的最小公开仓库（半天工作量，见 `09`）。
