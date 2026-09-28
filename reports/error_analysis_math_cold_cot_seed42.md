# 错误类别分析：`math_cold_cot_seed42`

- 输入：`runs/math_cold_cot_seed42/eval.jsonl`（共 400 条）
- 过滤：mode=greedy, difficulty=全部
- 长度口径：tokens ≥ 0.95×512 = 486.4 记 `length_burn`
- 复算命令：`python scripts/error_analysis.py --path runs/math_cold_cot_seed42/eval.jsonl --mode greedy --max-new-tokens 512`

## 1. 主类别（互斥，按优先级）

| 类别 | 条数 | 占比 | 说明 |
|---|---:|---:|---|
| `correct` | 149 | 37.2% | verifier 判定正确（仍可能同时命中重复等标记，见标记表） |
| `arithmetic_error` | 251 | 62.7% | 抽到数字但与 GT 不符 |

未命中（各 0 条，口径已检查）：`repetition`、`unclosed_think`、`redundant_answer`、`length_burn`、`no_answer`、`other`

## 2. 标记（非互斥，同一行可命中多项）

| 标记 | 条数 | 占比 | 说明 |
|---|---:|---:|---|
| `correct` | 149 | 37.2% | verifier 判定正确 |
| `arithmetic_error` | 251 | 62.7% | 答案错 |

未命中（各 0 条，口径已检查）：`no_answer`、`marker_unparsed`、`unclosed_think`、`repetition`、`redundant_answer`、`length_burn`

## 3. 难度 × 主类别

| 难度 | 条数 | `correct` | `repetition` | `unclosed_think` | `redundant_answer` | `length_burn` | `no_answer` | `arithmetic_error` | `other` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| easy | 134 | 74 | · | · | · | · | · | 60 | · |
| hard | 133 | 58 | · | · | · | · | · | 75 | · |
| medium | 133 | 17 | · | · | · | · | · | 116 | · |

## 4. 抽样明细（每个失败类别最多 2 条）

### arithmetic_error

- `easy` | gt=`1518` | extracted=`1728` | tokens=54 | status=`incorrect`

```text
66*23：23 拆成 20 和 3；66*20=1440；66*3=184；1440+184=1728。所以结果是 1728。
</think>

答案是 1728
```

- `easy` | gt=`5917` | extracted=`6577` | tokens=64 | status=`incorrect`

```text
97*61：61 拆成 60 和 1；97*60=6360；97*1=97；6360+97=6577。所以结果是 6577。
</think>

答案是 6577
```
