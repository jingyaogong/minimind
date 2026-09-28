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

## 4. 抽样明细（每个失败类别最多 20 条）

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

- `easy` | gt=`2350` | extracted=`625` | tokens=52 | status=`incorrect`

```text
25*94：29 拆成 20 和 9；25*20=500；25*9=275；500+275=625。所以结果是 625。
</think>

答案是 625
```

- `easy` | gt=`2300` | extracted=`2035` | tokens=57 | status=`incorrect`

```text
92*25：25 拆成 20 和 5；92*20=1920；92*5=475；1920+475=2035。所以结果是 2035。
</think>

答案是 2035
```

- `easy` | gt=`6853` | extracted=`6536` | tokens=67 | status=`incorrect`

```text
77*89：89 拆成 80 和 9；77*80=6360；77*9=636；6360+636=6536。所以结果是 6536。
</think>

答案是 6536
```

- `easy` | gt=`2304` | extracted=`3488` | tokens=58 | status=`incorrect`

```text
48*48：48 拆成 80 和 4；48*80=3200；48*4=248；3200+248=3488。所以结果是 3488。
</think>

答案是 3488
```

- `easy` | gt=`1254` | extracted=`1466` | tokens=57 | status=`incorrect`

```text
57*22：22 拆成 20 和 2；57*20=1160；57*2=116；1160+116=1466。所以结果是 1466。
</think>

答案是 1466
```

- `easy` | gt=`7` | extracted=`10` | tokens=35 | status=`incorrect`

```text
56-49：个位 6-9=0；十位 5-4=1。所以结果是 10。
</think>

答案是 10
```

- `easy` | gt=`5720` | extracted=`9860` | tokens=63 | status=`incorrect`

```text
88*65：65 拆成 60 和 5；88*60=5360；88*5=4360；5360+4360=9860。所以结果是 9860。
</think>

答案是 9860
```

- `easy` | gt=`3360` | extracted=`3620` | tokens=46 | status=`incorrect`

```text
84*40：40 先看成 4*84=362；再补一个 0 得 3620。所以结果是 3620。
</think>

答案是 3620
```

- `easy` | gt=`1350` | extracted=`1280` | tokens=54 | status=`incorrect`

```text
54*25：25 拆成 20 和 5；54*20=1080；54*5=280；1080+280=1280。所以结果是 1280。
</think>

答案是 1280
```

- `easy` | gt=`1005` | extracted=`1085` | tokens=56 | status=`incorrect`

```text
67*15：15 拆成 10 和 5；67*10=670；67*5=355；670+355=1085。所以结果是 1085。
</think>

答案是 1085
```

- `easy` | gt=`1445` | extracted=`1555` | tokens=57 | status=`incorrect`

```text
17*85：85 拆成 80 和 5；17*80=1360；17*5=65；1360+65=1555。所以结果是 1555。
</think>

答案是 1555
```

- `easy` | gt=`1632` | extracted=`1728` | tokens=53 | status=`incorrect`

```text
24*68：68 拆成 60 和 8；24*60=1440；24*8=168；1440+168=1728。所以结果是 1728。
</think>

答案是 1728
```

- `easy` | gt=`41` | extracted=`39` | tokens=62 | status=`incorrect`

```text
72-31：个位 不够减，2+10=12，12-1-1=9，向高位借 1；十位 7-1=6，6-3=3。所以结果是 39。
</think>

答案是 39
```

- `easy` | gt=`7` | extracted=`10` | tokens=35 | status=`incorrect`

```text
66-59：个位 6-9=0；十位 6-5=1。所以结果是 10。
</think>

答案是 10
```

- `easy` | gt=`1364` | extracted=`1394` | tokens=59 | status=`incorrect`

```text
44*31：31 拆成 30 和 1；44*30=1260；44*1=44；1260+44=1394。所以结果是 1394。
</think>

答案是 1394
```

- `easy` | gt=`2528` | extracted=`2882` | tokens=66 | status=`incorrect`

```text
32*79：79 拆成 70 和 9；32*70=2720；32*9=272；2720+272=2882。所以结果是 2882。
</think>

答案是 2882
```

- `easy` | gt=`2898` | extracted=`2592` | tokens=61 | status=`incorrect`

```text
63*46：46 拆成 40 和 6；63*40=2460；63*6=392；2460+392=2592。所以结果是 2592。
</think>

答案是 2592
```

- `easy` | gt=`3477` | extracted=`3577` | tokens=64 | status=`incorrect`

```text
57*61：61 拆成 60 和 1；57*60=3620；57*1=57；3620+57=3577。所以结果是 3577。
</think>

答案是 3577
```
