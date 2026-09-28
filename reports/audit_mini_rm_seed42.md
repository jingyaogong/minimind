# 错误类别分析：`mini_rm_seed42`

- 输入：`runs/mini_rm_seed42/eval.jsonl`（共 400 条）
- 过滤：mode=greedy, difficulty=全部
- 长度口径：tokens ≥ 0.95×512 = 486.4 记 `length_burn`
- 复算命令：`python scripts/error_analysis.py --path runs/mini_rm_seed42/eval.jsonl --mode greedy --max-new-tokens 512`

## 1. 主类别（互斥，按优先级）

| 类别 | 条数 | 占比 | 说明 |
|---|---:|---:|---|
| `correct` | 53 | 13.2% | verifier 判定正确（仍可能同时命中重复等标记，见标记表） |
| `arithmetic_error` | 347 | 86.8% | 抽到数字但与 GT 不符 |

未命中（各 0 条，口径已检查）：`repetition`、`unclosed_think`、`redundant_answer`、`length_burn`、`no_answer`、`other`

## 2. 标记（非互斥，同一行可命中多项）

| 标记 | 条数 | 占比 | 说明 |
|---|---:|---:|---|
| `correct` | 53 | 13.2% | verifier 判定正确 |
| `arithmetic_error` | 347 | 86.8% | 答案错 |

未命中（各 0 条，口径已检查）：`no_answer`、`marker_unparsed`、`unclosed_think`、`repetition`、`redundant_answer`、`length_burn`

## 3. 难度 × 主类别

| 难度 | 条数 | `correct` | `repetition` | `unclosed_think` | `redundant_answer` | `length_burn` | `no_answer` | `arithmetic_error` | `other` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| easy | 134 | 28 | · | · | · | · | · | 106 | · |
| hard | 133 | 24 | · | · | · | · | · | 109 | · |
| medium | 133 | 1 | · | · | · | · | · | 132 | · |

## 4. 抽样明细（每个失败类别最多 20 条）

### arithmetic_error

- `easy` | gt=`5` | extracted=`10` | tokens=35 | status=`incorrect`

```text
32-27：个位 2-7=0；十位 3-2=1。所以结果是 10。
</think>

答案是 10
```

- `easy` | gt=`1518` | extracted=`1334` | tokens=57 | status=`incorrect`

```text
66*23：23 拆成 20 和 3；66*20=1280；66*3=184；1280+184=1334。所以结果是 1334。
</think>

答案是 1334
```

- `easy` | gt=`5917` | extracted=`6577` | tokens=64 | status=`incorrect`

```text
97*61：61 拆成 60 和 1；97*60=6360；97*1=97；6360+97=6577。所以结果是 6577。
</think>

答案是 6577
```

- `easy` | gt=`74` | extracted=`50` | tokens=34 | status=`incorrect`

```text
51+23：个位 1+3=4；十位 5+2=7。所以结果是 50。
</think>

答案是 50
```

- `easy` | gt=`2350` | extracted=`625` | tokens=52 | status=`incorrect`

```text
25*94：29 拆成 20 和 9；25*20=500；25*9=225；500+225=625。所以结果是 625。
</think>

答案是 625
```

- `easy` | gt=`2300` | extracted=`2760` | tokens=57 | status=`incorrect`

```text
92*25：25 拆成 20 和 5；92*20=1920；92*5=460；1920+460=2760。所以结果是 2760。
</think>

答案是 2760
```

- `easy` | gt=`145` | extracted=`150` | tokens=56 | status=`incorrect`

```text
50+95：个位 0+5=5；十位 5+9=14，写 4、向高位进 1；最高位再加上进位的 1。所以结果是 150。
</think>

答案是 150
```

- `easy` | gt=`27` | extracted=`40` | tokens=34 | status=`incorrect`

```text
52-25：个位 2-5=0；十位 5-2=3。所以结果是 40。
</think>

答案是 40
```

- `easy` | gt=`172` | extracted=`118` | tokens=68 | status=`incorrect`

```text
85+87：个位 5+7=12，写 2、向高位进 1；十位 8+8+1=15，写 5、向高位进 1；最高位再加上进位的 1。所以结果是 118。
</think>

答案是 118
```

- `easy` | gt=`6853` | extracted=`65360` | tokens=67 | status=`incorrect`

```text
77*89：89 拆成 80 和 9；77*80=6360；77*9=6360；6360+6360=65360。所以结果是 65360。
</think>

答案是 65360
```

- `easy` | gt=`2304` | extracted=`3924` | tokens=54 | status=`incorrect`

```text
48*48：48 拆成 40 和 8；48*40=2480；48*8=324；2480+324=3924。所以结果是 3924。
</think>

答案是 3924
```

- `easy` | gt=`1254` | extracted=`1296` | tokens=57 | status=`incorrect`

```text
57*22：22 拆成 20 和 2；57*20=1160；57*2=116；1160+116=1296。所以结果是 1296。
</think>

答案是 1296
```

- `easy` | gt=`7` | extracted=`10` | tokens=35 | status=`incorrect`

```text
56-49：个位 6-9=0；十位 5-4=1。所以结果是 10。
</think>

答案是 10
```

- `easy` | gt=`5720` | extracted=`6560` | tokens=63 | status=`incorrect`

```text
88*65：65 拆成 60 和 5；88*60=5480；88*5=4320；5480+4320=6560。所以结果是 6560。
</think>

答案是 6560
```

- `easy` | gt=`3360` | extracted=`3680` | tokens=46 | status=`incorrect`

```text
84*40：40 先看成 4*84=3680；再补一个 0 得 3680。所以结果是 3680。
</think>

答案是 3680
```

- `easy` | gt=`1350` | extracted=`640` | tokens=54 | status=`incorrect`

```text
54*25：25 拆成 20 和 5；54*20=580；54*5=280；580+280=640。所以结果是 640。
</think>

答案是 640
```

- `easy` | gt=`1005` | extracted=`935` | tokens=56 | status=`incorrect`

```text
67*15：15 拆成 10 和 5；67*10=670；67*5=355；670+355=935。所以结果是 935。
</think>

答案是 935
```

- `easy` | gt=`1445` | extracted=`16525` | tokens=57 | status=`incorrect`

```text
17*85：85 拆成 80 和 5；17*80=1360；17*5=625；1360+625=16525。所以结果是 16525。
</think>

答案是 16525
```

- `easy` | gt=`1632` | extracted=`3240` | tokens=56 | status=`incorrect`

```text
24*68：68 拆成 60 和 8；24*60=1680；24*8=1680；1680+1680=3240。所以结果是 3240。
</think>

答案是 3240
```

- `easy` | gt=`41` | extracted=`40` | tokens=35 | status=`incorrect`

```text
72-31：个位 2-1=1；十位 7-3=4。所以结果是 40。
</think>

答案是 40
```
