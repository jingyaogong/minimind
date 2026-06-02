# Day 5 Prereading: 评测、推理参数、失败分析

## 今日目标

今天不追求继续训练，而是建立评测习惯。你要能判断自己的模型哪里变好了，哪里只是看起来变好了。

## 必须掌握的基础概念

- Offline eval：固定测试集，离线比较。
- Human eval：人工主观评分。
- Regression eval：新权重不能明显退化旧能力。
- Sampling parameters：temperature、top_p、top_k、repetition_penalty。
- Hallucination：模型生成看似合理但错误的信息。
- Exposure bias：训练时看标准前缀，推理时看自己生成的前缀。

## 面试考点

- 为什么 loss 不能完全代表生成质量？
- temperature 和 top_p 分别控制什么？
- 为什么小模型容易重复和幻觉？
- 如何设计一个低成本的 LLM eval？
- 如何判断 SFT 是否过拟合？

## 工业要求

- 每次训练后必须跑固定评测集。
- 评测 prompt 要覆盖：身份、事实、代码、总结、推理、工具调用。
- 生成评测要固定随机种子或多次采样。
- 不要只展示最好样例，要记录失败样例。
- 小模型不要用大模型标准苛责，但要诚实记录边界。

## 项目中要看的文件

- `eval_llm.py`
- `scripts/eval_toolcall.py`
- `scripts/serve_openai_api.py`
- `scripts/web_demo.py`

## 今日固定评测集

```text
1. 你是谁？
2. 解释什么是机器学习。
3. 为什么天空是蓝色的？
4. 请用 Python 写一个斐波那契函数。
5. 比较猫和狗作为宠物的优缺点。
6. 推荐一些中国美食。
7. 把下面这段话总结成三点：...
8. 如果明天下雨，我应该如何出门？
```

## 今日注意事项

- 同一个 prompt 至少用 2 组采样参数测试。
- 把失败样例原样保存。
- 评估时区分：语言流畅、指令跟随、事实正确、格式正确。
