# 第14节：autocast 与 dtype（混合精度）

> 代码中根据设备选择混合精度策略的逻辑

---

## 📚 本节目录

1. [autocast 是什么](#1-autocast-是什么)
2. [nullcontext 是什么](#2-nullcontext-是什么)
3. [为什么 CPU 要用 nullcontext](#3-为什么-cpu-要用-nullcontext)
4. [整体逻辑](#4-整体逻辑)
5. [dtype 和 autocast 的区别](#5-dtype-和-autocast-的区别)
6. [直觉例子](#6-直觉例子)
7. [一句话总结](#7-一句话总结)

---

## 1. autocast 是什么？

**PyTorch 混合精度控制器**

自动把部分计算变成 fp16/bf16，加速 GPU

```python
with autocast_ctx:
    loss = model(x)
```

---

## 2. nullcontext 是什么？

**"什么都不做的 context manager"**

```python
with nullcontext():
    ...  # 等价于直接执行，无任何包装
```

---

## 3. 为什么 CPU 要用 nullcontext？

**CPU 不支持 AMP autocast**

强行用 `torch.cuda.amp.autocast` 在 CPU 上会报错。

---

## 4. 整体逻辑

```python
if device_type == "cpu":
    autocast_ctx = nullcontext()  # CPU：不做任何事
else:
    autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)  # GPU：混合精度
```

| 设备 | 策略 |
|------|------|
| GPU | 用 autocast（混合精度） |
| CPU | 不做任何事 |

---

## 5. dtype 和 autocast 的区别

| 概念 | 作用 | 说明 |
|------|------|------|
| **dtype** | 模型权重/计算默认类型 | Linear layer 的 weight 是 fp16/bf16 |
| **autocast** | 运行时自动混合精度策略 | 某些 op 用 fp32，某些 op 用 fp16/bf16 |

---

## 6. 直觉例子

假设：`dtype = bf16`，`autocast = on`

| 操作 | 精度 |
|------|------|
| matmul | bf16 |
| softmax | fp32（稳定） |
| loss | fp32 |

---

## 7. 一句话总结

**根据设备选择混合精度策略：GPU 用 autocast + bf16/fp16 提速，CPU 则关闭 autocast 保持正常执行。**
