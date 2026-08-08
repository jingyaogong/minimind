# Lesson 18 笔记：LoRA 原理与实现

## 目录

- [setattr(module, "lora", lora) 的含义](#setattrmodule-lora-lora-的含义)
- [LoRA 生命周期工具链](#lora-生命周期工具链)
  - [总体流程图](#总体流程图)
  - [apply_lora（初始化时）](#apply_lora初始化时)
  - [save_lora（训练中/后）](#save_lora训练中后)
  - [load_lora（恢复/推理前）](#load_lora恢复推理前)
  - [merge_lora（部署前）](#merge_lora部署前)
  - [四个函数角色分工](#四个函数角色分工)
  - [三种典型使用模式](#三种典型使用模式)
  - [一句话理解整个系统](#一句话理解整个系统)

---

## setattr(module, "lora", lora) 的含义

给 `module` 动态新增属性 `lora`，等价于 `module.lora = lora`。

**为什么用 setattr 而不是直接赋值？**
- 属性名可以是变量，更灵活
- 在循环里更通用

**LoRA 代码里的作用：**

```python
setattr(module, "lora", lora)
```

在每个被改造的 Linear 层上额外挂一个 LoRA 子模块。

**结构变成：**
```
Linear layer:
    ├── weight (原始参数，冻结)
    ├── bias
    └── lora  ← 新增的模块
```

**挂上去有什么用？**

1. **方便保存 checkpoint**：`model.state_dict()` 会自动包含 `layer.lora.A.weight`、`layer.lora.B.weight`
2. **方便 freeze/unfreeze**：可以统一对 `module.lora.requires_grad_(True)`
3. **方便 debug/inspect**：可以直接 `print(module.lora)`
4. **方便 LoRA merge**：推理时 `W = W + BA`，需要访问 lora 参数

**为什么不 setattr 会怎样？**

仍然可以 `module.forward = ...`，但：
- state_dict 不一定能找到
- 不能统一 freeze
- 不能 merge
- 不好保存结构

**一句话总结：**

`setattr(module, "lora", lora)` 本质是在原始 PyTorch layer 上动态挂一个 LoRA 子模块，让它成为模型结构的一部分，便于训练、保存和管理。

**为什么 LoRA 要"挂在 layer 上"？**

因为 LoRA 本质是每个 Linear layer 的"附加模块"，不是独立模型：

```
Transformer Block
   ├── Linear
   │     ├── W (冻结)
   │     └── LoRA (训练)
```

---

## LoRA 生命周期工具链

**理解为一整套"LoRA 插件系统"：安装 → 训练 → 保存 → 加载 → 合并部署**

### 总体流程图

```
        ┌──────────────┐
        │ apply_lora   │  ← 初始化模型（插插件）
        └──────┬───────┘
               ↓
        训练（只更新 LoRA）
               ↓
        ┌──────────────┐
        │ save_lora    │  ← 保存 LoRA 参数
        └──────┬───────┘
               ↓
        训练结束 / 断点
               ↓
        ┌──────────────┐
        │ load_lora    │  ← 加载 LoRA 继续训练/推理
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ merge_lora   │  ← 合并进 W（部署用）
        └──────────────┘
```

### apply_lora（初始化时）

```python
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora
```

**作用：** 给模型每个 Linear 层外挂 LoRA 分支，改写 forward。

**用在：** 训练开始前

---

### save_lora（训练中/后）

```python
def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
```

**作用：** 只保存 LoRA 参数（A/B矩阵），不保存原模型

**用在：** checkpoint 或训练结束保存 adapter

---

### load_lora（恢复/推理前）

```python
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)
```

**作用：** 把 LoRA 权重重新塞回 model.lora

**用在：** 继续训练、推理前加载 adapter

**前提：** 必须先 `apply_lora(model)`

---

### merge_lora（部署前）

```python
def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
```

**作用：** 把 LoRA 融合进权重 W，生成纯模型

**核心步骤：**
1. `load_lora()` 加载 LoRA
2. `W' = W + B @ A` 合并权重
3. 保存纯模型

**用在：** 推理部署前

---

### 四个函数角色分工

| 函数 | 作用 | 什么时候用 |
|------|------|------------|
| apply_lora | 给模型插 LoRA | 训练前 |
| save_lora | 存 LoRA 参数 | 训练中/结束 |
| load_lora | 恢复 LoRA | 继续训练/推理 |
| merge_lora | 合并成完整模型 | 推理部署前 |

---

### 三种典型使用模式

**模式1：训练 LoRA**
```
apply_lora → train → save_lora
```

**模式2：继续训练**
```
apply_lora → load_lora → train
```

**模式3：推理部署**
```
apply_lora (可选) → load_lora → merge_lora → inference
```

---

### 一句话理解整个系统

- `apply_lora` 是"装插件"
- `save/load_lora` 是"存插件"
- `merge_lora` 是"拆插件并写死进模型"
