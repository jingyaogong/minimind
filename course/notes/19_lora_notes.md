# Lesson 19 笔记：named_children vs named_modules 详解

## 目录

- [核心区别](#核心区别)
- [named_modules() 的特点](#named_modules-的特点)
- [named_children() 的特点](#named_children-的特点)
- [原项目为什么用 named_modules()](#原项目为什么用-named_modules)
- [如果要做模块替换用什么](#如果要做模块替换用什么)
- [对比总结](#对比总结)

---

## 核心区别

| 方法 | 遍历范围 | 适合场景 |
|------|----------|----------|
| `named_modules()` | 递归所有后代模块 | 读、统计、原地修改 |
| `named_children()` | 只遍历直接子模块 | 替换子模块 |

---

## named_modules() 的特点

`named_modules()` 会递归遍历整棵模块树，返回**所有后代模块**：

```python
for name, module in model.named_modules():
    print(name, module)
```

输出类似：
```
""                  -> 整个 model 自己
"layers"           -> ModuleList
"layers.0"          -> 第 0 个 block
"layers.0.self_attn.q_proj" -> q_proj Linear
"layers.0.self_attn.k_proj" -> k_proj Linear
...
```

**特点：**
- 给你的是完整路径和模块对象本身
- **没有直接给你父模块**
- 适合"查看、统计、收集参数"

---

## named_children() 的特点

`named_children()` 只遍历**直接子模块**：

```python
for name, child in model.named_children():
    print(name, child)
```

输出类似：
```
"embed_tokens" -> Embedding
"layers"       -> ModuleList
"norm"         -> RMSNorm
```

**特点：**
- 每一层递归时，你都站在**父模块**上
- 可以直接 `setattr(model, name, new_child)` 替换子模块
- 需要自己递归进入子模块

---

## 原项目为什么用 named_modules()

原项目 `apply_lora` 的代码：

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

**关键：这里没有替换模块，而是修改现有模块对象本身。**

```
原始：q_proj: nn.Linear

改后：q_proj: nn.Linear (被 monkey-patch 了 forward)
                    └── 多了 .lora 属性
```

- `module.forward = new_forward` 直接改了 module 自己的方法
- `setattr(module, "lora", lora)` 直接在 module 上挂属性
- **不需要父模块参与**，只需要拿到 module 对象

所以 `named_modules()` 完全够用。

---

## 如果要做模块替换用什么

如果想**替换**一个模块，而不是修改它：

```python
# 这样写是无效的！
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        module = LoRALinear(module)  # 只改了局部变量，模型树没变
```

模型里的 `q_proj` 还是原来的 Linear。

**必须改父模块：**

```python
# 解析路径
parent_path, child_name = name.rsplit(".", 1)
parent = model.get_submodule(parent_path)
setattr(parent, child_name, LoRALinear(module))
```

**这比 named_children() 递归麻烦多了。**

用 `named_children()` 递归更自然：

```python
def apply_lora_to_linear_layers(model):
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(model, name, LoRALinear(child))  # 站在父模块上替换
        else:
            apply_lora_to_linear_layers(child)
```

每一层递归时，`model` 就是当前父模块，`name` 是直接子模块名，可以直接 `setattr` 替换。

---

## 对比总结

| 场景 | 用哪个 | 为什么 |
|------|--------|--------|
| monkey-patch forward / 挂属性 | `named_modules()` | 只需要对象本身，不需要父模块 |
| 替换模块 | `named_children()` | 需要父模块才能执行 `setattr(parent, child_name, new_module)` |
| 统计参数 / 收集信息 | `named_modules()` | 要遍历所有模块 |

**一句话总结：**

- `named_modules()` 是"我看到的所有节点"
- `named_children()` 是"我直接管理的子节点 + 我可以替换它们"
