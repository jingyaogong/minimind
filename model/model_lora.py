import torch
from torch import optim, nn


class LoRA(nn.Module):
    """
    Lora 网络结构
    """
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        # LoRA 的秩 (rank)
        self.rank = rank
        # 低秩矩阵 A & B
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        # 低秩矩阵初始化
        # A: 高斯分布
        # B: 全零
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        # 前向传播
        return self.B(self.A(x))


def apply_lora(model, rank=8):
    """
    对指定的 model 添加 lora 层
    """
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 检查是否为 "线性层" 且 "权重矩阵为方阵"
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 创建 LoRA 层并移动到模型设备上
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            # 针对 model 设定一个属性值: 属性: "lora" / 值: lora
            # https://docs.python.org/zh-cn/3.14/library/functions.html#setattr
            setattr(module, "lora", lora)
            
            # 保存原始前向传播函数
            original_forward = module.forward

            # 定义新的前向传播函数, 用于在原始前向传播基础上加入 LoRA 层
            # 这里显示绑定了 layer1=original_forward, layer2=lora 这两个参数
            def forward_with_lora(x, layer1=original_forward, layer2=lora): 
                # 返回原始前向传播结果 & 加上 LoRA 层的输出
                return layer1(x) + layer2(x)

            # 替换模块的前向传播函数
            module.forward = forward_with_lora


def load_lora(model, path):
    """
    加载 LoRA 权重文件并应用到模型

    :param model: 包含初始化的 LoRA 层的模型对象, 即不包含训练权重, 只包含结构和空的 LoRA 参数
    :param path:  LoRA 权重文件路径, 包含所有 A/B 矩阵的 state_dict
    """
    # 从文件加载 state_dict, 从而获得完整的模型权重
    state_dict = torch.load(path, map_location=model.device)
    # 处理模型权重的键名, 移除 state_dict 中所有键名前的 "module." 前缀
        # 如果使用了 DP/DDP 这样的数据并行, 那么 pytorch 就会在模型的权重字典中添加 "module." 前缀来表示这些
        # 如果没有用 DP/DDP 来包装模型, 那么在使用 model.load_state_dict() 时就会报错
        # 因此这里移除掉 state_dict 中每层的  "module." 前缀, 因为这里 lora 微调时不会采用多卡
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
    # #### 代码等效 (推荐字典推导式, 这里仅作理解) ####
    # tmp_state_dict = {}
    # for k, v in state_dict.items():
    #     if k.startswith('module.'):
    #         # 移除前缀 "module."
    #         k = k[7:]
    #     tmp_state_dict[k] = v
    # state_dict = tmp_state_dict
    # ##############################################

    # 遍历 model 的所有模块, 并用 state_dict 中的 数值替换 model 中的数值 (针对 lora 层)
    for name, module in model.named_modules():
        # 检查模块是否有 lora 属性
        # https://docs.python.org/zh-cn/3.14/library/functions.html#hasattr
        if hasattr(module, 'lora'):
            # 提取与当前模块相关的 LoRA 状态
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            # 加载 LoRA 层的状态字典
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """
    仅保存模型中 LoRA 适配器的权重参数, 不保存主干模型
    此函数适用于使用 Hugging Face PEFT 库进行 LoRA 微调的场景

    :param model: 训练完成的模型对象 (可以是 PeftModel 或原始 nn.Module)
    :param path:  保存路径
    """
    # 获取原始模型; 若 model 是 PEFT 包装后的 PeftModel
    # 则通过 _orig_mod 获取其封装的原始模型; 否则直接使用当前模型
        # https://docs.python.org/zh-cn/3.14/library/functions.html#getattr
        # getattr(object, name, default):
        # 如果对象有这个属性, 就返回它; 没有就返回默认值
    raw_model = getattr(model, '_orig_mod', model)
    # 在使用 Hugging Face 的 PEFT 库时, PEFT 会 "包装" 原始模型
        # 此时的 model 不再是原来的 AutoModelForCausalLM, 而是一个 PeftModel 类的对象
        # 它内部有一个属性 _orig_mod, 用来指向原始模型
    # 因此, 这行代码的实际作用如下:
        # 如果 model 是一个 PEFT 包装过的, 那么就取原始模型
        # 如果 model 没有经过 PEFT 包装, 则直接使用当前模型
    
    # 创建一个空的状态字典用于存储模型
    state_dict = {}
    for name, module in raw_model.named_modules():
        # 检查模块是否有 lora 属性
        if hasattr(module, 'lora'):
            # 处理模块名称，去除 "module." 缀
                # 如果使用了 DP/DDP 这样的数据并行, 那么 pytorch 就会在模型的权重字典中添加 "module." 前缀来表示这些
                # 如果没有用 DP/DDP 来包装模型, 那么在使用 model.load_state_dict() 时就会报错
            clean_name = name[7:] if name.startswith("module.") else name
            # 构建 LoRA 层 的 lora_state
            lora_state = {f'{clean_name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            # 更新状态字典
            state_dict.update(lora_state)

    # 保存状态字典到文件 (只保存 lora 层的权重)
    torch.save(state_dict, path)