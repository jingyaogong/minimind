import torch
from torch import optim, nn


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, lora_alpha=None):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        # 标准 LoRA 的 alpha/r 缩放；lora_alpha=None 表示 alpha=rank（scaling=1），保持历史行为
        scaling = (rank if lora_alpha is None else lora_alpha) / rank
        self.register_buffer("scaling", torch.tensor(scaling, dtype=torch.float32))
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x)) * self.scaling


def apply_lora(model, rank=16, lora_alpha=None):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank, lora_alpha=lora_alpha).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            # 兼容旧 checkpoint：旧版本没有 scaling buffer（等价于 scaling=1），缺失时保留默认值
            module.lora.load_state_dict(lora_state, strict=False)


def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {}
            for k, v in module.lora.state_dict().items():
                # 权重保持 half 以维持文件体积；scaling 等 buffer 用 float32，避免精度损失
                v = v.cpu().half() if k.endswith('.weight') else v.cpu().float()
                lora_state[f'{clean_name}.lora.{k}'] = v
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                delta = module.lora.B.weight.data @ module.lora.A.weight.data
                delta = (delta * module.lora.scaling.to(delta.device)).cpu().half()
                state_dict[f'{name}.weight'] += delta
    torch.save(state_dict, save_path)
