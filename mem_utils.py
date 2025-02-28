"""
一些显存调试的工具函数
"""
import torch


def get_model_memory_usage(model):
    total_param_memory = 0
    param_info = {}

    for name, param in model.named_parameters():
        # 计算单个参数的显存（单位：MB）
        param_size = param.data.element_size() * param.data.nelement() / 1024 ** 2
        total_param_memory += param_size
        param_info[name] = {
            "size_mb": f"{param_size:.2f} MB",
            "shape": param.shape,
            "dtype": param.dtype,
        }

    param_info["total_param_memory"] = f"{total_param_memory:.2f} MB"
    return param_info


def get_optimizer_memory_usage(optimizer):
    total_state_memory = 0
    state_info = {}

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p in optimizer.state:
                state = optimizer.state[p]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        # 计算优化器状态张量的显存
                        state_size = value.element_size() * value.nelement() / 1024 ** 2
                        total_state_memory += state_size
                        state_info[f"Param_{id(p)}_{key}"] = {
                            "size_mb": f"{state_size:.2f} MB",
                            "shape": value.shape,
                            "dtype": value.dtype,
                        }

    state_info["total_optimizer_state_memory"] = f"{total_state_memory:.2f} MB"
    return state_info


def log_memory_snapshot(model, optimizer, step):
    print(f"--- Memory Snapshot (Step {step}) ---")

    # 模型参数显存
    model_mem = get_model_memory_usage(model)
    print("Model Parameters:")
    for name, info in model_mem.items():
        print(f"  {name}: {info}")

    # 优化器状态显存
    optimizer_mem = get_optimizer_memory_usage(optimizer)
    print("\nOptimizer State:")
    for name, info in optimizer_mem.items():
        print(f"  {name}: {info}")

    # 总显存占用（模型 + 优化器）
    total_memory = float(model_mem["total_param_memory"].split()[0]) + \
                   float(optimizer_mem["total_optimizer_state_memory"].split()[0])
    print(f"\nTotal Memory: {total_memory:.2f} MB")


class ActivationMemoryTracker:
    def __init__(self):
        self.activation_info = {}
        self.hooks = []

    def track_activations(self, model):
        for name, module in model.named_modules():
            hook = module.register_forward_hook(self._hook_fn(name))
            self.hooks.append(hook)

    def _hook_fn(self, name):
        def hook(_, __, output):
            # 计算激活值的显存（单位：MB）
            if isinstance(output, torch.Tensor):
                mem = output.element_size() * output.nelement() / 1024 ** 2
                self.activation_info[name] = {
                    "size_mb": f"{mem:.2f} MB",
                    "size": mem,
                    "shape": output.shape,
                    "dtype": output.dtype,
                }
        return hook

    def clear(self):
        for hook in self.hooks:
            hook.remove()
        self.activation_info = {}
        self.hooks = []
