import os
import sys
__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import warnings
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
import argparse
warnings.filterwarnings('ignore', category=UserWarning)

# MiniMind 配置（需与训练时一致）
lm_config = MiniMindConfig(
    hidden_size=512,
    num_hidden_layers=8,
    max_position_embeddings=32768,
    vocab_size=6400,  # 确保与 tokenizer 一致
    use_moe=False,
    num_attention_heads=8,
    num_key_value_heads=2,
    rms_norm_eps=1e-5,
    rope_theta=10000.0
)


def merge_lora_weights(model, lora_state_dict):
    """
    将 LoRA 权重合并到基础模型中
    LoRA 权重格式: {module_name}.lora.A.weight 和 {module_name}.lora.B.weight
    合并公式: W_new = W_old + B @ A
    
    注意: 根据 model_lora.py，LoRA 只应用于方阵线性层 (shape[0] == shape[1])
    """
    # 清理 state_dict 中的 'module.' 前缀（如果有）
    cleaned_lora_dict = {k.replace('module.', ''): v for k, v in lora_state_dict.items()}
    
    # 收集所有 LoRA 权重
    lora_weights = {}
    for key in cleaned_lora_dict.keys():
        if '.lora.' in key:
            # 提取模块名称和权重类型
            parts = key.split('.lora.')
            if len(parts) == 2:
                module_name = parts[0]
                weight_type = parts[1]  # 'A.weight' 或 'B.weight'
                
                if module_name not in lora_weights:
                    lora_weights[module_name] = {}
                lora_weights[module_name][weight_type] = cleaned_lora_dict[key]
    
    print(f"找到 {len(lora_weights)} 个应用了 LoRA 的模块")
    
    # 合并 LoRA 权重到对应的线性层
    merged_count = 0
    skipped_count = 0
    
    for name, module in model.named_modules():
        # 只处理线性层
        if isinstance(module, torch.nn.Linear):
            # 检查是否有对应的 LoRA 权重
            if name in lora_weights:
                lora_data = lora_weights[name]
                if 'A.weight' in lora_data and 'B.weight' in lora_data:
                    A = lora_data['A.weight']  # shape: [rank, in_features]
                    B = lora_data['B.weight']  # shape: [out_features, rank]
                    
                    # 验证形状匹配
                    original_weight = module.weight.data  # shape: [out_features, in_features]
                    expected_out, expected_in = original_weight.shape
                    
                    # A: [rank, in_features], B: [out_features, rank]
                    if A.shape[1] != expected_in or B.shape[0] != expected_out:
                        print(f"  ⚠ 跳过 {name}: 形状不匹配 (A: {A.shape}, B: {B.shape}, 期望: [{expected_out}, {expected_in}])")
                        skipped_count += 1
                        continue
                    
                    # 合并: W_new = W_old + B @ A
                    # B @ A: [out_features, rank] @ [rank, in_features] = [out_features, in_features]
                    lora_delta = torch.matmul(B, A)  # shape: [out_features, in_features]
                    module.weight.data = original_weight + lora_delta.to(original_weight.device)
                    merged_count += 1
                    print(f"  ✓ 合并 LoRA 权重到: {name} (形状: {original_weight.shape}, rank: {A.shape[0]})")
    
    print(f"成功合并 {merged_count} 个 LoRA 权重")
    if skipped_count > 0:
        print(f"跳过 {skipped_count} 个不匹配的权重")
    
    return model


def merge_lora_to_minimind(lora_model_path, base_model_path, output_path, config=None, dtype=torch.float16):
    """
    将 MiniMind + LoRA 权重合并并保存为 safetensors 格式
    
    Args:
        lora_model_path: LoRA 权重文件路径 (.pth)
        base_model_path: 基础模型文件路径 (.pth)
        output_path: 输出目录路径
        config: MiniMindConfig 配置对象
        dtype: 输出模型的数据类型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 使用提供的配置或默认配置
    if config is None:
        config = lm_config
    
    print("=" * 60)
    print("开始合并 LoRA 权重到 MiniMind 模型")
    print("=" * 60)
    
    # 1. 加载基础模型
    print(f"\n1. 加载基础模型: {base_model_path}")
    model = MiniMindForCausalLM(config)
    
    if os.path.isfile(base_model_path):
        state_dict = torch.load(base_model_path, map_location='cpu')
        
        # 如果 checkpoint 包含 'model' 键，则提取模型权重
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
            print("  ℹ 检测到完整 checkpoint，提取模型权重")
        
        # 清理 state_dict 中的 'module.' 前缀（如果有，用于 DDP 模型）
        cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # 过滤出模型相关的键（排除 optimizer、scheduler 等）
        model_keys = set(model.state_dict().keys())
        filtered_state_dict = {k: v for k, v in cleaned_state_dict.items() if k in model_keys}
        
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        print("  ✓ 基础模型加载完成")
        if missing_keys:
            print(f"  ⚠ 警告: {len(missing_keys)} 个键未找到（已忽略）")
        if unexpected_keys:
            print(f"  ℹ 信息: {len(unexpected_keys)} 个额外键（已忽略）")
    else:
        raise FileNotFoundError(f"基础模型文件不存在: {base_model_path}")
    
    # 2. 加载 LoRA 权重
    print(f"\n2. 加载 LoRA 权重: {lora_model_path}")
    if not os.path.isfile(lora_model_path):
        raise FileNotFoundError(f"LoRA 权重文件不存在: {lora_model_path}")
    
    lora_state_dict = torch.load(lora_model_path, map_location='cpu')
    print(f"  ✓ LoRA 权重加载完成 (包含 {len(lora_state_dict)} 个键)")
    
    # 3. 合并 LoRA 权重
    print(f"\n3. 合并 LoRA 权重到基础模型...")
    model = merge_lora_weights(model, lora_state_dict)
    
    # 4. 转换精度
    print(f"\n4. 转换模型精度为 {dtype}...")
    model = model.to(dtype)
    model.eval()
    
    # 5. 保存模型为 safetensors 格式
    print(f"\n5. 保存模型到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # 注册模型类以便 transformers 能够识别
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    
    # 保存模型（使用 safe_serialization=True 保存为 safetensors）
    model.save_pretrained(output_path, safe_serialization=False)
    print("  ✓ 模型已保存为 safetensors 格式")
    
    # 6. 保存 tokenizer
    print(f"\n6. 保存 tokenizer...")
    tokenizer_path = os.path.join(os.path.dirname(__file__), '..', 'model')
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.save_pretrained(output_path)
        print("  ✓ Tokenizer 保存完成")
    else:
        print(f"  ⚠ 警告: Tokenizer 路径不存在: {tokenizer_path}")
    
    # 7. 打印参数量
    print(f"\n7. 模型信息:")
    model_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {model_params / 1e6:.2f} M = {model_params / 1e9:.2f} B")
    print(f"  可训练参数: {trainable_params / 1e6:.2f} M")
    
    print("\n" + "=" * 60)
    print(f"✅ LoRA 已合并并保存到: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    # 配置路径
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base MiniMind model")
    parser.add_argument(
        "--lora_model_path",
        type=str,
        required=True,
        # default="/home/dieu/minimind/out/lora/lora_identity_512.pth",
        help="Path to the LoRA weights (.pth file)"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        # default="/home/dieu/minimind/out/full_sft_512.pth",
        help="Path to the base model (.pth file)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        # default="/home/dieu/minimind/out/merged_model_test",
        help="Directory to save the merged model"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for the merged model (default: float16)"
    )

    args = parser.parse_args()

    # 映射 dtype 字符串到 torch.dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    # 调用合并函数
    merge_lora_to_minimind(
        lora_model_path=args.lora_model_path,
        base_model_path=args.base_model_path,
        output_path=args.output_path,
        config=lm_config,
        dtype=dtype
    )
