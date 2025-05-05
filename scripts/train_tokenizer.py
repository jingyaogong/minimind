# 分词器训练和评估脚本
# 本脚本实现了基于BPE算法的分词器训练、配置和评估功能
# 主要包含以下功能：
# 1. 使用HuggingFace tokenizers库训练自定义分词器
# 2. 配置特殊token和chat template用于对话格式化
# 3. 评估分词器的编码解码一致性

import random
import json
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
import os

# 设置随机种子以确保结果可复现
random.seed(42)


def train_tokenizer():
    # 内部函数：读取JSONL格式的训练数据
    # 使用生成器模式逐行读取，避免一次性加载全部数据到内存
    def read_texts_from_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['text']

    # 训练数据路径
    data_path = '../dataset/pretrain_hq.jsonl'

    # 初始化BPE分词器
    # BPE（Byte-Pair Encoding）是一种常用的分词算法
    # 它通过迭代合并最频繁出现的字符对来学习子词单元
    tokenizer = Tokenizer(models.BPE())
    # 设置字节级别的预分词器，不在词首添加空格
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 定义特殊token
    # endoftext: 文本结束标记
    # im_start: 对话角色开始标记
    # im_end: 对话角色结束标记
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

    # 配置BPE训练器
    trainer = trainers.BpeTrainer(
        vocab_size=6400,  # 词表大小
        special_tokens=special_tokens,  # 确保特殊token被包含在词表中
        show_progress=True,  # 显示训练进度
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()  # 使用字节级别的初始字母表
    )

    # 读取训练文本数据
    texts = read_texts_from_jsonl(data_path)

    # 训练分词器
    # 使用迭代器训练模式，支持大规模数据集
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置字节级别解码器
    # 确保解码结果与原始文本格式一致
    tokenizer.decoder = decoders.ByteLevel()

    # 验证特殊token的索引
    # 确保特殊token被正确分配到指定的索引位置
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2

    # 创建模型目录并保存分词器
    tokenizer_dir = "../model/"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("../model/")

    # 创建分词器配置
    # 包含token处理规则和chat template配置
    config = {
        "add_bos_token": False,  # 是否在序列开始添加BOS token
        "add_eos_token": False,  # 是否在序列结束添加EOS token
        "add_prefix_space": False,  # 是否在词首添加空格
        "added_tokens_decoder": {  # 特殊token的详细配置
            "0": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],  # 额外的特殊token列表
        "bos_token": "<|im_start|>",  # 序列开始token
        "clean_up_tokenization_spaces": False,  # 是否清理分词时产生的空格
        "eos_token": "<|im_end|>",  # 序列结束token
        "legacy": True,  # 是否使用遗留模式
        "model_max_length": 32768,  # 模型支持的最大序列长度
        "pad_token": "<|endoftext|>",  # 填充token
        "sp_model_kwargs": {},  # SentencePiece模型参数
        "spaces_between_special_tokens": False,  # 特殊token之间是否添加空格
        "tokenizer_class": "PreTrainedTokenizerFast",  # 分词器类名
        "unk_token": "<|endoftext|>",  # 未知token
        # chat template定义了对话格式的模板
        # 支持system、user和assistant三种角色的消息格式化
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<|im_start|>system\n' + system_message + '<|im_end|>\n' }}{% else %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\n' + content + '<|im_end|>\n<|im_start|>assistant\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\n' }}{% endif %}{% endfor %}"
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved.")


def eval_tokenizer():
    """评估分词器的功能
    包括：
    1. chat template的格式化效果
    2. 词表大小统计
    3. 编码长度检查
    4. 编码解码一致性验证
    """
    from transformers import AutoTokenizer

    # 加载训练好的分词器
    tokenizer = AutoTokenizer.from_pretrained("../model/")

    # 构造测试用的对话消息
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    # 应用chat template进行格式化
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print(new_prompt)

    # 统计词表大小
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    # 测试编码功能
    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))

    # 验证编码解码的一致性
    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    print('decoder和原始文本是否一致：', response == new_prompt)


def main():
    # 执行分词器训练和评估
    train_tokenizer()
    eval_tokenizer()


if __name__ == '__main__':
    main()
