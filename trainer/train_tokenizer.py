# 注: 不建议再重复训练 tokenizer("词典"), MiniMind 已自带, 此脚本仅供学习和参考
# 基于不同词典训练的模型将导致输出完全不统一, 降低社区的模型复用性
# Note: It is not recommended to re-train the tokenizer. MiniMind already includes one. This script is for learning and reference only. 
# Training models with different tokenizers will lead to inconsistent outputs and reduce model reusability in the community.
import os
import json
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

# 训练数据文件路径
DATA_PATH = '../dataset/pretrain_hq.jsonl'
# 训练好的 tokenizer 保存目录
TOKENIZER_DIR = '../model_learn_tokenizer/'
# 词表大小, 控制 tokenizer 的词汇量
VOCAB_SIZE = 6400


def get_texts(data_path):
    """
    从 JSONL 文件中逐行读取文本数据

    这是一个生成器函数, 用于逐行读取数据文件并提取文本内容
    最多读取前 10000 行用于实验性训练

    Args:
    - data_path: JSONL 格式的数据文件路径

    Yields:
    - str: 每行数据中 'text' 字段的内容
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # 实验性, 可只用前 10000 行测试
            if i >= 10000:
                break
            data = json.loads(line)
            yield data['text']


def train_tokenizer(data_path, tokenizer_dir, vocab_size):
    """
    训练 BPE(Byte Pair Encoding) tokenizer

    使用 Hugging Face tokenizers 库训练自定义的 BPE tokenizer
    包括配置预分词器、训练器、保存词表和配置文件等步骤

    Args:
    - data_path:        训练数据文件路径
    - tokenizer_dir:    tokenizer 保存目录
    - vocab_size:   目标词表大小
    """
    # 创建 BPE 模型的 tokenizer 实例
    tokenizer = Tokenizer(models.BPE())
    # 设置 ByteLevel 预分词器, 不添加前缀空格
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 配置 BPE 训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        # 定义特殊 token: 文本结束符、指令开始符、指令结束符
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        show_progress=True,
        # 使用 ByteLevel 预分词器的初始字母表
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 获取训练文本数据
    texts = get_texts(data_path)
    # 使用迭代器训练 tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)
    # 设置 ByteLevel 解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 验证特殊 token 的 ID 是否符合预期
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2

    # 创建保存目录 (如果不存在)
    os.makedirs(tokenizer_dir, exist_ok=True)
    # 保存完整的 tokenizer 配置
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    # 保存 BPE 模型文件 (vocab.json 和 merges.txt)
    tokenizer.model.save(tokenizer_dir)

    # 构建 Hugging Face 格式的 tokenizer 配置
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
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
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n {%- if messages[0]['role'] == 'system' -%}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else -%}\n        {{- '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}\n {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n   {{- '<|im_start|>' + message.role + '\\n' + content }}\n  {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
    }

    # 保存 tokenizer 配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print("Tokenizer training completed.")


def eval_tokenizer(tokenizer_dir):
    """
    评估和测试训练好的 tokenizer

    加载训练好的 tokenizer, 使用对话模板生成 prompt,
    并测试编码、解码的一致性和流式解码效果。

    Args:
    - tokenizer_dir: tokenizer 模型目录路径
    """
    from transformers import AutoTokenizer

    # 加载训练好的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # 构造测试对话消息
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人, 总是给我正确的回应!"},
        {"role": "user", "content": '你来自哪里?'},
        {"role": "assistant", "content": '我来自地球'}
    ]

    # 应用对话模板生成 prompt
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print('-' * 100)
    print(new_prompt)

    print('-' * 100)
    # 打印 tokenizer 词表长度
    print('tokenizer 词表长度:', len(tokenizer))
    # 对 prompt 进行编码
    model_inputs = tokenizer(new_prompt)
    print('encoder 长度:', len(model_inputs['input_ids']))
    # 解码并验证一致性
    response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
    print('decoder 一致性:', response == new_prompt, "\n")

    print('-' * 100)
    print('流式解码(字节缓冲)测试:')
    input_ids = model_inputs['input_ids']
    token_cache = []
    for tid in input_ids:
        token_cache.append(tid)
        # 尝试解码当前缓存的 token
        current_decode = tokenizer.decode(token_cache)
        # 当解码结果有效且不包含 Unicode 替换字符时输出
        if current_decode and '\ufffd' not in current_decode:
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache
            raw_tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in (token_cache if isinstance(token_cache, list) else [token_cache])]
            print(f'Token ID: {str(display_ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}')
            token_cache = []


if __name__ == '__main__':
    train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)
    eval_tokenizer(TOKENIZER_DIR)
