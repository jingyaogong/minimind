print("开始加载模型")

import argparse
import random
import numpy as np
from transformers import AutoTokenizer, TextStreamer, LlamaForCausalLM
import torch

def init_model():
    transformers_model_path = './MiniMind2'
    tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
    model = LlamaForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
    print(f'模型参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model.eval().to('cuda' if torch.cuda.is_available() else 'cpu'), tokenizer

# 设置可复现的随机种子
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description="Chat with MiniMind")
parser.add_argument('--temp', default=0.85, type=float)
parser.add_argument('--top_p', default=0.85, type=float)

parser.add_argument('--max_len', default=8192, type=int)
args = parser.parse_args()
model, tokenizer = init_model()

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

print("模型加载完毕")

messages = []

random_seed = random.randint(0, 2048)

setup_seed(random_seed)

print("使用的随机种子为：", random_seed)

for prompt in iter(lambda: input('👶: '), ''):
    if prompt == "exit":
        break

    elif print == "clear history":
        messages = []
        print("成功清除对话历史")
        continue

    messages.append({"role": "user", "content": prompt})

    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        new_prompt,
        return_tensors="pt",
        truncation=True
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    print('🤖️: ', end='')
    generated_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=args.max_len,
        num_return_sequences=1,
        do_sample=True,
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
        top_p=args.top_p,
        temperature=args.temp
    )

    response = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    messages.append({"role": "assistant", "content": response})
    print('\n')