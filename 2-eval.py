import random
import time

import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import Transformer
from model.LMConfig import LMConfig

warnings.filterwarnings('ignore')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model_from = 1  # 1从权重，2用transformers

    if model_from == 1:
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'./out/full_sft_{lm_config.dim}{moe_path}.pth'

        model = Transformer(lm_config)
        state_dict = torch.load(ckp, map_location=device)

        # 处理不需要的前缀
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        for k, v in list(state_dict.items()):
            if 'mask' in k:
                del state_dict[k]

        # 加载到模型中
        model.load_state_dict(state_dict, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained('./minimind-v1-small', trust_remote_code=True)
    model = model.to(device)

    print(f'模型参数: {count_parameters(model) / 1e6} 百万 = {count_parameters(model) / 1e9} B (Billion)')
    return model, tokenizer


def setup_seed(seed):
    random.seed(seed)  # 设置 Python 的随机种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
    torch.cuda.manual_seed(seed)  # 为当前 GPU 设置随机种子（如果有）
    torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置随机种子（如果有）
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 关闭 cuDNN 的自动调优，避免不确定性


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    out_dir = 'out'
    start = ""
    temperature = 0.7
    top_k = 16
    # device = 'cpu'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16'
    max_seq_len = 1 * 1024
    lm_config = LMConfig()
    lm_config.max_seq_len = max_seq_len
    # 对话是否携带历史对话（当前模型没有在连续对话数据集上训练，增大历史上文基本不会有新的问答能力）
    contain_history_chat = False
    # -----------------------------------------------------------------------------

    model, tokenizer = init_model(lm_config)

    model = model.eval()
    # 推送到huggingface
    # model.push_to_hub("minimind")
    # tokenizer.push_to_hub("minimind")

    # answer_way = int(input('输入0自动测试，输入1问题测试：'))
    answer_way = 0
    stream = True

    prompt_datas = [
        '你叫什么名字',
        '你是谁',
        '中国有哪些比较好的大学？',
        '全世界最好的大学是什么？',
        '你知道光速是多少吗？',
        '你知道长江吗？',
        '人类的血液主要由哪些成分组成？',
        '第一颗人造卫星是哪个国家发射的？',
        '你知道杭州有什么美食吗？',
        '你知道泰山在哪里吗？',
        '地球上最大的动物是什么？',
        '地球自转一圈大约需要多少时间？',
        '人类最早使用的金属是什么？',
        '水的化学分子式是什么？',
        '大气层中含量最多的气体是什么？',
        '世界上最高的山峰是什么？',
        '你知道世界上最深的海沟是什么吗？',
        '最早发明印刷术的是哪个国家？',
        '万有引力是谁提出的？',
        '光合作用的主要原理是什么？',
        '你知道大熊猫的主要食物是什么吗？',
        '海水为什么是咸的？',
        '我们平时喝的牛奶主要含有什么营养成分？',
        '一星期有多少天？'
    ]

    messages_origin = []
    messages = messages_origin

    i = 0
    while i < len(prompt_datas):
        # Generate a random seed
        random_seed = random.randint(0, 2 ** 32 - 1)
        setup_seed(random_seed)
        if not contain_history_chat:
            messages = messages_origin.copy()

        if answer_way == 1:
            prompt = input('[Q]: ')
        else:
            prompt = prompt_datas[i]
            print(f'[Q]: {prompt}')
            i += 1

        prompt = '请问，' + prompt
        messages.append({"role": "user", "content": prompt})

        # print(messages)
        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-(max_seq_len - 1):]

        x = tokenizer(new_prompt).data['input_ids']
        x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])

        answer = new_prompt

        with torch.no_grad():
            res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=max_seq_len, temperature=temperature,
                                   top_k=top_k, stream=stream)
            print('[A]: ', end='')
            try:
                y = next(res_y)
            except StopIteration:
                print("No answer")
                continue

            history_idx = 0
            while y != None:
                answer = tokenizer.decode(y[0].tolist())
                if answer and answer[-1] == '�':
                    try:
                        y = next(res_y)
                    except:
                        break
                    continue
                # print(answer)
                if not len(answer):
                    try:
                        y = next(res_y)
                    except:
                        break
                    continue

                print(answer[history_idx:], end='', flush=True)
                try:
                    y = next(res_y)
                except:
                    break
                history_idx = len(answer)
                if not stream:
                    break

            print('\n')

        if contain_history_chat:
            assistant_answer = answer.replace(new_prompt, "")
            messages.append({"role": "assistant", "content": assistant_answer})
