import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用 tokenizer 的并行处理

# 定义 PretrainDataset 类，继承自 Dataset
class PretrainDataset(Dataset):
    def __init__(self, data_path_lst, max_length=512, memmap=False):
        super().__init__()
        # 如果使用内存映射（memmap）
        if memmap:
            with open(data_path_lst[0], 'r') as f:
                nbytes = f.seek(0, 2)  # 获取文件总字节数
                flen = f.tell() // np.dtype('uint16').itemsize  # 计算文件长度
            self.data = np.memmap(data_path_lst[0], dtype=np.dtype('uint16'), shape=(flen // max_length, max_length))  # 使用内存映射加载数据
        else:
            data_lst = []
            for data_path in data_path_lst:
                with open(data_path, 'rb') as f:
                    data = np.fromfile(f, dtype=np.uint16)  # 从文件中读取数据
                    data_lst.append(data)
            data = np.concatenate(data_lst)  # 合并所有数据
            data = data[:max_length * int(len(data) / max_length)]  # 截取数据
            # np.random.shuffle(data)  # 打乱数据（注释掉了）
            self.data = data.reshape(-1, max_length)  # 将数据重塑为 (样本数, 最大长度) 的形状
        # 打印数据形状
        print("memmap:{} train data.shape:{}".format(memmap, self.data.shape))
        print("downloading finished.....")

    def __len__(self):
        return self.data.shape[0]  # 返回数据集的长度

    def __getitem__(self, index: int):
        # 获取指定索引的样本
        sample = self.data[index]
        X = np.array(sample[:-1]).astype(np.int64)  # 输入数据（去掉最后一个 token）
        Y = np.array(sample[1:]).astype(np.int64)  # 目标数据（去掉第一个 token）

        return torch.from_numpy(X), torch.from_numpy(Y)  # 返回 PyTorch 张量

# 定义 SFTDataset 类，继承自 Dataset
class SFTDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=1024, prompt_max_len=512, answer_max_len=256):
        super().__init__()
        self.df = df  # 数据框
        self.max_length = max_length  # 最大序列长度
        self.prompt_max_len = prompt_max_len  # 提示的最大长度
        self.answer_max_len = answer_max_len  # 回答的最大长度
        #
        self.tokenizer = tokenizer  # 分词器
        self.padding = 0  # 填充 token ID
        self.bos_id = self.tokenizer('<s>assistant').data['input_ids']  # 开始 token ID

    def __len__(self):
        return self.df.shape[0]  # 返回数据集的长度

    def find_sublist_index(self, main_list, sub_list) -> int:
        last_index = -1
        for i in range(len(main_list) - len(sub_list) + 1):
            if main_list[i:i + len(sub_list)] == sub_list:
                last_index = i
        return last_index  # 查找子列表在主列表中的最后一个索引

    def safe_eval(self, s):
        try:
            res = eval(s)
        except Exception as e:
            return []
        return res  # 安全地执行 eval 函数

    def __getitem__(self, index: int):
        # 获取指定索引的样本
        sample = self.df.iloc[index]
        history = self.safe_eval(sample['history'])  # 获取历史对话
        q = str(sample['q'])  # 获取问题
        a = str(sample['a'])  # 获取回答

        messages = []
        for history_message in history:
            if len(history_message) <= 1:
                continue
            messages.append(
                {"role": 'user', "content": str(history_message[0])[:self.max_length // 2]}
            )
            messages.append(
                {"role": 'assistant', "content": str(history_message[1])[:self.max_length // 2]}
            )

        messages += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        new_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )  # 生成新的提示
        input_id = self.tokenizer(new_prompt).data['input_ids'][:self.max_length]  # 分词并截取

        # 实际长度
        question_length = self.find_sublist_index(input_id, self.bos_id) + len(self.bos_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - len(input_id)
        input_id = input_id + [self.padding] * padding_len  # 填充到最大长度
        mask_len = len(input_id) - question_length - padding_len
        # 0表示不计算损失
        loss_mask = [0] * question_length + [1] * (mask_len) + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)  # 输入数据（去掉最后一个 token）
        Y = np.array(input_id[1:]).astype(np.int64)  # 目标数据（去掉第一个 token）
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)  # 损失掩码

        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        loss_mask_tensor = torch.from_numpy(loss_mask)

        return X_tensor, Y_tensor, loss_mask_tensor  # 返回 PyTorch 张量

# 主函数
if __name__ == "__main__":
    pass