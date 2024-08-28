import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path_lst, max_length=512, memmap=False):
        super().__init__()
        #
        if memmap:
            with open(data_path_lst[0], 'r') as f:
                nbytes = f.seek(0, 2)
                flen = f.tell() // np.dtype('uint16').itemsize
            self.data = np.memmap(data_path_lst[0], dtype=np.dtype('uint16'), shape=(flen // max_length, max_length))
        else:
            data_lst = []
            for data_path in data_path_lst:
                with open(data_path, 'rb') as f:
                    data = np.fromfile(f, dtype=np.uint16)
                    data_lst.append(data)
            data = np.concatenate(data_lst)
            data = data[:max_length * int(len(data) / max_length)]
            # np.random.shuffle(data)
            self.data = data.reshape(-1, max_length)
        #
        print("memmap:{} train data.shape:{}".format(memmap, self.data.shape))
        print("downloading finished.....")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        #
        sample = self.data[index]
        X = np.array(sample[:-1]).astype(np.int64)
        Y = np.array(sample[1:]).astype(np.int64)

        return torch.from_numpy(X), torch.from_numpy(Y)


class SFTDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=1024, prompt_max_len=512, answer_max_len=256):
        super().__init__()
        self.df = df
        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len
        #
        self.tokenizer = tokenizer
        self.padding = 0  # self.tokenizer.special_tokens['<pad>']
        self.bos_id = self.tokenizer('<s>assistant').data['input_ids']

    def __len__(self):
        return self.df.shape[0]

    def find_sublist_index(self, main_list, sub_list) -> int:
        last_index = -1
        for i in range(len(main_list) - len(sub_list) + 1):
            if main_list[i:i + len(sub_list)] == sub_list:
                last_index = i
        return last_index

    def __getitem__(self, index: int):
        #
        sample = self.df.iloc[index]
        history = eval(sample['history'])
        q = sample['q']
        a = sample['a']

        messages = []
        for history_message in history:
            if len(history_message) <= 1:
                continue
            messages.append(
                {"role": 'user', "content": history_message[0][:self.max_length // 2]}
            )
            messages.append(
                {"role": 'assistant', "content": history_message[1][:self.max_length // 2]}
            )

        messages += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        new_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_id = self.tokenizer(new_prompt).data['input_ids'][:self.max_length]

        # 实际长度
        question_length = self.find_sublist_index(input_id, self.bos_id) + len(self.bos_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - len(input_id)
        input_id = input_id + [self.padding] * padding_len
        mask_len = len(input_id) - question_length - padding_len
        # 0表示不计算损失
        loss_mask = [0] * question_length + [1] * (mask_len) + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        loss_mask_tensor = torch.from_numpy(loss_mask)

        return X_tensor, Y_tensor, loss_mask_tensor


if __name__ == "__main__":
    pass
