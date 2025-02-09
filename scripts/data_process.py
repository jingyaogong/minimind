import csv
import glob
import os
import re
import json
import jsonlines
import pandas as pd
from tqdm import tqdm

bos_token = "<s>"
eos_token = "</s>"


def pretrain_process():
    # 定义输入和输出路径
    input_dir = '../CCI3-HQ/data'
    output_file = '../dataset/pretrain_data_hq.csv'
    jsonl_files = glob.glob(os.path.join(input_dir, 'part_*.jsonl'))
    total_lines = 0
    print("正在计算总行数...")
    for file in jsonl_files:
        with open(file, 'r', encoding='utf-8') as f:
            for _ in f:
                total_lines += 1
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text', 'score'])  # 写入表头
        for jsonl_file in jsonl_files:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f'处理 {os.path.basename(jsonl_file)}', total=total_lines, unit='行',
                                 leave=False):
                    try:
                        data = json.loads(line)
                        text = data.get('text', '')
                        score = data.get('score', 0)
                        if len(text) <= 512 and score > 3.5:
                            writer.writerow([text, score])
                    except json.JSONDecodeError:
                        continue
    print(f"筛选完成，结果已保存到 {output_file}")


def sft_process():
    sft_file_name = 'sft_data.csv'

    def process_and_write_data(data):
        q_lst, a_lst, history_lst = [], [], []
        for per in data:
            history, q, a = per['history'], per['q'], per['a']
            if not q or not a:
                continue
            history_len = sum(len(s) for s in history)
            message_len = history_len + len(q) + len(a)
            if message_len < 70 or message_len > 512:
                continue
            q_lst.append(q)
            a_lst.append(a)
            history_lst.append(history)

        df = pd.DataFrame({'history': history_lst, 'q': q_lst, 'a': a_lst})
        df.to_csv(f'../dataset/{sft_file_name}',
                  mode='a', header=False, index=False,
                  lineterminator='\r\n', escapechar='\\', encoding='utf-8')

    chunk_size = 1000
    data = []
    with open(f'../dataset/{sft_file_name}', 'w', encoding='utf-8') as f:
        f.write('history,q,a\n')

    # sft_path = ['/root/shared-nvme/sft_data_zh.jsonl', '/root/shared-nvme/sft_data_en.jsonl']
    sft_path = ['/root/shared-nvme/sft_data_en.jsonl']
    chunk_num = 0
    for path in sft_path:
        with jsonlines.open(path) as reader:
            for idx, obj in enumerate(reader):
                try:
                    data.append({
                        'history': obj.get('history', ''),
                        'q': obj.get('input', '') + obj.get('q', ''),
                        'a': obj.get('output', '') + obj.get('a', '')
                    })

                    if len(data) >= chunk_size:
                        chunk_num += 1
                        process_and_write_data(data)
                        data = []
                        if chunk_num % 100 == 0:
                            print(f'chunk:{chunk_num} process end')
                except jsonlines.InvalidLineError as e:
                    print(f"Skipping invalid JSON line {idx + 1}: {e}")
                    continue

            if data:
                process_and_write_data(data)
                data = []


def rl_process():
    # 偏好数据默认只用中文（建议）
    input_paths = [
        # "../dataset/dpo_en.json",
        "../dataset/dpo_zh.json"
    ]
    output_path = "../dataset/dpo_data.jsonl"  # 修改输出文件扩展名为 .jsonl
    all_converted = []

    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # data is likely a list

        for item in data:
            new_data = {
                "chosen": [],
                "rejected": []
            }
            for turn in item["conversations"]:
                role = "user" if turn["from"] == "human" else "assistant"
                message = {"role": role, "content": turn["value"]}
                new_data["chosen"].append(message)
                new_data["rejected"].append(message)
            new_data["chosen"].append({
                "role": "assistant",
                "content": item["chosen"]["value"]
            })
            new_data["rejected"].append({
                "role": "assistant",
                "content": item["rejected"]["value"]
            })
            all_converted.append(new_data)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def lora_dataset():
    import json
    import csv

    # 读取JSON文件
    with open('../dataset/Chinese-medical-dialogue.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 准备CSV数据
    csv_data = []
    for item in data:
        # 提取input和output并去除首尾空白
        q = item['input'].strip()
        a = item['output'].strip()

        # 检查长度是否符合要求
        if len(q) + len(a) < 160:
            csv_data.append({
                'history': '[]',
                'q': q,
                'a': a
            })

    # 写入CSV文件
    with open('../dataset/medical_sft.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['history', 'q', 'a']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(csv_data)

    print(f'转换完成，共处理 {len(csv_data)} 条有效数据')


if __name__ == "__main__":
    ################
    # 1: pretrain
    # 2: sft
    # 3: RL
    ################
    process_type = 4

    if process_type == 1:
        pretrain_process()
    if process_type == 2:
        sft_process()
    if process_type == 3:
        rl_process()
    if process_type == 4:
        lora_dataset()
