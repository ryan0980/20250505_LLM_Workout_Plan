import os
import json
import random
import glob
from math import ceil

# 输入目录（原始数据）
input_dir = '/root/autodl-tmp/datasets/lawbench/zero_shot'

# 输出目录（保存分割后的数据）
output_base = '/root/autodl-tmp/datasets/lawbench3'
train_dir = os.path.join(output_base, 'train')
eval_dir = os.path.join(output_base, 'eval')

# 创建输出目录及子目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

# 获取 input_dir 下所有 .json 文件（排除子目录中的文件）
json_files = glob.glob(os.path.join(input_dir, '*.json'))

for file_path in json_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"读取 {file_path} 出错: {e}")
            continue

    if not isinstance(data, list):
        print(f"{file_path} 中的数据格式不是列表，跳过")
        continue

    # 随机打乱数据顺序
    random.shuffle(data)

    # 按 20% 分割（使用 ceil 确保 eval 至少有一个样本）
    split_index = ceil(len(data) * 0.2)
    eval_data = data[:split_index]
    train_data = data[split_index:]

    base_name = os.path.basename(file_path)
    train_file = os.path.join(train_dir, base_name)
    eval_file = os.path.join(eval_dir, base_name)

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print(f"{base_name}: 共 {len(data)} 条数据, train: {len(train_data)} 条, eval: {len(eval_data)} 条")
