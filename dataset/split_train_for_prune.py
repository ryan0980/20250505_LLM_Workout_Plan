import os
import json
import random
import glob
from math import ceil

# 原 train 目录
orig_train = '/root/autodl-tmp/datasets/lawbench3/train'
# 目标根目录
base_out = '/root/autodl-tmp/datasets/lawbench3/valid_for_prune'
prune_dir = os.path.join(base_out, 'prune')
train_dir = os.path.join(base_out, 'train')

# 创建目录
os.makedirs(prune_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)

random.seed(42)

for path in glob.glob(os.path.join(orig_train, '*.json')):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.shuffle(data)
    k = max(1, ceil(len(data) * 0.1))
    prune_subset = data[:k]
    train_subset = data[k:]

    name = os.path.basename(path)
    # 写入 prune/
    with open(os.path.join(prune_dir, name), 'w', encoding='utf-8') as f:
        json.dump(prune_subset, f, ensure_ascii=False, indent=2)
    # 写入 train/
    with open(os.path.join(train_dir, name), 'w', encoding='utf-8') as f:
        json.dump(train_subset, f, ensure_ascii=False, indent=2)

    print(f"{name}: total={len(data)}, prune={len(prune_subset)}, train={len(train_subset)}")
