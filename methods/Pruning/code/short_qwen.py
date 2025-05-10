#!/usr/bin/env python3
# short_qwen_xin.py - 针对 Qwen2.5-3B 模型的剪枝脚本

import sys
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import cosine_similarity
from datasets import load_dataset

# —— 1. 读取参数 & 创建输出目录 —— 
# 如果你打算写死模型路径，就直接赋值；否则改成 model_path = sys.argv[1]
model_path  = "/root/autodl-tmp/model/q3bo"
output_dir  = "/root/autodl-tmp/model/q3bp_attempt1"
os.makedirs(output_dir, exist_ok=True)

# —— 2. 加载模型 & tokenizer —— 
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16
)
model.eval()
model.to('cuda')

# —— 3. 计算剪枝层数 —— 
total_layers    = model.config.num_hidden_layers
prune_layer_num = int(total_layers * 0.20)
print(f"模型共有 {total_layers} 层，将剪掉 {prune_layer_num} 层（20%）。")

# —— 4. 用 WikiText 验证集计算 BI 分数 —— 
dataset      = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
texts        = dataset["text"]
layer_cos_sum = np.zeros(total_layers, dtype=np.float64)
token_count   = np.zeros(total_layers, dtype=np.int64)

print("正在计算每层的 Block Influence（BI）分数……")
for text in texts:
    if not text.strip():
        continue
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to('cuda')
    mask      = inputs["attention_mask"].to('cuda')

    outputs = model(input_ids, attention_mask=mask, output_hidden_states=True)
    hids    = outputs.hidden_states  # length = total_layers+1

    for i in range(total_layers):
        h0 = hids[i][0]    # (seq_len, dim)
        h1 = hids[i+1][0]
        cos = cosine_similarity(h0, h1, dim=-1)
        layer_cos_sum[i] += cos.sum().item()
        token_count[i]   += cos.size(0)

# —— 5. 计算 BI 分数 & 排序剪枝 —— 
bi_scores = []
for i in range(total_layers):
    avg_cos = (layer_cos_sum[i] / token_count[i]) if token_count[i] else 1.0
    bi      = 1.0 - avg_cos
    bi_scores.append(bi)
    print(f"层 {i} BI 分数 = {bi:.6f}")

# 挑出 BI 最低的那些层
ranked = sorted(range(total_layers), key=lambda i: bi_scores[i])
to_prune = ranked[:prune_layer_num]
print(f"将剪掉的层：{to_prune}")

for idx in sorted(to_prune, reverse=True):
    del model.model.layers[idx]

model.config.num_hidden_layers = total_layers - prune_layer_num

# —— 6. 保存结果 —— 
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"剪枝完成，模型保存在：{output_dir}")
