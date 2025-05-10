#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
short_qwen_xin.py

针对 Qwen2.5‑3B 模型的剪枝脚本，使用本地 JSON 数组格式的“prune”文件夹，
仅取前 200 条样本计算 Block Influence（BI），剪掉最低 20% 的层。
"""

import os
import glob
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import cosine_similarity

# —— 1. 配置路径 & 创建输出目录 —— 
MODEL_PATH = "/root/autodl-tmp/model/q3bft_q"
OUTPUT_DIR = "/root/autodl-tmp/model/q3bft_q_p"
PRUNE_DIR  = "/root/autodl-tmp/datasets/lawbench3/valid_for_prune/prune"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# —— 2. 手动加载本地 JSON 文件 —— 
print("🔍 正在加载本地 prune 文本…")
all_texts = []
for json_path in glob.glob(os.path.join(PRUNE_DIR, "*.json")):
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    for rec in records:
        all_texts.append(rec["question"])  # 可根据需要改为 rec["text"] 或拼接字段

# 仅取前 200 条样本
texts = all_texts[:200]
print(f"✅ 一共加载到 {len(all_texts)} 条文本，当前仅使用前 {len(texts)} 条进行 BI 计算\n")

# —— 3. 加载模型 & tokenizer —— 
print(f"🚀 正在加载模型：{MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16
)
model.eval()
model.to("cuda")

# —— 4. 计算剪枝层数（20%） —— 
total_layers    = model.config.num_hidden_layers
prune_layer_num = max(1, int(total_layers * 0.20))
print(f"⚙️ 模型共有 {total_layers} 层，计划剪除 {prune_layer_num} 层（约 20%）。\n")

# —— 5. 统计每层的余弦相似度 —— 
layer_cos_sum = np.zeros(total_layers, dtype=np.float64)
token_count   = np.zeros(total_layers, dtype=np.int64)

print("🔢 开始计算 Block Influence（BI）分数…")
for idx, text in enumerate(texts, start=1):
    if not text or not text.strip():
        continue

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=model.config.max_position_embeddings
    )
    input_ids      = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")

    outputs       = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple 长度 = total_layers+1

    for layer_idx in range(total_layers):
        h0 = hidden_states[layer_idx][0]
        h1 = hidden_states[layer_idx + 1][0]
        cos_vals = cosine_similarity(h0, h1, dim=-1)
        layer_cos_sum[layer_idx] += cos_vals.sum().item()
        token_count[layer_idx]   += cos_vals.numel()

    if idx % 50 == 0 or idx == len(texts):
        print(f"  已处理 {idx}/{len(texts)} 个样本")

print("\n✅ BI 统计完成，开始计算每层得分：")
bi_scores = []
for i in range(total_layers):
    avg_cos = (layer_cos_sum[i] / token_count[i]) if token_count[i] > 0 else 1.0
    bi = 1.0 - avg_cos
    bi_scores.append(bi)
    print(f"  层 {i:02d} BI = {bi:.6f}")

# —— 6. 排序 & 剪枝 —— 
to_prune = sorted(range(total_layers), key=lambda i: bi_scores[i])[:prune_layer_num]
print(f"\n✂️ 将剪除的层索引（Lowest BI）：{to_prune}")

for idx in sorted(to_prune, reverse=True):
    del model.model.layers[idx]
model.config.num_hidden_layers = total_layers - prune_layer_num

# —— 7. 保存剪枝后模型 & tokenizer —— 
print(f"\n💾 保存剪枝后模型到：{OUTPUT_DIR}\n")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("🎉 剪枝完成！")
