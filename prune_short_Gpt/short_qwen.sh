#!/bin/bash
# short_qwen_xin.sh - 针对本地 Qwen2.5-3B 模型的剪枝脚本（已修改）

# 模型路径：使用本地 Qwen2.5-3B 基础模型
model_path="/root/autodl-tmp/model/q3bo"  # 修改：本地非 instruct 模型路径

echo "开始对模型进行剪枝：$model_path"

# 直接调用 Python 脚本进行剪枝，脚本内部会自动计算要剪掉的层数（20%）
python short_qwen_xin.py "$model_path"

echo "剪枝完成，已将剪枝后模型保存。"
