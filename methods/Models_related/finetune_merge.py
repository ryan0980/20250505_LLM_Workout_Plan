from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch
import os

# 1. 加载基础模型（完整权重在 q3bo 目录中）
base_model_path = "/root/autodl-tmp/model/q3bp_q"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,  # 根据你的情况设置
)

# 2. 加载适配器（PEFT 模型），此处使用导出目录中的适配器权重
adapter_path = "/root/autodl-tmp/llamafactory/LLaMA-Factory/saves/Qwen2.5-3B-Instruct/lora/train_2025-04-20-01-04-25"
peft_model = PeftModel.from_pretrained(base_model, adapter_path)

# 3. 合并适配器权重到基础模型
peft_model.merge_and_unload()

# 4. 创建目标目录（如果不存在），再保存合并后的完整模型
merged_model_path = "/root/autodl-tmp/model/q3bp_q_ft"
os.makedirs(merged_model_path, exist_ok=True)  # 如果目录不存在则创建
base_model.save_pretrained(merged_model_path)

print(f"合并后的完整模型已保存到 {merged_model_path}")
