#!/usr/bin/env python3
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# —— 路径配置 —— 
quant_model_path = "/root/autodl-tmp/model/q3bft_q"

# —— 加载量化模型 & 分词器 —— 
print(f"🔄 从 {quant_model_path} 加载量化模型…")
model = AutoAWQForCausalLM.from_pretrained(
    quant_model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(quant_model_path, trust_remote_code=True)

model.eval()

# —— 手动确定设备 —— 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 如果是单卡，也可以直接 model.to(device)，不过 device_map="auto" 时已分片到 GPU
# 这里依然确保 tokenizer 输出能正确搬到第一个 GPU 或 CPU
print(f"ℹ️  使用设备: {device}")

# —— 构造测试 Prompt —— 
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# —— 生成文本 —— 
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )

# —— 输出结果 —— 
print("\n📝 Prompt:")
print(prompt)
print("\n📝 Generated:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
