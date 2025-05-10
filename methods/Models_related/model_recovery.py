import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW

model_path = "/root/autodl-tmp/model/q3bp"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

# 准备一些微调数据
texts = [
    "Hello, this is a test sentence.",
    "Another example for fine-tuning.",
    # 其他文本...
]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 3

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(**inputs)
    loss = outputs.loss  # 假设模型返回 loss（或你自定义损失函数）
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, loss: {loss.item()}")

# 微调后保存模型
model.save_pretrained("/root/autodl-tmp/model/q3bp_finetuned")
tokenizer.save_pretrained("/root/autodl-tmp/model/q3bp_finetuned")
