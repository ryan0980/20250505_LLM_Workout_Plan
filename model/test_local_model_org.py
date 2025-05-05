from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 模型保存的路径（AWQ 量化模型）
model_path = "/root/autodl-tmp/model/q3b_ft_adapter"

# 尝试使用 device_map 自动映射到 GPU（如果可用）
print("加载模型，请稍候...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    ignore_mismatched_sizes=True,
    device_map="auto"  # 尝试自动加载到 GPU 上
)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# 如果模型仍在 CPU 上，可主动转移到 GPU（如果有 CUDA）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print("模型加载完毕，当前设备：", device)

# 测试输入文本
input_text = (
    "请你运用法律知识从A,B,C,D中选出一个正确的答案，并写在[正确答案]和<eoa>之间。例如[正确答案]A<eoa>。"
    "请你严格按照这个格式回答。\n如果同一案件有几个鉴定人，下列说法正确的是?"
    "A:几个鉴定人之间不能相互讨论;"
    "B:几个鉴定人必须共同作出一份鉴定结论;"
    "C:几个鉴定人意见不一致时，要遵循少数服从多数的原则，作出一份鉴定结论;"
    "D:几个鉴定人意见不一致时，每个鉴定人有权单独作出自己的鉴定结论。"
)

# 编码输入，并确保张量在同一设备上
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# 设置生成参数
generation_args = {
    "max_length": 500,
    "do_sample": True,     # 启用采样
    "temperature": 1.0,    # 温度参数，值越高生成分布越平滑
    "top_k": 50,           # top-k 策略
    "top_p": 0.9,          # top-p 策略
}

# 调试：首先计算 logits 并打印其统计信息
with torch.no_grad():
    outputs_debug = model(**inputs, return_dict=True)
    logits = outputs_debug.logits[:, -1, :]
    # 将 logits 转换为 float32 以确保数值稳定性（量化模型有时存储低精度值）
    logits = logits.to(torch.float32)
    print("logits - min: {:.6f}, max: {:.6f}, mean: {:.6f}".format(
        torch.min(logits).item(),
        torch.max(logits).item(),
        torch.mean(logits).item()
    ))
    probs = torch.softmax(logits, dim=-1)
    print("概率分布 - min: {:.6f}, max: {:.6f}".format(
        torch.min(probs).item(), torch.max(probs).item()
    ))
    print("是否包含 nan:", torch.isnan(probs).any().item())

# 执行生成过程
with torch.no_grad():
    try:
        outputs = model.generate(**inputs, **generation_args)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n输入：", input_text)
        print("\n输出：", generated_text)
    except RuntimeError as e:
        print("生成时发生错误：", str(e))
