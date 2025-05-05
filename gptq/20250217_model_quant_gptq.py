import logging
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

pretrained_model_dir = "/home/sagemaker-user/DQFP/model/qwen_3b_I"
quantized_model_dir = "/home/sagemaker-user/DQFP/model/qwen_3b_I_GPTQ_self"

# 确保 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)

# 量化参数
quantize_config = BaseQuantizeConfig(
    bits=4,  # 4-bit 量化
    group_size=64,  # 设置 64 以提高数值稳定性
    desc_act=True,  # 启用激活值降维，提高数值稳定性
)

# 加载模型并移动到 GPU
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config, device_map="auto")
model.to("cuda:0")

# 确保输入数据在 GPU
examples = [
    {k: v.to("cuda:0") for k, v in tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm.",
        return_tensors="pt"
    ).items()}
]

# 量化模型
model.quantize(examples)

# 保存量化模型
model.save_quantized(quantized_model_dir, use_safetensors=True)

# 加载量化模型到 GPU
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")

# 推理
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
print(pipeline("auto-gptq is")[0]["generated_text"])

