import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 原模型路径和输出的量化模型路径
model_path = "/root/autodl-tmp/model/q3b_p_ft"
quant_path = "/root/autodl-tmp/model/q3b_p_ft_q"

# 量化配置
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# 如果输出目录不存在就创建
os.makedirs(quant_path, exist_ok=True)

# 加载模型和分词器
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 执行量化
model.quantize(tokenizer, quant_config=quant_config)

# 保存量化后的模型和分词器
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
