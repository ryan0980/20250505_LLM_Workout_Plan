import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer
from awq.utils.utils import get_best_device

# 获取最佳设备（如 GPU 或 CPU）
device = get_best_device()

# 定义量化模型的存储路径
quant_path = "/home/sagemaker-user/DQFP/model/qwen_3b_I_awq_self"

# 加载量化后的模型并移动到指定设备上
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
model = model.to(device)

# 加载 tokenizer（注意 trust_remote_code 参数）
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)

# 配置 TextStreamer 用于流式输出
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 定义对话模板
prompt_template = """\
<|system|>
</s>
<|user|>
{prompt}</s>
<|assistant|>"""

# 定义输入提示语
prompt = (
    "You're standing on the surface of the Earth. "
    "You walk one mile south, one mile west and one mile north. "
    "You end up exactly where you started. Where are you?"
)

# 根据模板生成最终的输入文本
input_text = prompt_template.format(prompt=prompt)

# 将输入文本转换为模型所需的 tokens，并移动到设备上
inputs = tokenizer(input_text, return_tensors='pt')
inputs = {key: value.to(device) for key, value in inputs.items()}

# 调用生成函数，采用采样策略生成输出，同时使用 streamer 实时输出
output_ids = model.generate(
    **inputs,
    max_length=512,        # 最大输出长度，根据实际情况调整
    do_sample=True,        # 使用采样生成
    top_k=50,              # top-k 采样
    top_p=0.95,            # nucleus 采样
    temperature=0.7,       # 温度参数
    streamer=streamer      # 使用流式输出
)

# 如果不使用 streamer，也可以直接解码生成结果：
# output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
# print("生成结果：", output_text)
