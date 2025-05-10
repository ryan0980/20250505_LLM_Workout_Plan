from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

base_model_path = "/root/autodl-tmp/model/q3b_q_p"
adapter_path    = "/root/autodl-tmp/llamafactory/LLaMA-Factory/saves/.../train_2025-04-20-01-04-23"

# 1. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# 2. 直接在 from_pretrained 里传量化参数
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    # 下面这三个就是旧版量化接口
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 3. 挂载 LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    device_map="auto"
)

# 4. 推理
gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7
)

print(gen("请解释量化+剪枝+LoRA 微调结合的优势。")[0]["generated_text"])
