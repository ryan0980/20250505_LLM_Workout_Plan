---
aliases: 
tags: 
date created: 20250216 01:58
date updated: 20250419 05:11
---

# Llamafactory

【大模型微调！手把手带你用 LLaMA-Factory 工具微调 Qwen 大模型！有手就行，零代码微调任意大语言模型】 <https://www.bilibili.com/video/BV1Q8rYYdErf/?p=2&share_source=copy_web&vd_source=0966bfa7a4bab72b260e36ae95146477>

```
conda activate llamafactory
cd /root/autodl-tmp/llamafactory/LLaMA-Factory
llamafactory-cli webui

```

注意 data 文件夹下需要注册数据集

预热步数设置为 4
LoRA 缩放系数 256

先用 webui 生成训练代码，类似：

```shell
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /root/autodl-tmp/model/q3bo \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset lawbench_train \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/Qwen2.5-3B/lora/train_2025-04-16-11-42-22 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all

```

然后用类似

```python
from transformers import AutoModelForCausalLM

from peft import PeftModel

import torch

  

# 1. 加载基础模型（完整权重在 q3bo 目录中）

base_model_path = "/root/autodl-tmp/model/q3bo"

base_model = AutoModelForCausalLM.from_pretrained(

    base_model_path,

    torch_dtype=torch.bfloat16,  # 根据你的情况设置

)

  

# 2. 加载适配器（PEFT 模型），此处使用导出目录中的适配器权重

adapter_path = "/root/autodl-tmp/llamafactory/LLaMA-Factory/saves/Qwen2.5-3B/lora/train_2025-04-18-05-29-27"

peft_model = PeftModel.from_pretrained(base_model, adapter_path)

  

# 3. 合并适配器权重到基础模型

peft_model.merge_and_unload()

  

# 4. 将合并后的完整模型保存到新的目录中

merged_model_path = "/root/autodl-tmp/model/q3b_ft_adapter"

base_model.save_pretrained(merged_model_path)

  

print(f"合并后的完整模型已保存到 {merged_model_path}")
```

的代码 merge，最后复制 Tokenizer 相关文件

```bash
# 1. 创建目标目录（如果不存在）
mkdir -p /root/autodl-tmp/model/q3b_ft_adapter

# 2. 复制常见的 tokenizer 文件
cp /root/autodl-tmp/model/q3bo/{tokenizer.json,tokenizer_config.json,vocab.json,special_tokens_map.json,merges.txt} \
   /root/autodl-tmp/model/q3b_ft_adapter/

# 3. 如果存在 added_tokens.json，单独再复制一遍（防止上面报错）
cp /root/autodl-tmp/model/q3bo/added_tokens.json \
   /root/autodl-tmp/model/q3b_ft_adapter/ 2>/dev/null || true

```

20250216_lawbench2/8 微调 loss：
![image.png](https://raw.githubusercontent.com/ryan0980/expert-potato/main/img/202502162125481.png)

llama 7b
![image.png](https://raw.githubusercontent.com/ryan0980/expert-potato/main/img/202502211916230.png)

qwen 3b
![image.png](https://raw.githubusercontent.com/ryan0980/expert-potato/main/img/202504160137471.png)
q3bo
![image.png](https://raw.githubusercontent.com/ryan0980/expert-potato/main/img/202504171846206.png)

q3b_q_p

/root/autodl-tmp/llamafactory/LLaMA-Factory/saves/Qwen2.5-3B-Instruct/lora/train_2025-04-20-01-04-23
`NotImplementedError` 出现在 PEFT 的保护机制里：当基础模型使用 GPTQ/AWQ 量化时，LoRA 的 `merge_and_unload()` 功能会被禁用，因而抛出错误

![image.png](https://raw.githubusercontent.com/ryan0980/expert-potato/main/img/202504191515421.png)

## q3bp_q_ft

```
/root/autodl-tmp/llamafactory/LLaMA-Factory/saves/Qwen2.5-3B-Instruct/lora/train_2025-04-20-01-04-25
```

![image.png](https://raw.githubusercontent.com/ryan0980/expert-potato/main/img/202504191710487.png)
