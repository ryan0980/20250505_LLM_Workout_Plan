---
title: 
aliases: 
tags: 
author: tusrau
date created: 20250318 06:52
date updated: 20250510 03:57
---

# Torch-pruning

```python

model = get_llm(args.model, max_seq_len=args.max_seq_len)
model.config.use_cache = False  # 禁用缓存机制，防止 tuple 返回错误
model.eval()




2. 


for name, m in model.named_modules():
    if name.endswith("self_attn"):
        if seperate_qkv:
            m.hidden_size = m.q_proj.out_features
        else:
            m.hidden_size = m.qkv_proj.out_features // 3        
        m.num_heads = m.hidden_size // m.head_dim
        model.config.num_attention_heads = m.num_heads
        # 如果是 Qwen2 模型，强制使键值头数与查询头数一致
        if model.config.model_type.startswith("qwen"):
            m.num_key_value_heads = m.num_heads
            m.num_key_value_groups = 1
        else:
            # 如果模块存在 num_key_value_heads，则正常计算
            if hasattr(m, "num_key_value_heads"):
                if not _is_gqa:
                    m.num_key_value_heads = m.num_heads
                m.num_key_value_groups = m.num_heads // m.num_key_value_heads
            else:
                m.num_key_value_groups = 1
    elif name.endswith("mlp"):
        if hasattr(m, "gate_proj"):
            m.hidden_size = m.gate_proj.in_features
            model.config.intermediate_size = m.gate_proj.out_features
        elif hasattr(m, "gate_up_proj"):
            m.hidden_size = m.gate_up_proj.in_features
            model.config.intermediate_size = m.gate_up_proj.out_features // 2
        else:
            raise ValueError("Unknown mlp layer")

```

 [[20250318]]
