---
aliases: 
tags: 
date created: 20250218 11:41
date updated: 20250225 08:18
---

# Pruning

```bash

(prune_llm) root@autodl-container-bbd74aba75-f6db4254:~/autodl-tmp/wanda/wanda# python main.py     --model /root/autodl-tmp/model/q3bo     --prune_method wanda     --sparsity_ratio 0.5     --sparsity_type unstructured     --save /root/autodl-tmp/model/q3bp

python main.py \
    --model /root/autodl-tmp/llama/7b_o_HF \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save_model /root/autodl-tmp/llama/7b_p \
    --save /root/autodl-tmp/llama/wanda_output



```

[WYXSCIR/CFSP --- wyxscir/CFSP](https://github.com/wyxscir/CFSP)
[IST-DASLab/sparsegpt: Code for the ICML 2023 paper "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot".](https://github.com/IST-DASLab/sparsegpt?tab=readme-ov-file)
[马/LLM - 普鲁纳：[神经2023]LLM - 普鲁纳：关于大语言模型的结构修剪。支持Llama-3/3.1，Llama-2，Llama，Bloom，Vicuna，Baichuan，Tinyllama等 --- horseee/LLM-Pruner: [NeurIPS 2023] LLM-Pruner: On the Structural Pruning of Large Language Models. Support Llama-3/3.1, Llama-2, LLaMA, BLOOM, Vicuna, Baichuan, TinyLlama, etc.](https://github.com/horseee/LLM-Pruner?tab=readme-ov-file)
[locuslab/wanda: A simple and effective LLM pruning approach.](https://github.com/locuslab/wanda/tree/main)
[PPRP/很棒 - LLM-prune：很棒的清单LLM修剪。 --- pprp/Awesome-LLM-Prune: Awesome list for LLM pruning.](https://github.com/pprp/Awesome-LLM-Prune)
[[2403.17887] The Unreasonable Ineffectiveness of the Deeper Layers](https://arxiv.org/abs/2403.17887)
[arcee-ai/PruneMe: Automated Identification of Redundant Layer Blocks for Pruning in Large Language Models](https://github.com/arcee-ai/PruneMe)
[论文阅读：The Unreasonable Ineffectiveness of the Deeper Layers 层剪枝与模型嫁接的“双生花” | clvsit 个人博客](https://clvsit.github.io/%E5%B1%82%E5%89%AA%E6%9E%9D%E4%B8%8E%E6%A8%A1%E5%9E%8B%E5%AB%81%E6%8E%A5%E7%9A%84%E2%80%9C%E5%8F%8C%E7%94%9F%E8%8A%B1%E2%80%9D/)

查找了很多关于 qwen 2.5 进行剪枝的操作，发现大多数都不支持，看了 [QWen2.5 简单分析 - 知乎](https://zhuanlan.zhihu.com/p/721084591) 后发现原因可能是 QWen2.5 本身已经通过模型剪枝和稀疏化技术降低了 20% 的参数量
