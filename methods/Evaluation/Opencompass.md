---
aliases: 
tags: 
date created: 20250224 04:28
date updated: 20250423 04:29
---

# Opencompass

```bash
conda activate opencompass

pip install pypinyin ltp cn2an

python run.py --models hf_llama2_7b --datasets mmmlu_lite_gen

cd /root/autodl-tmp/opencompass/opencompass
python /root/autodl-tmp/opencompass/opencompass/run.py --datasets lawbench3 --hf-type base --hf-path /root/autodl-tmp/model/q3bft_q --debug

python /root/autodl-tmp/opencompass/opencompass/run.py --datasets mmlu_gen --hf-type base --hf-path /root/autodl-tmp/model/q3bp --debug 

triton-3.3.0(awq)

python tools/list_configs.py lawbench

python run.py \
    --datasets demo_gsm8k_chat_gen \
    --hf-type base \
    --hf-path /root/autodl-tmp/model/q3bo \
    --debug
    
python run.py --datasets siqa_gen winograd_ppl --hf-type base --hf-path /path/to/model

```
