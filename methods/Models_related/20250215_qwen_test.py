#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
该脚本用于加载本地 Qwen2.5-7B-Instruct 模型，并调用模型进行文本生成测试。
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

print("CUDA 可用:", torch.cuda.is_available())  # True 表示 PyTorch 识别到了 GPU
print("CUDA 版本:", torch.version.cuda)  # 12.1
print("GPU 名称:", torch.cuda.get_device_name(0))  # NVIDIA A10G
print("GPU 数量:", torch.cuda.device_count())  # 1



def main():
    # 指定模型的本地路径
    model_dir = "/home/sagemaker-user/DQFP/model/qwen_3b_I_GPTQ_self"
    
    # 加载 tokenizer 和模型
    print("加载 Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    print("加载模型 ...")
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    
    # 创建文本生成 pipeline
    text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    # 测试 prompt
    prompt = "下面是一段新闻报道，请用一句话给出这段报道的摘要。据人民法院诉讼资产网显示，乐视控股（北京）有限公司持有的乐视影业（北京）有限公司21.8122%的股权拍卖，9点02分出现第一笔出价，随后没有其他报价出现，截至拍卖结束以起拍价53160.616422万元成交。此外与乐视控股相关另外两个标的拍卖显示：乐视控股（北京）有限公司持有的新乐视智家电子科技（天津）有限公司26183537元出资额的股权拍卖，9点01分出现10983.693万元的第一笔报价，随后没有其他报价出现，截至拍卖结束以起拍价10983.693万元成交。新乐视智家电子科技（天津）有限公司中31245271.2元出资额的股权拍卖，9点01分出现13107.031万元的第一笔报价，随后没有其他报价出现，截至拍卖结束以起拍价13107.031万元成交。"
    print("开始生成文本 ...")
    outputs = text_gen(prompt, max_length=10000, do_sample=True, temperature=0.7)
    
    print("模型生成结果:")
    print(outputs[0]["generated_text"])

if __name__ == "__main__":
    main()


