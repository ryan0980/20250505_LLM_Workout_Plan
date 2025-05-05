#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
该脚本用于下载 Hugging Face 上的 Qwen2.5-3B-Instruct 模型，
并将其保存到本地目录 DQFP/model/qwen_7b_I。
"""

import os
from huggingface_hub import snapshot_download

def main():
    # 模型仓库 ID 和本地保存目录
    repo_id = "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"
    local_dir = "DQFP/model/qwen_3b_I_GPTQ_Official"
    
    # 如果目标目录不存在，则创建目录
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        print(f"创建目录: {local_dir}")
    
    # 下载模型
    print(f"开始下载模型 {repo_id} 到 {local_dir} ...")
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print("模型下载完成。")

if __name__ == "__main__":
    main()
