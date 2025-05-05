import os
import glob
import json
import random

# 输入目录，包含分散的 JSON 文件
input_dir = "/home/sagemaker-user/DQFP/dataset/lawbench_self_split/eval"

# 输出合并后的文件路径
output_file = "/home/sagemaker-user/DQFP/dataset/lawbench_self_split/lawbench_eval.json"

# 获取目录下所有 .json 文件（不包含子目录中的）
json_files = glob.glob(os.path.join(input_dir, "*.json"))

merged_data = []

# 遍历所有 JSON 文件，加载数据并合并
for file_path in json_files:
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                print(f"文件 {file_path} 中的数据格式不是列表，已跳过。")
        except Exception as e:
            print(f"读取 {file_path} 失败: {e}")

# 对合并后的数据随机打乱
random.shuffle(merged_data)

# 保存打散后的合并数据到输出文件
with open(output_file, "w", encoding="utf-8") as f_out:
    json.dump(merged_data, f_out, ensure_ascii=False, indent=2)

print(f"合并并打散完成，共 {len(merged_data)} 条数据，保存到 {output_file}")
