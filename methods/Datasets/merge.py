import os
import json

def merge_json_files(input_directory, output_file):
    merged_data = []
    # 遍历目录下所有文件
    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            file_path = os.path.join(input_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    # 如果文件内容是数组，则扩展列表，否则直接添加对象
                    if isinstance(data, list):
                        merged_data.extend(data)
                    else:
                        merged_data.append(data)
                except Exception as e:
                    print(f"读取 {filename} 时出错：{e}")
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # 写入合并后的数据到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    print(f"已将 {len(merged_data)} 条记录合并到 {output_file}")

if __name__ == '__main__':
    # 设定输入文件夹和输出文件名称
    input_directory = '/root/autodl-tmp/datasets/lawbench3/valid_for_prune/train'  # 输入目录
    output_file = '/root/autodl-tmp/datasets/lawbench3/valid_for_prune/train_merged/merged.json'  # 输出文件路径
    merge_json_files(input_directory, output_file)
