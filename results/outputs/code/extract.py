import os

root_dir = '/root/autodl-tmp/model/outputs/default'
output_dir = '/root/autodl-tmp/model/outputs/code'
output_file = os.path.join(output_dir, 'lawbench_avg_summary.txt')

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

results = []

# 需要的两个数据集名
required_datasets = {'lukaemon_mmlu_international_law', 'lukaemon_mmlu_professional_law'}

for folder in os.listdir(root_dir):
    summary_folder = os.path.join(root_dir, folder, 'summary')
    if os.path.isdir(summary_folder):
        csv_files = [f for f in os.listdir(summary_folder) if f.endswith('.csv')]
        if csv_files:
            csv_path = os.path.join(summary_folder, csv_files[0])
            try:
                with open(csv_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 3:
                        header = lines[0].strip().split(',')
                        rows = [line.strip().split(',') for line in lines[1:]]
                        datasets_in_file = {row[0] for row in rows}
                        
                        # 必须正好是这两个数据集，不能多也不能少
                        if datasets_in_file == required_datasets:
                            scores = [float(row[-1]) for row in rows]
                            avg_score = sum(scores) / len(scores)
                            model_name = header[-1]
                            results.append((model_name, round(avg_score, 2)))
            except Exception as e:
                print(f"读取 {csv_path} 失败，原因：{e}")

# 保存到文件
with open(output_file, 'w') as f:
    f.write('model,avg_score\n')
    for model, avg_score in results:
        f.write(f'{model},{avg_score}\n')

print(f"✅ 提取完成，结果保存在: {output_file}")
