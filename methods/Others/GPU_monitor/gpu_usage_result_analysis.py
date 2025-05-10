#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime

# -------------------------
# Configure input and output paths
# -------------------------
# Input CSV file path
input_file = '/root/autodl-tmp/gpu_load/20250227051710_gpu_usage.csv'

# Use file prefix to create a new output folder (remove '_gpu_usage.csv' suffix)
base_name = os.path.basename(input_file).split('_gpu_usage.csv')[0]
# Create output folder to store summary text and plot image
output_dir = os.path.join(os.path.dirname(input_file), base_name + '_summary')
os.makedirs(output_dir, exist_ok=True)

# Output file paths for summary text and plot image
summary_file = os.path.join(output_dir, base_name + '_gpu_usage_summary.txt')
plot_file = os.path.join(output_dir, base_name + '_gpu_usage_plot.png')

# -------------------------
# Read and clean data
# -------------------------
# Read CSV file with skipinitialspace to remove extra spaces after commas
df = pd.read_csv(input_file, skipinitialspace=True)
# Strip any whitespace in column names (to avoid KeyError)
df.columns = df.columns.str.strip()

# Expected columns:
# ['timestamp', 'utilization.gpu [%]', 'utilization.memory [%]', 'memory.used [MiB]']
# Remove '%' from utilization columns and 'MiB' from memory usage column, then convert to float
df['utilization.gpu [%]'] = df['utilization.gpu [%]'].str.replace('%', '').str.strip().astype(float)
df['utilization.memory [%]'] = df['utilization.memory [%]'].str.replace('%', '').str.strip().astype(float)
df['memory.used [MiB]'] = df['memory.used [MiB]'].str.replace('MiB', '').str.strip().astype(float)

# Convert timestamp column to datetime object; expected format "YYYY/MM/DD HH:MM:SS.sss"
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M:%S.%f')

# -------------------------
# Filter data and compute statistics
# -------------------------
# Filter data with GPU utilization >= 10%
df_filtered = df[df['utilization.gpu [%]'] >= 10]

if df_filtered.empty:
    summary_text = "No data with GPU utilization >= 10% after filtering; unable to compute statistics."
else:
    # GPU utilization statistics
    gpu_avg   = df_filtered['utilization.gpu [%]'].mean()
    gpu_max   = df_filtered['utilization.gpu [%]'].max()
    gpu_top95 = df_filtered['utilization.gpu [%]'].quantile(0.95)  # 95th percentile
    gpu_top90 = df_filtered['utilization.gpu [%]'].quantile(0.90)  # 90th percentile

    # Memory utilization statistics (in %)
    mem_util_avg   = df_filtered['utilization.memory [%]'].mean()
    mem_util_max   = df_filtered['utilization.memory [%]'].max()
    mem_util_top95 = df_filtered['utilization.memory [%]'].quantile(0.95)
    mem_util_top90 = df_filtered['utilization.memory [%]'].quantile(0.90)

    # Memory usage statistics (in MiB)
    mem_usage_avg   = df_filtered['memory.used [MiB]'].mean()
    mem_usage_max   = df_filtered['memory.used [MiB]'].max()
    mem_usage_top95 = df_filtered['memory.used [MiB]'].quantile(0.95)
    mem_usage_top90 = df_filtered['memory.used [MiB]'].quantile(0.90)

    # Compute sampling time range from filtered data
    time_start = df_filtered['timestamp'].min()
    time_end   = df_filtered['timestamp'].max()
    duration   = time_end - time_start

    # Construct summary text in English
    summary_text = f"""GPU Usage Statistics (data with GPU utilization >= 10%):
-------------------------------------------------------
Sampling Time Range: {time_start} to {time_end}
Duration: {duration}

GPU Utilization (%):
    Mean: {gpu_avg:.2f} %
    Max: {gpu_max:.2f} %
    95th Percentile: {gpu_top95:.2f} %
    90th Percentile: {gpu_top90:.2f} %

Memory Utilization (%):
    Mean: {mem_util_avg:.2f} %
    Max: {mem_util_max:.2f} %
    95th Percentile: {mem_util_top95:.2f} %
    90th Percentile: {mem_util_top90:.2f} %

Memory Usage (MiB):
    Mean: {mem_usage_avg:.2f} MiB
    Max: {mem_usage_max:.2f} MiB
    95th Percentile: {mem_usage_top95:.2f} MiB
    90th Percentile: {mem_usage_top90:.2f} MiB
-------------------------------------------------------
Total data points: {len(df)}
Data points after filtering: {len(df_filtered)}
"""

# Write summary text to file
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write(summary_text)
print(f"Summary saved to {summary_file}")

# -------------------------
# Plotting (Optimized for research publication)
# -------------------------

# 设置 Matplotlib 风格
# 选择一个可用的 Matplotlib 样式
available_styles = plt.style.available
if 'seaborn' in available_styles:
    plt.style.use('seaborn')
elif 'ggplot' in available_styles:
    plt.style.use('ggplot')
else:
    plt.style.use('default')  # 兜底方案，保证不会出错
  # 科研风格
#plt.rcParams['font.family'] = 'Times New Roman'  # 论文风格字体
plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签大小
plt.rcParams['xtick.labelsize'] = 12  # x轴刻度大小
plt.rcParams['ytick.labelsize'] = 12  # y轴刻度大小
plt.rcParams['legend.fontsize'] = 12  # 图例字体大小

# 创建画布和轴
fig, ax1 = plt.subplots(figsize=(14, 7))

# 绘制 GPU 利用率
ax1.plot(df['timestamp'], df['utilization.gpu [%]'], 
         label='GPU Utilization (%)', color='navy', 
         marker='o', linestyle='-', markersize=3, linewidth=1.5, alpha=0.8)

# 绘制内存利用率
ax1.plot(df['timestamp'], df['utilization.memory [%]'], 
         label='Memory Utilization (%)', color='darkorange', 
         marker='s', linestyle='-', markersize=3, linewidth=1.5, alpha=0.8)

# 设置 X 轴格式
ax1.set_xlabel('Timestamp')
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

# 设置 Y 轴（左）
ax1.set_ylabel('Utilization (%)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

# 添加右侧 Y 轴
ax2 = ax1.twinx()

# 绘制内存使用量
ax2.plot(df['timestamp'], df['memory.used [MiB]'], 
         label='Memory Usage (MiB)', color='darkgreen', 
         marker='^', linestyle='--', markersize=3, linewidth=1.5, alpha=0.8)

ax2.set_ylabel('Memory Usage (MiB)', color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

# 图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True)

# 添加网格
ax1.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

# 设置标题
plt.title('GPU Utilization, Memory Utilization, and Memory Usage Over Time', fontsize=16)

# 调整布局，避免标签被裁剪
fig.tight_layout()

# 保存高分辨率图像
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Optimized plot saved to {plot_file}")
