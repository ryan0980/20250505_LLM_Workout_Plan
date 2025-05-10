#!/bin/bash

# 指定 GPU 监控数据保存目录
GPU_OUTPUT_DIR="/root/autodl-tmp/gpu_load"
mkdir -p "${GPU_OUTPUT_DIR}"

# 生成带时间戳的 GPU 监控数据文件名，例如 20250219045055_gpu_usage.csv
GPU_OUTPUT_FILE="${GPU_OUTPUT_DIR}/$(date +%Y%m%d%H%M%S)_gpu_usage.csv"

# 指定 benchmark 结果保存目录（通过 run.py 的 -w 参数来指定工作路径）
RESULT_DIR="/root/autodl-tmp/benchmark_result"
mkdir -p "${RESULT_DIR}"

# 后台运行 nvidia-smi，每秒记录一次 GPU 信息，并将输出保存到文件中
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv -l 1 > "${GPU_OUTPUT_FILE}" &
NVIDIA_SMI_PID=$!

# 执行 benchmark 脚本，注意这里使用 -w 参数指定输出目录
python /root/autodl-fs/opencompass/run.py \
  --datasets demo_gsm8k_chat_gen \
  --hf-type chat \
  --hf-path /root/autodl-tmp/model/q1bo \
  --debug \
  -w "${RESULT_DIR}"

# benchmark 执行结束后，终止 nvidia-smi 的监控进程
kill $NVIDIA_SMI_PID

echo "GPU 监控数据已保存至 ${GPU_OUTPUT_FILE}"
echo "Benchmark 结果已输出至 ${RESULT_DIR}"
