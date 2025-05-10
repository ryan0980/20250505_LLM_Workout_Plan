#!/bin/bash


# 指定日志和 GPU 监控数据保存目录，并创建目录
GPU_OUTPUT_DIR="/root/autodl-tmp/gpu_load"
mkdir -p "${GPU_OUTPUT_DIR}"

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d%H%M%S)

# 生成 GPU 监控数据文件和任务日志文件的文件名
GPU_OUTPUT_FILE="${GPU_OUTPUT_DIR}/${TIMESTAMP}_gpu_usage.csv"
LOG_FILE="${GPU_OUTPUT_DIR}/${TIMESTAMP}_benchmark.log"

# 将所有标准输出和错误输出重定向到日志文件，不打印到终端
exec > "${LOG_FILE}" 2>&1

# 记录并显示所有 GPU 的详细信息
echo "===================="
echo "任务开始时间：$(date)"
echo "日志文件：${LOG_FILE}"
echo "GPU监控数据文件：${GPU_OUTPUT_FILE}"
echo "当前所有 GPU 信息："
nvidia-smi -L
echo "--------------------"
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "可用 GPU 数量：${NUM_GPUS}"
echo "===================="

# 定义清理函数，确保退出时终止 nvidia-smi 进程
cleanup() {
    if [ -n "$NVIDIA_SMI_PID" ] && ps -p $NVIDIA_SMI_PID > /dev/null 2>&1; then
        kill $NVIDIA_SMI_PID
        echo "Stopped GPU monitoring process (PID: $NVIDIA_SMI_PID)"
    fi
}
trap cleanup EXIT

# 后台运行 nvidia-smi，每秒记录一次 GPU 信息，并将输出保存到文件中
echo "Starting GPU monitoring, output file: ${GPU_OUTPUT_FILE}"
nvidia-smi --query-gpu=index,timestamp,utilization.gpu,utilization.memory,memory.used --format=csv -l 1 > "${GPU_OUTPUT_FILE}" &
NVIDIA_SMI_PID=$!

# 切换到 OpenCompass 工作目录（确保相对路径正确）
cd /root/autodl-tmp/opencompass/opencompass

# 执行 benchmark 脚本（如有需要，可添加 -w 参数指定输出目录）
echo "Starting benchmark..."
python run.py --datasets lawbench2_test --hf-type base --hf-path /root/autodl-tmp/model/q3bp --debug
echo "Benchmark finished."

echo "GPU monitoring data saved to ${GPU_OUTPUT_FILE}"
echo "Task finished at: $(date)"

# 最后关机
echo "Shutting down system now..."
#shutdown -h now
