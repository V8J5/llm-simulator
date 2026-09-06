#!/bin/bash

# 脚本：run_profiling.sh
# 功能：一键启动 L20 上的 vLLM 全维度性能剖析

set -e # 遇到错误则退出

echo "========================================"
echo "🚀 LLM 仿真器 - L20 性能剖析脚本"
echo "========================================"

# 1. 检查依赖
echo "🔍 正在检查依赖..."
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到 python 命令"
    exit 1
fi

# 2. 激活 Conda 环境 (请根据你的实际环境名修改)
CONDA_ENV="llm-simulator"
echo "⚡ 正在激活 Conda 环境: $CONDA_ENV"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# 3. 检查 vLLM 是否安装
python -c "import vllm" 2>/dev/null || { echo "❌ 错误: 未找到 vllm 库，请检查环境"; exit 1; }

# 4. 创建数据目录
mkdir -p ../data

# 5. 运行核心压测脚本
echo "🏃 开始执行压测..."
python /data/home/lihaozhe/llm-simulator/scripts/profile_vllm.py

echo "========================================"
echo "✅ 所有压测完成！"
echo "📊 结果文件: ../data/profiling_results.csv"
echo "📝 服务日志: vllm_server.log"
echo "========================================"