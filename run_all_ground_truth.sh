#!/bin/bash
# 自动化生成 Ground Truth（仅使用 GPU 0,1,2,3）
# 依次测试 TP=1, 2, 4

# 切换到脚本所在目录（确保路径正确）
cd /data/home/lihaozhe/llm-simulator || exit 1

# 定义要测试的 TP 配置及对应的 GPU 列表
# TP=1 -> 使用 GPU 4
# TP=2 -> 使用 GPU 4,5
# TP=4 -> 使用 GPU 4,5,6,7
declare -A TP_GPUS
TP_GPUS[1]="0"
TP_GPUS[2]="0,1"
TP_GPUS[4]="0,1,2,3"

# 清除旧的 Ground Truth 文件（防止重复追加）
# 如果不想清空，可以注释掉下面这行
# rm -f /data/home/lihaozhe/llm-simulator/data/ground_truth_pytorch.csv
# echo "🗑️ 已清除旧的 ground_truth_pytorch.csv"

# 循环执行
for tp in 2 4; do
    echo "=========================================="
    echo "🚀 开始测试 TP=${tp}，使用 GPU: ${TP_GPUS[$tp]}"
    echo "=========================================="
    
    # 设置环境变量并运行 torchrun
    CUDA_VISIBLE_DEVICES=${TP_GPUS[$tp]} \
    torchrun --nproc_per_node=${tp} \
        /data/home/lihaozhe/llm-simulator/scripts/generate_ground_truth.py
    
    # 检查上一个命令是否成功
    if [ $? -eq 0 ]; then
        echo "✅ TP=${tp} 测试完成"
    else
        echo "❌ TP=${tp} 测试失败，请检查日志"
        # 继续执行后续 TP，不中断
    fi
done

echo "=========================================="
echo "🎯 所有测试完成！"
echo "📁 结果文件: /data/home/lihaozhe/llm-simulator/data/ground_truth_pytorch.csv"
echo "=========================================="

# 显示文件行数（含表头）
wc -l /data/home/lihaozhe/llm-simulator/data/ground_truth_pytorch.csv