import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ================== 配置区 ==================
DATA_FILE = "/data/home/lihaozhe/llm-simulator/data/profiling_results.csv"
OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 从 analyze_data.py 提取的 TP=4 拟合参数
# Prefill: TTFT = A*(B*P) + B*B + C*P + D
PREFILL_PARAMS = {
    'A': 0.018700,
    'B': 15.343160,
    'C': 0.067241,
    'D': 93.995527
}

# Decode: TPOT = A*Batch + B*ln(Total_Len) + C
DECODE_PARAMS = {
    'A': 0.482230,
    'B': 0.768009,
    'C': 20.536811
}

# 需要验证的 TP 配置 (排除掉用于训练的 TP=4)
VALIDATE_TP_SIZES = [1, 2, 8] 
# =============================================

def calculate_metrics(df):
    """计算预测值并返回合并后的 DataFrame"""
    # Prefill 预测
    prefill_mask = df['output_tokens'] == 1
    df.loc[prefill_mask, 'pred_ttft'] = (
        PREFILL_PARAMS['A'] * df.loc[prefill_mask, 'batch_size'] * df.loc[prefill_mask, 'prompt_tokens'] +
        PREFILL_PARAMS['B'] * df.loc[prefill_mask, 'batch_size'] +
        PREFILL_PARAMS['C'] * df.loc[prefill_mask, 'prompt_tokens'] +
        PREFILL_PARAMS['D']
    )
    
    # Decode 预测
    decode_mask = df['output_tokens'] > 1
    total_len = df.loc[decode_mask, 'prompt_tokens'] + df.loc[decode_mask, 'output_tokens']
    df.loc[decode_mask, 'pred_tpot'] = (
        DECODE_PARAMS['A'] * df.loc[decode_mask, 'batch_size'] +
        DECODE_PARAMS['B'] * np.log(total_len) +
        DECODE_PARAMS['C']
    )
    return df

def evaluate_model(df, stage, metric_col, pred_col):
    """计算 MAPE 并打印结果"""
    valid_df = df[(df[metric_col] > 0) & (df[pred_col].notna())]
    if valid_df.empty:
        print(f"⚠️ 没有有效的 {stage} 数据用于验证。")
        return 0.0
    
    mape = np.mean(np.abs((valid_df[metric_col] - valid_df[pred_col]) / valid_df[metric_col])) * 100
    print(f"📊 {stage} 验证结果 (MAPE): {mape:.2f}%")
    return mape

def plot_validation(df, stage, metric_col, pred_col, title):
    """绘制预测值 vs 真实值散点图"""
    valid_df = df[(df[metric_col] > 0) & (df[pred_col].notna())]
    if valid_df.empty: return
    
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'c']
    
    for i, tp in enumerate(sorted(valid_df['tp_size'].unique())):
        subset = valid_df[valid_df['tp_size'] == tp]
        plt.scatter(subset[metric_col], subset[pred_col], label=f'TP={tp}', color=colors[i%4], alpha=0.6)
    
    # 画一条 y=x 的完美预测线
    min_val = min(valid_df[metric_col].min(), valid_df[pred_col].min())
    max_val = max(valid_df[metric_col].max(), valid_df[pred_col].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction (y=x)')
    
    plt.xlabel(f'Real {metric_col} (ms)')
    plt.ylabel(f'Predicted {metric_col} (ms)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, f"validation_{stage.lower()}.png")
    plt.savefig(save_path)
    print(f"✅ 验证图表已保存至: {save_path}")

def main():
    print("🔄 正在加载数据并计算预测值...")
    df = pd.read_csv(DATA_FILE)
    numeric_cols = ['output_tokens', 'batch_size', 'prompt_tokens', 'ttft_mean_ms', 'tpot_mean_ms', 'tp_size']
    for col in numeric_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 只保留需要验证的 TP 配置
    df_validate = df[df['tp_size'].isin(VALIDATE_TP_SIZES)].copy()
    df_validate = calculate_metrics(df_validate)
    
    print(f"\n🎯 正在验证 TP={VALIDATE_TP_SIZES} 的数据 (未参与训练)...\n")
    
    # 1. 验证 Prefill
    print("🚀 Prefill (TTFT) 验证:")
    ttft_mape = evaluate_model(df_validate, "Prefill", "ttft_mean_ms", "pred_ttft")
    plot_validation(df_validate, "Prefill", "ttft_mean_ms", "pred_ttft", "Prefill Validation (TP=1,2,8)")
    
    # 2. 验证 Decode
    print("\n🏃 Decode (TPOT) 验证:")
    tpot_mape = evaluate_model(df_validate, "Decode", "tpot_mean_ms", "pred_tpot")
    plot_validation(df_validate, "Decode", "tpot_mean_ms", "pred_tpot", "Decode Validation (TP=1,2,8)")
    
    # 3. 最终裁决
    print("\n" + "="*50)
    max_mape = max(ttft_mape, tpot_mape)
    if max_mape < 15:
        print("🎉 恭喜！模型泛化能力极强，误差 < 15%，可以直接进入第三阶段（在线调度仿真）！")
    elif max_mape < 30:
        print("⚠️ 警告：模型误差在 15%-30% 之间。建议进入第三阶段，但在仿真器中加入通信开销补偿。")
    else:
        print("❌ 失败：模型误差 > 30%！跨 TP 预测严重失效。必须回头补充 AllReduce 通信测试，升级为灰盒模型！")

if __name__ == "__main__":
    main()