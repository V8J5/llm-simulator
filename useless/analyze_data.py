import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
import os

# 忽略拟合过程中的警告
warnings.filterwarnings("ignore")

# 设置中文字体，防止图表乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ================== 配置区 ==================
DATA_FILE = "/data/home/lihaozhe/llm-simulator/data/profiling_results.csv"
OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =============================================

def load_and_clean_data(filepath):
    print("🔄 正在加载并清洗数据...")
    df = pd.read_csv(filepath)
    
    # 1. 强制类型转换，防止字符串和整数比较导致筛选失败
    numeric_cols = ['output_tokens', 'batch_size', 'prompt_tokens', 'ttft_mean_ms', 'tpot_mean_ms', 'tp_size']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 2. 智能清洗策略
    #    - 对于 Prefill (output_tokens == 1): 只要 ttft > 0 就保留，允许 tpot 为 0
    #    - 对于 Decode (output_tokens > 1): 只要 tpot > 0 就保留，允许 ttft 为 0 (有些日志可能不记 ttft)
    
    initial_count = len(df)
    
    # 分离 Prefill 和 Decode 数据
    df_prefill = df[df['output_tokens'] == 1]
    df_decode = df[df['output_tokens'] > 1]
    
    # 清洗 Prefill: 必须 ttft > 0
    valid_prefill = df_prefill[df_prefill['ttft_mean_ms'] > 0]
    
    # 清洗 Decode: 必须 tpot > 0
    valid_decode = df_decode[df_decode['tpot_mean_ms'] > 0]
    
    # 合并
    df = pd.concat([valid_prefill, valid_decode])
    df = df.sort_index() # 恢复原始顺序
    
    print(f"✅ 清洗完成。剔除了 {initial_count - len(df)} 条无效数据 (OOM/Timeout)。")
    print(f"   剩余有效数据: {len(df)} 条 (Prefill: {len(valid_prefill)}, Decode: {len(valid_decode)})")
    
    return df

def fit_prefill_model(df):
    print("\n🚀 正在拟合 Prefill (TTFT) 模型...")
    # 筛选 Prefill 数据 (output_tokens == 1)
    prefill_df = df[df['output_tokens'] == 1].copy()
    
    if prefill_df.empty:
        print("⚠️ 警告: 没有找到有效的 Prefill 数据，跳过 Prefill 拟合。")
        return None

    # 模型公式: TTFT = a * (Batch * Prompt) + b * Batch + c * Prompt + d
    # 这是一个简化的线性模型，适合 Transformer 的计算复杂度分析
    def model_func(X, a, b, c, d):
        batch, prompt = X
        return a * batch * prompt + b * batch + c * prompt + d

    X_data = (prefill_df['batch_size'].values, prefill_df['prompt_tokens'].values)
    y_data = prefill_df['ttft_mean_ms'].values

    try:
        # 增加 maxfev 防止迭代次数不足
        popt, _ = curve_fit(model_func, X_data, y_data, maxfev=10000)
        a, b, c, d = popt
        print(f"✅ Prefill 模型拟合成功!")
        print(f"   公式: TTFT = {a:.6f} * (Batch * Prompt) + {b:.6f} * Batch + {c:.6f} * Prompt + {d:.6f}")
        return popt
    except Exception as e:
        print(f"❌ Prefill 模型拟合失败: {e}")
        print("   尝试使用更简单的模型 (仅依赖 Prompt 长度)...")
        # 降级方案：如果复杂模型拟合失败，尝试只拟合 Prompt 长度
        try:
            def simple_model(x, a, b):
                return a * x + b
            popt, _ = curve_fit(simple_model, prefill_df['prompt_tokens'].values, prefill_df['ttft_mean_ms'].values, maxfev=10000)
            print(f"   简化公式: TTFT = {popt[0]:.6f} * Prompt + {popt[1]:.6f}")
            return popt
        except:
            return None

def fit_decode_model(df):
    print("\n🏃 正在拟合 Decode (TPOT) 模型...")
    # 筛选 Decode 数据 (output_tokens > 1)
    decode_df = df[df['output_tokens'] > 1].copy()
    
    if decode_df.empty:
        print("⚠️ 警告: 没有找到有效的 Decode 数据，跳过 Decode 拟合。")
        return None

    # 模型公式: TPOT = a * Batch + b * log(Total_Len) + c
    def model_func(X, a, b, c):
        batch, total_len = X
        return a * batch + b * np.log(total_len) + c

    # Total Length = Prompt + Output
    total_lens = (decode_df['prompt_tokens'] + decode_df['output_tokens']).values
    
    X_data = (decode_df['batch_size'].values, total_lens)
    y_data = decode_df['tpot_mean_ms'].values

    try:
        popt, _ = curve_fit(model_func, X_data, y_data, maxfev=10000)
        a, b, c = popt
        print(f"✅ Decode 模型拟合成功!")
        print(f"   公式: TPOT = {a:.6f} * Batch + {b:.6f} * log(Total_Len) + {c:.6f}")
        return popt
    except Exception as e:
        print(f"❌ Decode 模型拟合失败: {e}")
        return None

def plot_results(df, prefill_popt, decode_popt):
    print("\n🎨 正在生成可视化图表...")
    
    if prefill_popt is None and decode_popt is None:
        print("⚠️ 没有有效的拟合结果，跳过绘图。")
        return

    # 根据拟合结果数量动态调整画布
    if prefill_popt is not None and decode_popt is not None:
        fig = plt.figure(figsize=(16, 6))
    else:
        fig = plt.figure(figsize=(8, 6))

    # --- 1. TTFT 3D 曲面图 ---
    if prefill_popt is not None:
        ax = fig.add_subplot(121, projection='3d')
        prefill_df = df[df['output_tokens'] == 1]
        
        # 绘制散点
        scatter = ax.scatter(prefill_df['batch_size'], prefill_df['prompt_tokens'], prefill_df['ttft_mean_ms'], c='r', label='Real Data')
        
        # 绘制拟合曲面
        # 生成网格数据
        batch_min, batch_max = prefill_df['batch_size'].min(), prefill_df['batch_size'].max()
        prompt_min, prompt_max = prefill_df['prompt_tokens'].min(), prefill_df['prompt_tokens'].max()
        
        # 避免除以0或范围太小
        if batch_max == batch_min: batch_max += 1
        if prompt_max == prompt_min: prompt_max += 1

        batch_grid = np.linspace(batch_min, batch_max, 10)
        prompt_grid = np.linspace(prompt_min, prompt_max, 10)
        B, P = np.meshgrid(batch_grid, prompt_grid)
        
        # 判断 popt 长度以适配不同模型
        if len(prefill_popt) == 4:
            Z = prefill_popt[0] * B * P + prefill_popt[1] * B + prefill_popt[2] * P + prefill_popt[3]
        else:
            # 简化模型的情况 (只依赖 Prompt)
            Z = prefill_popt[0] * P + prefill_popt[1]
            # 为了显示曲面，让 Z 在 Batch 维度上平铺
            B_flat = np.ones_like(P) * np.mean(batch_grid)
            B, P = B_flat, P

        ax.plot_surface(B, P, Z, alpha=0.3, color='b', label='Fitted Surface')
        
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Prompt Length')
        ax.set_zlabel('TTFT (ms)')
        ax.set_title('Prefill Performance (TTFT)')
        # ax.legend() # 3D图例有时显示有问题，可选

    # --- 2. TPOT 曲线图 ---
    if decode_popt is not None:
        # 如果左图没画，右图就占满整个画布
        if prefill_popt is None:
            ax2 = fig.add_subplot(111)
        else:
            ax2 = fig.add_subplot(122)
        
        # 选取几个典型的 Prompt 长度画散点
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        unique_prompts = sorted(df['prompt_tokens'].unique())
        
        for i, prompt_len in enumerate(unique_prompts):
            subset = df[(df['output_tokens'] > 1) & (df['prompt_tokens'] == prompt_len)]
            if not subset.empty:
                color = colors[i % len(colors)]
                ax2.scatter(subset['batch_size'], subset['tpot_mean_ms'], label=f'Real (Prompt={int(prompt_len)})', color=color, alpha=0.6)
        
        # 绘制拟合曲线 (取一个中间值的 Prompt 长度作为代表)
        rep_prompt = unique_prompts[len(unique_prompts)//2]
        batch_range = np.linspace(df['batch_size'].min(), df['batch_size'].max(), 100)
        total_len_rep = rep_prompt + np.mean(df[df['output_tokens']>1]['output_tokens'])
        
        tpot_pred = decode_popt[0] * batch_range + decode_popt[1] * np.log(total_len_rep) + decode_popt[2]
        ax2.plot(batch_range, tpot_pred, 'k--', linewidth=2, label=f'Fitted (Prompt≈{int(rep_prompt)})')
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('TPOT (ms)')
        ax2.set_title('Decode Performance (TPOT)')
        ax2.set_xscale('log', base=2)
        ax2.legend()
        ax2.grid(True, which="both", ls="-", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "performance_fitting.png"))
    print(f"✅ 图表已保存至: {os.path.join(OUTPUT_DIR, 'performance_fitting.png')}")

def main():
    df = load_and_clean_data(DATA_FILE)
    
    # 只针对 TP=4 拟合 (或者你可以改成 tp_size == 8 等)
    target_tp = 4
    df_target = df[df['tp_size'] == target_tp]
    
    if df_target.empty:
        print(f"⚠️ 警告：数据中没有找到 TP={target_tp} 的数据，将使用所有数据进行拟合。")
        df_target = df
    
    prefill_params = fit_prefill_model(df_target)
    decode_params = fit_decode_model(df_target)
    plot_results(df_target, prefill_params, decode_params)
    
    print("\n🎉 分析完成！")

if __name__ == "__main__":
    main()