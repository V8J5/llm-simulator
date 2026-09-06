# import os
# import sys
# import time
# import json
# import subprocess
# import signal
# import csv
# import requests
# from datetime import datetime

# # ================== 配置区 ==================
# MODEL_PATH = "/data/home/public/weight/Qwen3-32B"
# HOST = "127.0.0.1"
# PORT = 8000

# # 测试维度 (对齐你的 Cost Model 测试矩阵)
# TP_SIZES = [1, 2, 4, 8]       
# BATCH_SIZES = [1, 4, 8, 16, 32]        
# PROMPT_LENGTHS = [512, 1024, 2048, 4096]    

# # 1 用于纯 Prefill 测试 (TTFT)，128 用于 Decode 测试 (TPOT)
# OUTPUT_LENGTHS = [1, 128]       
# RESULTS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/vllm_ground_truth.csv'))
# RESULT_JSON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/vllm_raw_results'))
# # =============================================

# def get_vllm_cmd(tp_size):
#     return [
#         sys.executable, "-m", "vllm.entrypoints.openai.api_server",
#         "--host", HOST, "--port", str(PORT),
#         "--model", MODEL_PATH,
#         "--tensor-parallel-size", str(tp_size),
#         "--gpu-memory-utilization", "0.90",
#         "--max-model-len", "8192",  # 稍微调大，防止 4096+128 时 OOM
#         "--disable-log-requests"    # 关闭请求日志，防止 IO 影响性能
#     ]

# def wait_for_server(process, timeout=600):
#     print(f"⏳ 正在等待服务启动，最长等待 {timeout} 秒...")
#     start_time = time.time()
#     while time.time() - start_time < timeout:
#         if process.poll() is not None:
#             print(f"❌ 服务进程意外退出，返回码: {process.returncode}")
#             return False
#         try:
#             if requests.get(f"http://{HOST}:{PORT}/health", timeout=2).status_code == 200:
#                 print("✅ 服务已就绪")
#                 return True
#         except requests.exceptions.RequestException:
#             pass
#         time.sleep(3)
#     print("❌ 服务启动超时")
#     return False

# def run_benchmark(prompt_len, output_len, concurrency, tp_size):
#     """执行单次压测，使用 --save-result 输出 JSON 以确保数据绝对准确"""
#     json_filename = f"tp{tp_size}_b{concurrency}_in{prompt_len}_out{output_len}.json"
#     json_path = os.path.join(RESULT_JSON_DIR, json_filename)

#     cmd = [
#         "vllm", "bench", "serve",
#         "--backend", "openai",
#         "--base-url", f"http://{HOST}:{PORT}",
#         "--model", MODEL_PATH,
#         "--dataset-name", "random",
#         "--num-prompts", str(concurrency),
#         "--request-rate", "inf",
#         "--max-concurrency", str(concurrency),
#         "--seed", "0",
#         "--random-input-len", str(prompt_len),
#         "--random-output-len", str(output_len),
#         "--save-result",             # 💡 核心修改：保存为 JSON
#         "--result-dir", RESULT_JSON_DIR,
#         "--result-filename", json_filename
#     ]
    
#     print(f"   🚀 压测中: Batch={concurrency}, In={prompt_len}, Out={output_len}")
#     result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    
#     if result.returncode != 0 or not os.path.exists(json_path):
#         print(f"   ❌ 压测失败或 JSON 未生成: {result.stderr[:200]}")
#         return None

#     # 💡 核心修改：直接解析 JSON，抛弃脆弱的正则表达式
#     try:
#         with open(json_path, 'r') as f:
#             data = json.load(f)
        
#         return {
#             "ttft_mean_ms": data.get("mean_ttft_ms", 0.0),
#             "tpot_mean_ms": data.get("mean_tpot_ms", 0.0),
#             "itl_mean_ms": data.get("mean_itl_ms", 0.0),
#             "req_throughput": data.get("request_throughput", 0.0),
#             "out_throughput": data.get("output_throughput", 0.0)
#         }
#     except Exception as e:
#         print(f"   ❌ JSON 解析失败: {e}")
#         return None

# def main():
#     # 💡 修复1：在循环外增加启动提示
#     print("🔥 脚本已启动，正在初始化环境...")
    
#     for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
#         os.environ.pop(var, None)
#     os.environ["NO_PROXY"] = "localhost,127.0.0.1"
    
#     os.makedirs(RESULT_JSON_DIR, exist_ok=True)

#     # 写入 CSV 表头
#     with open(RESULTS_FILE, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "timestamp", "tp_size", "batch_size", "prompt_tokens", 
#             "kv_length", "output_tokens", "ttft_mean_ms", "tpot_mean_ms", 
#             "itl_mean_ms", "req_throughput", "out_throughput"
#         ])
#     print(f"📝 Ground Truth 结果将保存至: {RESULTS_FILE}")
    
#     # 💡 修复2：打印测试矩阵，确认 TP_SIZES 是否为空
#     print(f"📊 测试矩阵: TP={TP_SIZES}, Batch={BATCH_SIZES}")

#     if not TP_SIZES:
#         print("⚠️ 警告: TP_SIZES 为空，没有需要测试的配置！")
#         return

#     for tp in TP_SIZES:
#         print(f"\n{'='*60}")
#         print(f"🔥 开始测试 TP={tp} 配置")
#         print(f"{'='*60}")
        
#         # ... (中间的 vLLM 启停和压测逻辑保持不变) ...
#         vllm_cmd = get_vllm_cmd(tp)
#         log_file = open(f"vllm_server_tp{tp}.log", "w", buffering=1)
#         process = subprocess.Popen(vllm_cmd, stdout=log_file, stderr=log_file)
        
#         try:
#             if not wait_for_server(process):
#                 continue

#             print("   🔥 正在执行 Warmup 请求...")
#             run_benchmark(512, 10, 1, tp)
#             time.sleep(2)

#             total_tests = len(BATCH_SIZES) * len(PROMPT_LENGTHS) * len(OUTPUT_LENGTHS)
#             current_test = 0
            
#             for batch in BATCH_SIZES:
#                 for prompt_len in PROMPT_LENGTHS:
#                     for output_len in OUTPUT_LENGTHS:
#                         current_test += 1
#                         print(f"\n[{current_test}/{total_tests}] TP={tp} | Batch={batch} | In={prompt_len} | Out={output_len}")
                        
#                         metrics = run_benchmark(prompt_len, output_len, batch, tp)
                        
#                         if metrics:
#                             avg_kv_length = prompt_len + (output_len // 2) if output_len > 1 else prompt_len
                            
#                             with open(RESULTS_FILE, 'a', newline='') as f:
#                                 writer = csv.writer(f)
#                                 writer.writerow([
#                                     datetime.now().isoformat(), tp, batch, prompt_len,
#                                     avg_kv_length, output_len,
#                                     metrics["ttft_mean_ms"], metrics["tpot_mean_ms"],
#                                     metrics["itl_mean_ms"], metrics["req_throughput"], metrics["out_throughput"]
#                                 ])
#                             print(f"   ✅ TTFT={metrics['ttft_mean_ms']:.2f}ms | TPOT={metrics['tpot_mean_ms']:.2f}ms | KV_Len={avg_kv_length}")
                        
#                         time.sleep(3)
#         finally:
#             print(f"\n🛑 正在关闭 TP={tp} 服务...")
#             process.send_signal(signal.SIGINT)
#             try: process.wait(timeout=30)
#             except subprocess.TimeoutExpired: process.kill()
#             log_file.close()

#     # 💡 修复3：在循环外增加结束提示
#     print("\n🎉 所有维度的压测已完成！")

import os
import re
import sys
import time
import subprocess
import signal
import csv
import requests
from datetime import datetime

# ================== 配置区 ==================
MODEL_PATH = "/data/home/public/weight/Qwen3-32B"
HOST = "127.0.0.1"  # 强制使用 IP，避免 localhost 解析问题
PORT = 8000

# 测试维度 
TP_SIZES = [1,2,4,8]       
BATCH_SIZES = [1, 2, 4, 8, 16, 32]        # 模拟不同并发
PROMPT_LENGTHS = [128, 512, 1024, 2048, 4096]    # 增加梯度，覆盖长文本
OUTPUT_LENGTHS = [1, 128]       # 1 用于纯 Prefill 测试，128 用于 Decode 测试

RESULTS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/profiling_results.csv'))
# =============================================

def get_vllm_cmd(tp_size):
    """根据 TP 大小动态生成启动命令"""
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--host", HOST,
        "--port", str(PORT),
        "--model", MODEL_PATH,
        "--tensor-parallel-size", str(tp_size),
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "4096",
    ]

def wait_for_server(process, timeout=600):
    """等待 vLLM 服务启动，并监控进程状态"""
    print(f"⏳ 正在等待服务启动，最长等待 {timeout} 秒...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        # 1. 检查进程是否已经退出
        if process.poll() is not None:
            print(f"❌ 服务进程意外退出，返回码: {process.returncode}")
            return False
        
        # 2. 检查服务是否就绪
        try:
            resp = requests.get(f"http://{HOST}:{PORT}/health", timeout=2)
            if resp.status_code == 200:
                print("✅ 服务已就绪")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(3)
    
    print("❌ 服务启动超时")
    return False

def run_benchmark(prompt_len, output_len, concurrency):
    """执行单次压测并解析结果"""
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--base-url", f"http://{HOST}:{PORT}",
        "--model", MODEL_PATH,
        "--dataset-name", "random",
        "--num-prompts", str(concurrency),
        "--request-rate", "inf",
        "--max-concurrency", str(concurrency),
        "--seed", "0",
        "--random-input-len", str(prompt_len),
        "--random-output-len", str(output_len),
    ]
    print(f"   🚀 压测中: Batch={concurrency}, In={prompt_len}, Out={output_len}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
        print(f"   ❌ 压测失败: {result.stderr[:200]}")
        return None

    # 正则提取指标
    patterns = {
        "ttft_mean_ms": r"Mean TTFT \(ms\):\s+([\d.]+)",
        "tpot_mean_ms": r"Mean TPOT \(ms\):\s+([\d.]+)",
        "req_throughput": r"Request throughput \(req/s\):\s+([\d.]+)",
        "out_throughput": r"Output token throughput \(tok/s\):\s+([\d.]+)",
    }
    
    metrics = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, result.stdout)
        metrics[key] = float(match.group(1)) if match else 0.0
        
    return metrics

def main():
    # 清除代理环境变量
    for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        os.environ.pop(var, None)
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"

    # 1. 创建结果文件并写入表头 (增加 TP 维度)
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "tp_size", "batch_size", "prompt_tokens", "output_tokens",
            "ttft_mean_ms", "tpot_mean_ms", "req_throughput", "out_throughput"
        ])
    print(f"📝 结果将保存至: {RESULTS_FILE}")

    # 2. 外层循环：遍历不同的 TP 配置
    for tp in TP_SIZES:
        print(f"\n{'='*50}")
        print(f"🔥 开始测试 TP={tp} 配置")
        print(f"{'='*50}")
        
        vllm_cmd = get_vllm_cmd(tp)
        log_file = open(f"vllm_server_tp{tp}.log", "w", buffering=1)
        process = subprocess.Popen(vllm_cmd, stdout=log_file, stderr=log_file)
        
        try:
            if not wait_for_server(process):
                continue  # 启动失败则跳过当前 TP

            # 3. 内层循环：遍历 Batch、Prompt、Output 组合
            total_tests = len(BATCH_SIZES) * len(PROMPT_LENGTHS) * len(OUTPUT_LENGTHS)
            current_test = 0
            
            for batch in BATCH_SIZES:
                for prompt_len in PROMPT_LENGTHS:
                    for output_len in OUTPUT_LENGTHS:
                        current_test += 1
                        print(f"\n[{current_test}/{total_tests}] TP={tp} | Batch={batch}")
                        
                        metrics = run_benchmark(prompt_len, output_len, batch)
                        
                        if metrics:
                            with open(RESULTS_FILE, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    datetime.now().isoformat(), tp, batch, prompt_len, output_len,
                                    metrics["ttft_mean_ms"], metrics["tpot_mean_ms"],
                                    metrics["req_throughput"], metrics["out_throughput"]
                                ])
                            print(f"   ✅ TTFT={metrics['ttft_mean_ms']:.2f}ms, TPOT={metrics['tpot_mean_ms']:.2f}ms")
                        
                        time.sleep(5) # 服务冷却
        finally:
            print(f"\n🛑 正在关闭 TP={tp} 服务...")
            process.send_signal(signal.SIGINT)
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: process.kill()
            log_file.close()

    print("\n🎉 所有维度的压测已完成！")

if __name__ == "__main__":
    main()