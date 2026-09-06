#!/usr/bin/env python3
"""
自动远程 Profiling - 登录目标机器，采集数据并回传
支持实时日志输出，供前端展示进度

用法（命令行）:
    python scripts/auto_profile.py --host 192.168.1.100 --user root --gpu-type 昇腾910B --tp 1

用法（作为模块调用）:
    from scripts.auto_profile import run_auto_profile
    result = run_auto_profile(host="192.168.1.100", user="root", gpu_type="昇腾910B", tp=1)
"""

import paramiko
import os
import sys
import json
import time
import re
import subprocess
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable
from scp import SCPClient
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


def run_remote_command(ssh_client, cmd: str, log_cb: Optional[Callable] = None) -> tuple:
    """执行远程命令，实时输出日志"""
    if log_cb:
        log_cb(f"执行: {cmd}")
    
    stdin, stdout, stderr = ssh_client.exec_command(cmd)
    
    out_lines = []
    err_lines = []
    
    # 使用非阻塞方式读取，确保所有输出都被捕获
    import select
    
    # 设置通道为非阻塞模式
    stdout.channel.setblocking(0)
    stderr.channel.setblocking(0)
    
    while not stdout.channel.exit_status_ready():
        # 读取 stdout
        if stdout.channel.recv_ready():
            try:
                line = stdout.channel.recv(1024).decode(errors='ignore')
                if line and log_cb:
                    for l in line.split('\n'):
                        if l.strip():
                            log_cb(f"[stdout] {l.strip()}")
                out_lines.append(line)
            except:
                pass
        
        # 读取 stderr
        if stderr.channel.recv_stderr_ready():
            try:
                line = stderr.channel.recv_stderr(1024).decode(errors='ignore')
                if line and log_cb:
                    for l in line.split('\n'):
                        if l.strip():
                            log_cb(f"[stderr] {l.strip()}")
                err_lines.append(line)
            except:
                pass
        
        time.sleep(0.05)
    
    # 命令执行完毕后，读取剩余输出
    try:
        remaining_out = stdout.channel.recv(4096).decode(errors='ignore')
        if remaining_out and log_cb:
            for l in remaining_out.split('\n'):
                if l.strip():
                    log_cb(f"[stdout] {l.strip()}")
        out_lines.append(remaining_out)
    except:
        pass
    
    try:
        remaining_err = stderr.channel.recv_stderr(4096).decode(errors='ignore')
        if remaining_err and log_cb:
            for l in remaining_err.split('\n'):
                if l.strip():
                    log_cb(f"[stderr] {l.strip()}")
        err_lines.append(remaining_err)
    except:
        pass
    
    return ''.join(out_lines), ''.join(err_lines)


def run_auto_profile(
    host: str,
    user: str,
    password: Optional[str] = None,
    key_file: Optional[str] = None,
    gpu_type: str = "Unknown",
    tp: int = 1,
    port: int = 22,
    skip_layer_bench: bool = True,  # ← 新增
    price: Optional[float] = None,  # ← 新增
    on_log: Optional[Callable] = None
) -> Dict:
    """
    执行自动 Profiling
    
    Returns:
        {
            'success': bool,
            'gpu_type': str,
            'tp': int,
            'data_dir': str,
            'files': list,
            'error': str or None
        }
    """
    
    result = {
        'success': False,
        'gpu_type': gpu_type,
        'tp': tp,
        'data_dir': '',
        'files': [],
        'error': None
    }
    
    def log(msg, level="info"):
        if on_log:
            on_log(msg)
    
    try:
        log(f"🔗 连接目标机器: {host}:{port}")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        ssh.connect(
            host,
            username=user,
            password=password,
            port=port,
            timeout=30,
            key_filename=key_file,
            allow_agent=False,
            look_for_keys=False
        )
        
        log("✅ SSH 连接成功")
        
        # ============================================================
        # 第一步：检查 Python 环境
        # ============================================================
        log("📊 检查 Python 环境...")
        stdin, stdout, stderr = ssh.exec_command("python3 --version 2>/dev/null || echo 'not found'")
        python_version = stdout.read().decode().strip()
        if "not found" in python_version:
            log("❌ Python3 未安装，尝试安装...")
            ssh.exec_command("apt-get update -qq 2>/dev/null && apt-get install -y python3 python3-pip -qq 2>/dev/null || yum install -y python3 python3-pip 2>/dev/null")
            time.sleep(5)
            stdin, stdout, stderr = ssh.exec_command("python3 --version 2>/dev/null || echo 'not found'")
            python_version = stdout.read().decode().strip()
            if "not found" in python_version:
                log("❌ Python3 安装失败，请手动安装", "error")
                result['error'] = "Python3 installation failed"
                return result
        log(f"   ✅ Python: {python_version}")
        
        # ============================================================
        # 检测 Conda 环境（优先使用）
        # ============================================================
        log("📊 检测 Conda 环境...")
        python_cmd = "python3"
        pip_cmd = "pip3"

        # 检测是否有 Conda
        stdin, stdout, stderr = ssh.exec_command("which conda 2>/dev/null || echo 'no_conda'")
        conda_output = stdout.read().decode().strip()
        if "no_conda" not in conda_output:
            log("   ✅ 检测到 Conda")
            stdin, stdout, stderr = ssh.exec_command("conda run which python 2>/dev/null || echo 'no_conda_python'")
            conda_python = stdout.read().decode().strip()
            if "no_conda_python" not in conda_python and conda_python:
                python_cmd = conda_python
                pip_cmd = f"{conda_python} -m pip"
                log(f"   🔹 使用 Conda Python: {python_cmd}")
            else:
                stdin, stdout, stderr = ssh.exec_command("which python 2>/dev/null || echo 'no_python'")
                conda_python = stdout.read().decode().strip()
                if "no_python" not in conda_python and conda_python:
                    python_cmd = conda_python
                    pip_cmd = f"{conda_python} -m pip"
                    log(f"   🔹 使用 Conda Python: {python_cmd}")
        else:
            # 检查是否有 miniconda3 目录
            stdin, stdout, stderr = ssh.exec_command("ls -d ~/miniconda3/bin/python 2>/dev/null || echo 'no_miniconda'")
            miniconda_python = stdout.read().decode().strip()
            if "no_miniconda" not in miniconda_python:
                python_cmd = "/home/lihz/miniconda3/bin/python"
                pip_cmd = "/home/lihz/miniconda3/bin/pip"
                log(f"   🔹 使用 Miniconda Python: {python_cmd}")

        log(f"   ✅ Python 命令: {python_cmd}")
        
        # ============================================================
        # 第二步：检查并安装 pip（使用 Conda pip 如果可用）
        # ============================================================
        log("📊 检查 pip...")
        stdin, stdout, stderr = ssh.exec_command(f"{pip_cmd} --version 2>/dev/null || echo 'not found'")
        pip_status = stdout.read().decode().strip()
        if "not found" in pip_status:
            log("📦 安装 pip...")
            ssh.exec_command(f"{python_cmd} -m ensurepip --upgrade 2>/dev/null || apt-get install -y python3-pip -qq 2>/dev/null || yum install -y python3-pip 2>/dev/null")
            time.sleep(3)
        log("   ✅ pip 就绪")
        
        # ============================================================
        # 第三步：检测 GPU 类型（提前执行，供 PyTorch 安装使用）
        # ============================================================
        log("📊 检测 GPU 类型...")
        
        has_nvidia = False
        has_ascend = False
        gpu_family = "unknown"
        
        # 检测 NVIDIA
        stdin, stdout, stderr = ssh.exec_command("nvidia-smi 2>/dev/null && echo 'nvidia_ok' || echo 'no_nvidia'")
        output = stdout.read().decode().strip()
        if "nvidia_ok" in output:
            has_nvidia = True
        
        # 检测昇腾
        stdin, stdout, stderr = ssh.exec_command("npu-smi 2>/dev/null && echo 'ascend_ok' || echo 'no_ascend'")
        output = stdout.read().decode().strip()
        if "ascend_ok" in output:
            has_ascend = True
        
        if has_nvidia:
            log("   ✅ 检测到 NVIDIA GPU")
            gpu_family = "nvidia"
        elif has_ascend:
            log("   ✅ 检测到 昇腾 NPU")
            gpu_family = "ascend"
        else:
            log("❌ 未检测到任何 GPU/NPU", "error")
            result['error'] = "No GPU/NPU detected"
            return result
        
        # ============================================================
        # 第四步：检查并安装 PyTorch（根据 GPU 类型）
        # ============================================================
        log("📊 检查 PyTorch...")
        stdin, stdout, stderr = ssh.exec_command(
            f"{python_cmd} -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed'"
        )
        torch_version = stdout.read().decode().strip()
        
        if "not installed" in torch_version:
            log("📦 PyTorch 未安装，正在自动安装（可能需要 2-3 分钟）...")
            
            if gpu_family == "nvidia":
                # 检测 CUDA 版本
                stdin, stdout, stderr = ssh.exec_command(
                    "nvcc --version 2>/dev/null | grep 'release' | sed 's/.*release //' | sed 's/,.*//' || echo 'cpu'"
                )
                cuda_version = stdout.read().decode().strip()
                
                if cuda_version and cuda_version != "cpu":
                    cuda_major = cuda_version.split('.')[0]
                    log(f"   🔹 检测到 CUDA {cuda_version}，安装 GPU 版本 PyTorch...")
                    if cuda_major == "12":
                        install_cmd = f"{pip_cmd} install torch --index-url https://download.pytorch.org/whl/cu121 -q"
                    elif cuda_major == "11":
                        install_cmd = f"{pip_cmd} install torch --index-url https://download.pytorch.org/whl/cu118 -q"
                    else:
                        install_cmd = f"{pip_cmd} install torch -q"
                else:
                    log("   🔹 未检测到 CUDA，安装 CPU 版本 PyTorch...")
                    install_cmd = f"{pip_cmd} install torch --index-url https://download.pytorch.org/whl/cpu -q"
            elif gpu_family == "ascend":
                log("   🔹 检测到昇腾 NPU，安装 torch_npu...")
                install_cmd = f"{pip_cmd} install torch torch_npu -q 2>/dev/null || {pip_cmd} install torch -q"
            else:
                log("   🔹 未检测到 GPU，安装 CPU 版本 PyTorch...")
                install_cmd = f"{pip_cmd} install torch --index-url https://download.pytorch.org/whl/cpu -q"
            
            # 执行安装
            stdin, stdout, stderr = ssh.exec_command(install_cmd)
            for _ in range(60):
                if stdout.channel.exit_status_ready():
                    break
                time.sleep(5)
                log("   ⏳ 仍在安装中，请稍候...")
            
            # 验证安装
            stdin, stdout, stderr = ssh.exec_command(
                f"{python_cmd} -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'install failed'"
            )
            torch_version = stdout.read().decode().strip()
            if "install failed" in torch_version:
                log("   ⚠️ PyTorch 安装失败，尝试使用国内镜像...", "warning")
                ssh.exec_command(f"{pip_cmd} install torch -i https://pypi.tuna.tsinghua.edu.cn/simple -q")
                time.sleep(30)
                stdin, stdout, stderr = ssh.exec_command(
                    f"{python_cmd} -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'install failed'"
                )
                torch_version = stdout.read().decode().strip()
                if "install failed" in torch_version:
                    log("❌ PyTorch 安装失败，请手动安装: {pip_cmd} install torch", "error")
                    result['error'] = "PyTorch installation failed"
                    return result
        
        log(f"   ✅ PyTorch: {torch_version}")
        
        # ============================================================
        # 第五步：显示 GPU 状态
        # ============================================================
        log("📊 检查 GPU 状态...")
        if gpu_family == "nvidia":
            stdin, stdout, stderr = ssh.exec_command("nvidia-smi 2>/dev/null | head -5")
        else:
            stdin, stdout, stderr = ssh.exec_command("npu-smi 2>/dev/null | head -5")
        gpu_info = stdout.read().decode().strip()
        if gpu_info:
            log(f"   GPU 信息:\n{gpu_info}")
        
        # ============================================================
        # 第六步：创建远程临时目录
        # ============================================================
        remote_dir = f"/tmp/benchmark_{gpu_type}_{int(time.time())}"
        log(f"📁 创建远程临时目录: {remote_dir}")
        run_remote_command(ssh, f"mkdir -p {remote_dir}", log)
        
        # ============================================================
        # 第七步：上传脚本
        # ============================================================
        log("📤 上传 Profiling 脚本...")
        scp = SCPClient(ssh.get_transport())
        
        scripts = [
            ('prefill_layer_bench.py', 'qwen3_32b_prefill_layer_lookup.json'),
            ('decode_layer_bench.py', 'qwen3_32b_decode_layer_lookup.json'),
            ('comm_micro_bench.py', f'comm_lookup_table_tp{tp}.json')
        ]
        
        uploaded_files = []
        for script_name, expected_output in scripts:
            local_path = os.path.join(PROJECT_ROOT, 'scripts', script_name)
            if not os.path.exists(local_path):
                log(f"⚠️ 脚本不存在: {script_name}")
                continue
            
            log(f"  上传 {script_name}")
            scp.put(local_path, remote_dir)
            uploaded_files.append((script_name, expected_output))
        
        scp.close()
        log(f"✅ 上传完成: {len(uploaded_files)} 个脚本")
        

        # ============================================================
        # 第八步：执行 Profiling（根据 skip_layer_bench 决定）
        # ============================================================

        if not skip_layer_bench:
            # 首次采集或需要重新采集层时间
            log("📊 准备执行 Prefill Profiling，清理显存...")
            stdout, stderr = run_remote_command(
                ssh, f"{python_cmd} -c 'import torch; torch.cuda.empty_cache(); print(\"✅ 显存已清理\")'", log
            )

            log("📊 执行 Prefill 整层 Profiling...")
            stdout, stderr = run_remote_command(
                ssh, f"cd {remote_dir} && {python_cmd} prefill_layer_bench.py", log
            )
            
            log("📊 执行 Decode 整层 Profiling...")
            stdout, stderr = run_remote_command(
                ssh, f"cd {remote_dir} && {python_cmd} decode_layer_bench.py", log
            )
        else:
            log("⏭️ 跳过层时间 Profiling，复用本地已有数据")
            # 从本地 data/{gpu_type}/ 目录复制已有的层时间文件到远程目录
            local_prefill_path = os.path.join(DATA_DIR, gpu_type, 'qwen3_32b_prefill_layer_lookup.json')
            local_decode_path = os.path.join(DATA_DIR, gpu_type, 'qwen3_32b_decode_layer_lookup.json')
            
            if os.path.exists(local_prefill_path) and os.path.exists(local_decode_path):
                # 用 scp 上传到远程目录
                scp = SCPClient(ssh.get_transport())
                scp.put(local_prefill_path, remote_dir)
                scp.put(local_decode_path, remote_dir)
                scp.close()
                log("✅ 已复用本地层时间数据")
            else:
                log("⚠️ 本地层时间数据不存在，将创建空文件", "warning")
                run_remote_command(
                    ssh,
                    f"cd {remote_dir} && echo '[]' > qwen3_32b_prefill_layer_lookup.json && echo '[]' > qwen3_32b_decode_layer_lookup.json",
                    log
                )

        # 通信 Profiling 总是执行（如果 TP > 1）
        if tp > 1:
            log(f"📊 执行通信 Profiling (TP={tp})...")
            stdout, stderr = run_remote_command(
                ssh, f"cd {remote_dir} && {python_cmd} -m torch.distributed.run --nproc_per_node={tp} comm_micro_bench.py", log
            )
        else:
            log("ℹ️ TP=1，跳过通信 Profiling")
        
        # ============================================================
        # 第九步：回传数据
        # ============================================================
        log("📥 回传 Profiling 数据...")
        local_data_dir = os.path.join(DATA_DIR, gpu_type)
        os.makedirs(local_data_dir, exist_ok=True)
        
        scp = SCPClient(ssh.get_transport())
        for _, expected_output in uploaded_files:
            remote_file = os.path.join(remote_dir, expected_output)
            try:
                scp.get(remote_file, local_data_dir)
                log(f"  ✅ 回传: {expected_output}")
                result['files'].append(expected_output)
            except Exception as e:
                log(f"  ⚠️ 回传失败: {expected_output} -> {e}", "warning")
        
        scp.close()
        result['data_dir'] = local_data_dir
        
        # 检查是否有足够的文件回传
        if len(result['files']) < 2:
            log("❌ 数据回传失败：缺少关键文件 (需要 prefill 和 decode)", "error")
            result['error'] = "Missing profiling data files"
            return result
        
        # ============================================================
        # 第十步：清理远程临时文件
        # ============================================================
        log("🧹 清理远程临时文件...")
        run_remote_command(ssh, f"rm -rf {remote_dir}", log)
        
        ssh.close()
        log("✅ SSH 连接已关闭")
        
        # ============================================================
        # 第十一步：自动运行评分引擎
        # ============================================================
        log("📊 自动生成评分报告...")
        benchmark_script = os.path.join(PROJECT_ROOT, 'scripts', 'generate_hardware_benchmark.py')
        cmd = [
            sys.executable, benchmark_script,
            '--gpu-type', gpu_type,
            '--tp', str(tp),
            '--no-run'
        ]
        # ===== 传递价格 =====
        if price:
            cmd.extend(['--price', str(price)])
        log(f"执行: {' '.join(cmd)}")
        result_proc = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        
        score_success = result_proc.returncode == 0
        if score_success:
            log("✅ 评分报告生成成功")
        else:
            log(f"⚠️ 评分报告生成失败: {result_proc.stderr}", "warning")
        
        # ============================================================
        # 第十二步：最终判断（根据 skip_layer_bench 动态判断）
        # ============================================================
        
        # 根据 skip_layer_bench 和 tp 判断需要的文件
        needed_files = []
        
        if not skip_layer_bench:
            # 需要层时间数据
            needed_files.append('qwen3_32b_prefill_layer_lookup.json')
            needed_files.append('qwen3_32b_decode_layer_lookup.json')
        else:
            log("ℹ️ 已跳过层时间采集，不检查层时间文件", "info")
        
        if tp > 1:
            # 需要通信数据
            needed_files.append(f'comm_lookup_table_tp{tp}.json')
        else:
            log("ℹ️ TP=1，不需要通信文件", "info")
        
        # 检查是否所有需要的文件都已回传
        all_needed_ok = all(f in result['files'] for f in needed_files)
        
        if all_needed_ok:
            result['success'] = True
            if score_success:
                log("🎉 自动 Profiling 完成！")
            else:
                log("🎉 数据采集完成，但评分报告生成失败", "warning")
                log(f"💡 可手动运行: python scripts/generate_hardware_benchmark.py --gpu-type {gpu_type} --tp {tp} --no-run", "info")
        else:
            result['success'] = False
            missing = [f for f in needed_files if f not in result['files']]
            log(f"❌ 采集不完整，缺少文件: {missing}", "error")
            if not result['error']:
                result['error'] = f"Missing files: {missing}"
            log("❌ 采集不完整，请检查日志", "error")
        
    except paramiko.AuthenticationException:
        log("❌ SSH 认证失败，请检查用户名和密码", "error")
        result['error'] = "Authentication failed"
    except paramiko.SSHException as e:
        log(f"❌ SSH 连接异常: {e}", "error")
        result['error'] = f"SSH error: {e}"
    except Exception as e:
        log(f"❌ 采集失败: {e}", "error")
        log(traceback.format_exc(), "error")
        result['error'] = str(e)
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True, help='目标机器 IP')
    parser.add_argument('--user', required=True, help='SSH 用户名')
    parser.add_argument('--password', help='SSH 密码')
    parser.add_argument('--key-file', help='SSH 私钥文件路径')
    parser.add_argument('--gpu-type', required=True, help='GPU 型号标识')
    parser.add_argument('--tp', type=int, default=1, help='TP 大小')
    parser.add_argument('--port', type=int, default=22, help='SSH 端口')
    parser.add_argument('--price', type=float, help='GPU 价格（元）')
    args = parser.parse_args()
    
    def print_log(msg):
        print(msg)
    
    result = run_auto_profile(
        host=args.host,
        user=args.user,
        password=args.password,
        key_file=args.key_file,
        gpu_type=args.gpu_type,
        tp=args.tp,
        port=args.port,
        price=args.price,  # ← 新增
        on_log=print_log
    )
    
    if result['success']:
        print(f"\n✅ 采集成功！数据已保存至: {result['data_dir']}")
        print(f"📁 文件: {result['files']}")
        sys.exit(0)
    else:
        print(f"\n❌ 采集失败: {result['error']}")
        sys.exit(1)


if __name__ == '__main__':
    main()