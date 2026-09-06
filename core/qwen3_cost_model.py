import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from memory_estimator import MemoryEstimator  


class LLMCostModel:
    def __init__(self, data_dir: str, 
                 peak_tflops: float = 119.5,      
                 mem_bw_gb_s: float = 864.0,
                # ===== 新增：端到端系统开销补偿参数 =====
                 enable_e2e_compensation: bool = True,
                 prefill_base_scale: float = 4.8,
                 prefill_tau: float = 500,
                 decode_base_scale: float = 4.5,
                 decode_tau: float = 500):    
        """
        初始化成本模型
        :param data_dir: 存放 profiling JSON 文件的目录
        :param peak_tflops: 单卡 FP16 峰值算力 (TFLOPS)，用于外推
        :param mem_bw_gb_s: 单卡显存带宽 (GB/s)，用于外推
        """
        self.data_dir = data_dir
        self.peak_tflops = peak_tflops
        self.mem_bw_gb_s = mem_bw_gb_s

        self.enable_e2e_compensation = enable_e2e_compensation
        self.prefill_base_scale = prefill_base_scale
        self.prefill_tau = prefill_tau
        self.decode_base_scale = decode_base_scale
        self.decode_tau = decode_tau
        
        
        # 1. 加载 Prefill 计算表
        compute_path = os.path.join(data_dir, "qwen3_32b_prefill_lookup_table.json")
        if os.path.exists(compute_path):
            with open(compute_path, 'r') as f:
                self.compute_table = json.load(f)
            print(f"✅ 已加载 Prefill 计算模型: {compute_path} ({len(self.compute_table)} 条)")
        else:
            raise FileNotFoundError(f"未找到计算查找表: {compute_path}")

        # 2. 加载 Decode 计算表
        decode_path = os.path.join(data_dir, "qwen3_32b_decode_lookup_table.json")
        if os.path.exists(decode_path):
            with open(decode_path, 'r') as f:
                self.decode_table = json.load(f)
            print(f"✅ 已加载 Decode 计算模型: {decode_path} ({len(self.decode_table)} 条)")
        else:
            raise FileNotFoundError(f"未找到 Decode 查找表: {decode_path}")

        # 3. 加载通信表 (TP=1,2,4,8)
        self.comm_tables: Dict[int, List] = {}
        for tp_size in [1, 2, 4, 8]:
            comm_path = os.path.join(data_dir, f"comm_lookup_table_tp{tp_size}.json")
            if os.path.exists(comm_path):
                with open(comm_path, 'r') as f:
                    self.comm_tables[tp_size] = json.load(f)
                print(f"✅ 已加载通信模型 (TP={tp_size})")
            else:
                print(f"⚠️ 未找到通信表 TP={tp_size}，该配置将不可用")

        # 模型固定参数
        self.num_layers = 64
        self.hidden_size = 5120
        self.intermediate_size = 25600
        self.bytes_per_elem = 2  # FP16 = 2 bytes

        # ========== 新增：初始化显存估算器 ==========
        self.memory_estimator = MemoryEstimator(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_kv_heads=8,
            head_dim=80,
            num_params=32_000_000_000,
            bytes_per_param=self.bytes_per_elem,
            safety_margin=0.10
        )

        # 构建计算表的网格索引，供插值使用
        self._build_compute_grid()

    def _compute_e2e_scale(self, workload: int, base_scale: float, tau: float) -> float:
        """
        计算端到端系统开销缩放因子
        
        公式: scale = 1 + (base_scale - 1) × exp(-workload / tau)
        
        物理含义:
        - workload 很小时: scale ≈ base_scale（固定开销占比大）
        - workload 很大时: scale ≈ 1.0（固定开销被稀释）
        
        Args:
            workload: 计算量（Prefill: batch × seq_len, Decode: batch × kv_len）
            base_scale: 小 workload 时的最大缩放
            tau: 衰减常数（workload 越大，缩放越小）
        """
        if not self.enable_e2e_compensation:
            return 1.0
        
        scale = 1 + (base_scale - 1) * math.exp(-workload / tau)
        # 限制范围：不能小于 1.0，不能大于 base_scale
        return min(max(scale, 1.0), base_scale)

    # ================= 核心插值与外推工具 =================
    def _build_compute_grid(self):
        """提取计算表的唯一 batch 和 seq/kv 维度，构建排序列表"""
        # Prefill 网格
        batch_set = sorted(set(d['batch_size'] for d in self.compute_table))
        seq_set = sorted(set(d['prompt_length'] for d in self.compute_table))
        self.prefill_grid = (batch_set, seq_set) # prefill_grid = ([1,2,4,8,16,32], [128,512,1024,2048,4096])
        
        # Decode 网格
        batch_set_d = sorted(set(d['batch_size'] for d in self.decode_table))
        kv_set_d = sorted(set(d['kv_length'] for d in self.decode_table))
        self.decode_grid = (batch_set_d, kv_set_d) # decode_grid = ([1,2,4,8,16,32], [128,512,1024,2048,4096])

    def _interpolate_2d(self, table: List[Dict], x_key: str, y_key: str, val_key: str, 
                        x_val: int, y_val: int, grid: Tuple[List, List]) -> float:
        """
            二维线性插值（在网格内）。如果超出边界，使用最近边界外推（按比例缩放）。
            简单来说：给定一个不在查找表上的点 (x_val, y_val)，从周围的已知点估算出它的值。
        """
        x_list, y_list = grid
        if not x_list or not y_list:
            return 0.0

        # 1. 边界裁剪（用于外推） 如果查询值超出查找表范围，不崩溃，而是裁剪到边界。例如用户查询 seq=8192 → 裁剪为 4096，然后用边界点的值做外推
        x_clamped = max(min(x_val, x_list[-1]), x_list[0])
        y_clamped = max(min(y_val, y_list[-1]), y_list[0])
        
        # 如果完全匹配，直接返回
        for item in table:
            if item[x_key] == x_clamped and item[y_key] == y_clamped:
                return item[val_key]

        # 2. 找到四个相邻点（或边界点）
        # 寻找 x 轴左右索引
        x_idx = np.searchsorted(x_list, x_clamped)

        if x_idx == 0:
            x0, x1 = x_list[0], x_list[1]
            frac_x = 0
        elif x_idx >= len(x_list) - 1:
            x0, x1 = x_list[-2], x_list[-1]
            frac_x = 1
        else:
            x0, x1 = x_list[x_idx-1], x_list[x_idx]
            frac_x = (x_clamped - x0) / (x1 - x0) if x1 != x0 else 0

        y_idx = np.searchsorted(y_list, y_clamped)
        if y_idx == 0:
            y0, y1 = y_list[0], y_list[1]
            frac_y = 0
        elif y_idx >= len(y_list) - 1:
            y0, y1 = y_list[-2], y_list[-1]
            frac_y = 1
        else:
            y0, y1 = y_list[y_idx-1], y_list[y_idx]
            frac_y = (y_clamped - y0) / (y1 - y0) if y1 != y0 else 0

        # 3. 获取四个点的值（若不存在则按比例估算，但这里为了精确，用最近邻保底）
        def get_val(xx, yy):
            for item in table:
                if item[x_key] == xx and item[y_key] == yy:
                    return item[val_key]
            # 如果网格点缺失（极少情况），用最近邻
            near = min(table, key=lambda d: abs(d[x_key]-xx) + abs(d[y_key]-yy))
            return near[val_key]

        v00 = get_val(x0, y0)
        v01 = get_val(x0, y1)
        v10 = get_val(x1, y0)
        v11 = get_val(x1, y1)

        # 4. 双线性插值
        if x0 == x1 or y0 == y1:
            return (v00 + v01 + v10 + v11) / 4
        
        v_y0 = v00 + (v10 - v00) * frac_x
        v_y1 = v01 + (v11 - v01) * frac_x
        result = v_y0 + (v_y1 - v_y0) * frac_y
        
        # 如果是外推（超出边界），按 batch*seq 的乘积比例微调，并取与插值的较大者（保守）
        if x_val > x_list[-1] or y_val > y_list[-1]:
            # 用最后一个点作为基准，按计算量增长比例放大
            last_point = min(table, key=lambda d: abs(d[x_key]-x_list[-1]) + abs(d[y_key]-y_list[-1]))
            base_val = last_point[val_key]
            scale_factor = (x_val * y_val) / (x_list[-1] * y_list[-1])
            extrapolated = base_val * scale_factor
            # 取较大的（保守估计，不低估时间）
            return max(result, extrapolated)
        
        return result

    def _roofline_compute(self, flops: float, bytes: float) -> float: #后备方案。如果查找表中没有某个配置的数据，且无法插值（如极端值），用 Roofline 模型做理论估算。 目前尚未被调用
        """Roofline 模型：给定计算量和访存量，返回预估时间 (ms)"""
        time_flops = flops / (self.peak_tflops * 1e12)  # 秒
        time_bw = bytes / (self.mem_bw_gb_s * 1e9)      # 秒
        return max(time_flops, time_bw) * 1000.0        # 转毫秒

    def _get_comm_time(self, tp_size: int, msg_size_mb: float) -> float:
        """纯查表插值版本，依赖通信表中的测量点"""
        if tp_size not in self.comm_tables or not self.comm_tables[tp_size]:
            return 0.0

        table = self.comm_tables[tp_size]
        points = sorted([(d['msg_size_mb'], d['allreduce_ms']) for d in table])

        if not points:
            return 0.0

        # 如果消息小于最小测量点，线性外推（从原点）
        if msg_size_mb <= points[0][0]:
            if points[0][0] > 0:
                slope = points[0][1] / points[0][0]
                return max(slope * msg_size_mb, 0.001)
            return points[0][1]

        # 如果消息大于最大点，线性外推
        if msg_size_mb >= points[-1][0]:
            if points[-1][0] > 0:
                slope = points[-1][1] / points[-1][0]
                return slope * msg_size_mb
            return points[-1][1]

        # 正常插值
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            if x0 <= msg_size_mb <= x1:
                ratio = (msg_size_mb - x0) / (x1 - x0)
                comm_time = y0 + ratio * (y1 - y0)

                # ========== TP=4 通信高估修正 ==========
                if tp_size == 4:
                    # 基于消息大小分段修正
                    if msg_size_mb < 1.0:
                        # 小消息：基本不修正（误差小）
                        pass
                    elif msg_size_mb < 10.0:
                        # 中等消息：修正 10-20%
                        scale = 0.90 - (msg_size_mb / 10.0) * 0.10
                        return comm_time * scale
                    else:
                        # 大消息：修正 25-40%
                        scale = 0.75 - (min(msg_size_mb, 256) / 256.0) * 0.15
                        scale = max(scale, 0.60)  # 最多降低 40%
                        return comm_time * scale
                
                return comm_time

        return 0.0

    # ================= 核心预测 API =================
    def predict_prefill(self, batch_size: int, seq_len: int, tp_size: int) -> Dict:
        """
        预测 Prefill 阶段耗时 (单层累加 + 通信同步)
        """
        # 1. 计算侧时间 (二维插值)
        single_layer_compute = self._interpolate_2d(
            self.compute_table, 
            'batch_size', 'prompt_length', 'ffn_gemm_ms', 
            batch_size, seq_len, self.prefill_grid
        ) + self._interpolate_2d(
            self.compute_table, 
            'batch_size', 'prompt_length', 'attention_prefill_ms', 
            batch_size, seq_len, self.prefill_grid
        )
        # 累加64层
        total_compute = single_layer_compute * self.num_layers

        # 2. 通信侧时间 (修正公式：AllReduce 传输的是完整 hidden，不是 shard)
        # 标准 Megatron TP: 每层 2 次 AllReduce (Attention 后 + MLP 第一个线性层后)
        
        msg_size_bytes = batch_size * seq_len * self.hidden_size * self.bytes_per_elem # 通信量计算
        msg_size_mb = msg_size_bytes / (1024 * 1024)
        comm_per_layer = self._get_comm_time(tp_size, msg_size_mb)

        # 每层 2 次 AllReduce，累加 64 层
        total_comm = comm_per_layer * 2 * self.num_layers  # 每层2次

        # 3. ===== 新增：端到端系统开销补偿 =====
        if self.enable_e2e_compensation and tp_size == 1:
            workload = batch_size * seq_len
            scale = self._compute_e2e_scale(workload, self.prefill_base_scale, self.prefill_tau)
            total_compute = total_compute * scale

        # 4. 总时间 (采用同步串行模型，假设计算和通信不重叠，顺序执行)
        total_time = total_compute + total_comm # 实际 GPU：计算和通信可以重叠（Async AllReduce），但重叠比例不确定。采用串行模型是保守估计

        # ========== 新增：获取显存估算 ==========
        mem_info = self.memory_estimator.estimate_total(
            batch_size=batch_size,
            seq_len=seq_len,
            tp_size=tp_size,
            stage='prefill'
        )

        return {
            "stage": "prefill",
            "batch_size": batch_size,
            "seq_len": seq_len,
            "tp_size": tp_size,
            "total_time_ms": round(total_time, 2),
            "breakdown": {
                "compute_ms": round(total_compute, 2),
                "comm_ms": round(total_comm, 2),
                "comm_ratio": f"{(total_comm/total_time)*100:.1f}%" if total_time > 0 else "0%"
            },
            "memory_mb": mem_info  # ← 新增字段
        }

    def predict_decode(self, batch_size: int, kv_len: int, tp_size: int) -> Dict:
        """
        预测 Decode 阶段单次 Iteration 耗时 (TPOT)
        """
        # 1. 计算侧时间 (二维插值)
        single_layer_compute = self._interpolate_2d(
            self.decode_table, 
            'batch_size', 'kv_length', 'ffn_gemv_ms', 
            batch_size, kv_len, self.decode_grid
        ) + self._interpolate_2d(
            self.decode_table, 
            'batch_size', 'kv_length', 'attention_decode_ms', 
            batch_size, kv_len, self.decode_grid
        )
        total_compute = single_layer_compute * self.num_layers

        # 2. 通信侧时间 (Decode 时 AllReduce 传输量：batch * 1 * hidden)
        msg_size_bytes = batch_size * 1 * self.hidden_size * self.bytes_per_elem
        msg_size_mb = msg_size_bytes / (1024 * 1024)
        
        comm_per_layer = self._get_comm_time(tp_size, msg_size_mb)
        total_comm = comm_per_layer * 2 * self.num_layers

        # 3. ===== 端到端系统开销补偿 =====
        if self.enable_e2e_compensation and tp_size == 1:
            # Decode 特殊补偿：误差随 KV 长度增大而增大
            # 从数据拟合：kv=128 时缩放 1.2，kv=4096 时缩放 3.9
            # 公式：scale = 1 + (kv_len / 4096) * 2.7
            # 2.7 来自 (3.9 - 1.2) 的近似值
            kv_scale = 1 + (kv_len / 4096) * 2.7
            total_compute = total_compute * kv_scale

        # 4. 总时间 (同步串行模型)
        total_time = total_compute + total_comm

        # ========== 新增：获取显存估算 ==========
        mem_info = self.memory_estimator.estimate_total(
            batch_size=batch_size,
            seq_len=kv_len,  # Decode 中 seq_len 实际是 kv_len
            tp_size=tp_size,
            stage='decode'
        )

        return {
            "stage": "decode",
            "batch_size": batch_size,
            "kv_length": kv_len,
            "tp_size": tp_size,
            "total_time_ms": round(total_time, 2),
            "tokens_per_sec": round(1000.0 / total_time, 2) if total_time > 0 else 0,
            "breakdown": {
                "compute_ms": round(total_compute, 2),
                "comm_ms": round(total_comm, 2),
                "comm_ratio": f"{(total_comm/total_time)*100:.1f}%" if total_time > 0 else "0%"
            },
            "memory_mb": mem_info  # ← 新增字段
        }

    def save_prediction(self, result: Dict, save_path: str):
        """持久化保存预测结果 (追加模式)"""
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        result_with_timestamp = result.copy()
        result_with_timestamp["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        history = []
        if os.path.exists(save_path):
            try:
                with open(save_path, 'r') as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ 历史文件损坏，覆盖写入")

        history.append(result_with_timestamp)
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
        print(f"💾 预测结果已保存至: {save_path}")


# ================= 使用示例 =================
if __name__ == "__main__":
    DATA_DIR = "/data/home/lihaozhe/llm-simulator/data"
    SAVE_PATH = os.path.join(DATA_DIR, "predictions_history_v2.json")
    
    # 初始化引擎（L20 配置）
    engine = LLMCostModel(DATA_DIR, peak_tflops=119.5, mem_bw_gb_s=864.0)
    
    # 测试 Prefill (注意：tp=8 需要你有对应的通信表)
    prefill_result = engine.predict_prefill(batch_size=8, seq_len=2048, tp_size=8)
    print("\n" + "="*40)
    print(f"🚀 Prefill 预测结果: {prefill_result}")
    
    decode_result = engine.predict_decode(batch_size=8, kv_len=4096, tp_size=8)
    print(f"🚀 Decode 预测结果: {decode_result}")
    print("="*40)
    
    # 保存
    engine.save_prediction(prefill_result, SAVE_PATH)
    engine.save_prediction(decode_result, SAVE_PATH)