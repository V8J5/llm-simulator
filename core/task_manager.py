#!/usr/bin/env python3
"""
任务调度器 - 管理后台 Profiling 任务
支持实时日志、状态查询、任务取消
"""

import json
import os
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_FILE = os.path.join(PROJECT_ROOT, 'data', 'tasks.json')
os.makedirs(os.path.dirname(TASKS_FILE), exist_ok=True)

# 导入 auto_profile
sys.path.insert(0, PROJECT_ROOT)
from scripts.auto_profile import run_auto_profile


class TaskManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.tasks = self._load_tasks()
        self._lock = threading.RLock()
    
    def _load_tasks(self) -> List[Dict]:
        if os.path.exists(TASKS_FILE):
            try:
                with open(TASKS_FILE, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_tasks(self):
        with self._lock:
            with open(TASKS_FILE, 'w') as f:
                json.dump(self.tasks, f, indent=2, ensure_ascii=False)
    
    def _append_log(self, task_id: str, log_line: str):
        with self._lock:
            for task in self.tasks:
                if task['id'] == task_id:
                    if 'logs' not in task:
                        task['logs'] = []
                    task['logs'].append(log_line)
                    if len(task['logs']) > 1000:  # 限制日志长度
                        task['logs'] = task['logs'][-1000:]
                    self._save_tasks()
                    break
    
    def submit_task(self, task_type: str, params: Dict) -> str:
        """提交任务，返回任务 ID"""
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        task = {
            'id': task_id,
            'type': task_type,  # 'profile' | 'benchmark' | 'simulate'
            'status': 'pending',
            'params': params,
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'completed_at': None,
            'logs': [],
            'result': None,
            'error': None
        }
        
        with self._lock:
            self.tasks.append(task)
            self._save_tasks()
        
        # 启动后台线程执行
        thread = threading.Thread(target=self._run_task, args=(task_id,))
        thread.daemon = True
        thread.start()
        
        return task_id
    
    def _run_task(self, task_id: str):
        """后台执行任务（支持单个或批量 TP）"""
        # 获取任务
        task = None
        with self._lock:
            for t in self.tasks:
                if t['id'] == task_id:
                    task = t
                    break
        
        if not task:
            return
        
        def log_callback(msg):
            self._append_log(task_id, msg)
        
        try:
            task['status'] = 'running'
            task['started_at'] = datetime.now().isoformat()
            self._save_tasks()
            
            log_callback(f"🚀 任务 {task_id} 开始执行")
            log_callback(f"   类型: {task['type']}")
            log_callback(f"   参数: {json.dumps(task['params'], ensure_ascii=False)}")
            
            # ============================================================
            # 新增：处理批量 TP 采集任务
            # ============================================================
            if task['type'] == 'profile_batch':
                params = task['params']
                tps = params.get('tps', [params.get('tp', 1)])
                
                # 确保 tps 是列表
                if not isinstance(tps, list):
                    tps = [tps]
                
                # 去重并排序
                tps = sorted(set(int(t) for t in tps))
                
                log_callback(f"📋 共 {len(tps)} 个 TP 配置待采集: {tps}")
                
                # 存储子任务结果
                subtasks = []
                all_success = True
                
                for i, tp in enumerate(tps):
                    log_callback(f"\n{'='*50}")
                    log_callback(f"🚀 开始第 {i+1}/{len(tps)} 个 TP={tp} 的采集")
                    log_callback(f"{'='*50}")
                    
                    result = run_auto_profile(
                        host=params['host'],
                        user=params['user'],
                        password=params.get('password'),
                        key_file="/data/home/lihaozhe/.ssh/id_ed25519_work",  # ← 添加这一行
                        gpu_type=params['gpu_type'],
                        tp=tp,
                        port=params.get('port', 22),
                        skip_layer_bench=params.get('skip_layer_bench', True),  # ← 新增
                        price=params.get('price'),  # ← 新增
                        on_log=log_callback
                    )
                    
                    subtasks.append({
                        'tp': tp,
                        'success': result.get('success', False),
                        'result': result,
                        'error': result.get('error')
                    })
                    
                    if not result.get('success'):
                        all_success = False
                        log_callback(f"❌ TP={tp} 采集失败，继续下一个...")
                        # 如果某个 TP 失败，可以选择继续或中断
                        # 这里选择继续，让用户看到所有结果
                    else:
                        log_callback(f"✅ TP={tp} 采集完成")
                
                task['result'] = {
                    'total': len(tps),
                    'successful': sum(1 for s in subtasks if s['success']),
                    'failed': sum(1 for s in subtasks if not s['success']),
                    'subtasks': subtasks
                }
                task['status'] = 'completed' if all_success else 'partial'
                
                log_callback(f"\n{'='*50}")
                log_callback(f"📊 批量采集完成: {task['result']['successful']}/{task['result']['total']} 成功")
                if task['result']['failed'] > 0:
                    log_callback(f"⚠️ 失败的 TP: {[s['tp'] for s in subtasks if not s['success']]}")
                log_callback(f"{'='*50}")
            
            # ============================================================
            # 原有的单 TP 采集（保留兼容）
            # ============================================================
            elif task['type'] == 'profile':
                result = run_auto_profile(
                    host=task['params']['host'],
                    user=task['params']['user'],
                    password=task['params'].get('password'),
                    key_file="/data/home/lihaozhe/.ssh/id_ed25519_work",  # ← 添加这一行
                    gpu_type=task['params']['gpu_type'],
                    tp=task['params'].get('tp', 1),
                    port=task['params'].get('port', 22),
                    skip_layer_bench=params.get('skip_layer_bench', True),  # ← 新增
                    price=params.get('price'),  # ← 新增
                    on_log=log_callback
                )
                task['result'] = result
                task['status'] = 'completed' if result.get('success') else 'failed'
                if result.get('error'):
                    task['error'] = result['error']
            
            # ============================================================
            # 评分引擎任务（保持不变）
            # ============================================================
            elif task['type'] == 'benchmark':
                import subprocess
                cmd = [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, 'scripts', 'generate_hardware_benchmark.py'),
                    '--gpu-type', task['params']['gpu_type'],
                    '--tp', str(task['params'].get('tp', 1)),
                    '--no-run'
                ]
                if task['params'].get('price'):
                    cmd.extend(['--price', str(task['params']['price'])])
                log_callback(f"执行: {' '.join(cmd)}")
                result_proc = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
                task['result'] = {
                    'stdout': result_proc.stdout,
                    'stderr': result_proc.stderr,
                    'returncode': result_proc.returncode
                }
                task['status'] = 'completed' if result_proc.returncode == 0 else 'failed'
            
            task['completed_at'] = datetime.now().isoformat()
            log_callback(f"🏁 任务 {task_id} 完成，状态: {task['status']}")
            
        except Exception as e:
            import traceback
            task['status'] = 'failed'
            task['error'] = str(e)
            task['result'] = {'traceback': traceback.format_exc()}
            task['completed_at'] = datetime.now().isoformat()
            log_callback(f"❌ 任务 {task_id} 失败: {e}")
        
        self._save_tasks()
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        for task in self.tasks:
            if task['id'] == task_id:
                return task
        return None
    
    def get_tasks(self, limit: int = 50) -> List[Dict]:
        return self.tasks[-limit:]
    
    def get_logs(self, task_id: str, since: int = 0) -> List[str]:
        task = self.get_task(task_id)
        if not task:
            return []
        return task.get('logs', [])[since:]


# 单例
task_manager = TaskManager()