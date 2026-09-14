#!/usr/bin/env python3
"""
Benchmark 平台后端 API
启动: python web/app.py
访问: http://localhost:5000
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import os
import json
import glob
import sys
import csv
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# 添加项目根目录到 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.task_manager import task_manager
from core.benchmark_protocol import (
    adapt_legacy_report,
    compare_reports,
    load_benchmark_spec,
)

app = Flask(__name__, static_folder='.')
CORS(app)

REPORT_DIR = os.path.join(PROJECT_ROOT, 'data', 'benchmark_reports')
TASKS_FILE = os.path.join(PROJECT_ROOT, 'data', 'tasks.json')
BENCHMARK_SPEC_FILE = os.path.join(
    PROJECT_ROOT, 'configs', 'benchmark_specs', 'qwen3_32b_v1.json')


def _latest_report(gpu_type, tp):
    pattern = os.path.join(REPORT_DIR, f'benchmark_{gpu_type}_TP{tp}_*.json')
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def _load_comparable_report(path):
    """Load a report and explicitly adapt legacy v0 files in memory.

    Legacy files used seq/time and 1/time.  Recompute token throughput from
    batch and total time, and mark the provenance so the UI cannot present the
    result as independently verified ground truth.
    """
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    spec = load_benchmark_spec(BENCHMARK_SPEC_FILE)
    return adapt_legacy_report(payload, spec)


def _ground_truth_ratios(candidate_gpu, reference_gpu, tp):
    pattern = os.path.join(
        PROJECT_ROOT, 'data', 'relative_ground_truth', '**',
        'relative_ground_truth_report.json')
    matches = []
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
            acceptance = payload.get('acceptance', {})
            if not (acceptance.get('ground_truth_complete') and
                    acceptance.get('runtime_compatible') and
                    acceptance.get('repeatability_ok')):
                continue
            candidate = payload.get('candidate', {})
            reference = payload.get('reference', {})
            same = (candidate.get('hardware_id') == candidate_gpu and
                    reference.get('hardware_id') == reference_gpu and
                    int(candidate.get('tp_size', -1)) == tp and
                    int(reference.get('tp_size', -1)) == tp)
            reverse = (candidate.get('hardware_id') == reference_gpu and
                       reference.get('hardware_id') == candidate_gpu and
                       int(candidate.get('tp_size', -1)) == tp and
                       int(reference.get('tp_size', -1)) == tp)
            if same:
                matches.append((os.path.getmtime(path), path,
                                payload.get('ground_truth_ratios', {})))
            elif reverse:
                ratios = {key: 1.0 / float(value)
                          for key, value in payload.get('ground_truth_ratios', {}).items()
                          if float(value) > 0}
                matches.append((os.path.getmtime(path), path, ratios))
        except (OSError, ValueError, TypeError, ZeroDivisionError):
            continue
    if not matches:
        return None, None
    _, path, ratios = max(matches, key=lambda item: item[0])
    return ratios, path


def _get_task_from_file(task_id):
    """直接从文件读取任务，确保实时性"""
    if not os.path.exists(TASKS_FILE):
        return None
    try:
        with open(TASKS_FILE, 'r') as f:
            tasks = json.load(f)
        for task in tasks:
            if task.get('id') == task_id:
                return task
    except Exception as e:
        print(f"读取任务文件失败: {e}")
    return None

def _get_task_logs_from_file(task_id, since=0):
    """直接从文件读取日志"""
    task = _get_task_from_file(task_id)
    if not task:
        return []
    logs = task.get('logs', [])
    return logs[since:]

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/summary')
def get_summary():
    summary_csv = os.path.join(REPORT_DIR, 'benchmark_summary.csv')
    if os.path.exists(summary_csv):
        results = []
        with open(summary_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gpu = row['GPU']
                tp = int(row['TP']) if row['TP'] else 1
                
                # 构建 case_scores：从该 GPU 的所有 JSON 文件中读取数据
                case_scores = {}
                pattern = os.path.join(REPORT_DIR, f'benchmark_{gpu}_TP{tp}_*.json')
                files = glob.glob(pattern)
                data = None
                if files:
                    # 读取匹配的 JSON 文件（按 TP 精确匹配）
                    latest = max(files, key=os.path.getmtime)
                    data = _load_comparable_report(latest)
                    for item in data.get('prefill', []):
                        case_scores[item['case']] = item.get('throughput_tok_s', 0)
                    for item in data.get('decode', []):
                        case_scores[item['case']] = item.get('throughput_tok_s', 0)
                prefill_values = [item.get('throughput_tok_s', 0)
                                  for item in (data or {}).get('prefill', [])]
                decode_values = [item.get('throughput_tok_s', 0)
                                 for item in (data or {}).get('decode', [])]
                
                results.append({
                    'gpu': gpu,
                    'tp': tp,
                    'prefill_avg': (sum(prefill_values) / len(prefill_values)
                                    if prefill_values else 0),
                    'decode_avg': (sum(decode_values) / len(decode_values)
                                   if decode_values else 0),
                    'combined_score': float(row['综合评分']) if row['综合评分'] else 0,
                    'combined_score_status': 'legacy_not_for_ranking',
                    'comparison_status': 'provisional',
                    'price': float(row['价格']) if row.get('价格') and row['价格'] else None,
                    'price_performance': float(row['性价比']) if row.get('性价比') and row['性价比'] else None,
                    'case_scores': case_scores,
                    'color': '#76b900' if 'L20' in gpu else '#ff6b6b' if '昇腾' in gpu else '#4ecdc4'
                })
        return jsonify(results)
    
    # fallback: 读取所有 JSON 报告
    json_files = glob.glob(os.path.join(REPORT_DIR, 'benchmark_*.json'))
    results = []
    for f in json_files:
        data = _load_comparable_report(f)
        meta = data.get('meta', {})
        summary = data.get('summary', {})
        gpu = meta.get('gpu_type', 'Unknown')
        tp = meta.get('tp', 1)
        case_scores = {}
        for item in data.get('prefill', []):
            case_scores[item['case']] = item.get('throughput_tok_s', 0)
        for item in data.get('decode', []):
            case_scores[item['case']] = item.get('throughput_tok_s', 0)
        prefill_values = [item.get('throughput_tok_s', 0)
                          for item in data.get('prefill', [])]
        decode_values = [item.get('throughput_tok_s', 0)
                         for item in data.get('decode', [])]
        results.append({
            'gpu': gpu,
            'tp': tp,
            'prefill_avg': (sum(prefill_values) / len(prefill_values)
                            if prefill_values else 0),
            'decode_avg': (sum(decode_values) / len(decode_values)
                           if decode_values else 0),
            'combined_score': summary.get('combined_score', 0),
            'combined_score_status': 'legacy_not_for_ranking',
            'comparison_status': 'provisional',
            'price': meta.get('price'),
            'case_scores': case_scores,
            'color': '#76b900' if 'L20' in gpu else '#ff6b6b' if '昇腾' in gpu else '#4ecdc4'
        })
    return jsonify(results)


@app.route('/api/benchmark/spec')
def get_benchmark_spec():
    """Return the versioned model workload and fairness contract."""
    return jsonify(load_benchmark_spec(BENCHMARK_SPEC_FILE))


@app.route('/api/benchmark/options')
def get_benchmark_options():
    options = set()
    for path in glob.glob(os.path.join(REPORT_DIR, 'benchmark_*_TP*_*.json')):
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                meta = json.load(handle).get('meta', {})
            if meta.get('gpu_type') and meta.get('tp'):
                options.add((str(meta['gpu_type']), int(meta['tp'])))
        except (OSError, ValueError, TypeError):
            continue
    return jsonify([
        {'gpu': gpu, 'tp': tp, 'id': f'{gpu}::TP{tp}'}
        for gpu, tp in sorted(options)
    ])


@app.route('/api/benchmark/relative')
def get_relative_benchmark():
    candidate_gpu = request.args.get('candidate')
    reference_gpu = request.args.get('reference')
    tp = request.args.get('tp', type=int)
    if not candidate_gpu or not reference_gpu or tp is None:
        return jsonify({'error': 'candidate, reference and tp are required'}), 400
    candidate_path = _latest_report(candidate_gpu, tp)
    reference_path = _latest_report(reference_gpu, tp)
    if not candidate_path or not reference_path:
        return jsonify({'error': 'matching report not found'}), 404
    candidate = _load_comparable_report(candidate_path)
    reference = _load_comparable_report(reference_path)
    truth_ratios, truth_path = _ground_truth_ratios(
        candidate_gpu, reference_gpu, tp)
    result = compare_reports(candidate, reference, truth_ratios)
    result['sources'] = {
        'candidate': os.path.basename(candidate_path),
        'reference': os.path.basename(reference_path),
        'ground_truth': (os.path.relpath(truth_path, PROJECT_ROOT)
                         if truth_path else None),
    }
    if any(report.get('meta', {}).get('report_provenance') ==
           'legacy_v0_adapted_in_memory' for report in (candidate, reference)):
        result['compatibility']['warnings'].append(
            'legacy report throughput was corrected in memory; regenerate reports for protocol-native artifacts')
    return jsonify(result)


@app.route('/api/report/<gpu_type>')
def get_report(gpu_type):
    """获取某个 GPU 的详细报告"""
    pattern = os.path.join(REPORT_DIR, f'benchmark_{gpu_type}_*.json')
    files = glob.glob(pattern)
    if not files:
        return jsonify({'error': 'Not found'}), 404
    with open(files[-1], 'r') as f:
        return jsonify(json.load(f))


@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """获取所有任务列表"""
    limit = request.args.get('limit', 50, type=int)
    tasks = task_manager.get_tasks(limit)
    return jsonify(tasks)


@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task(task_id):
    task = _get_task_from_file(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task)


@app.route('/api/tasks/<task_id>/logs', methods=['GET'])
def get_task_logs(task_id):
    since = request.args.get('since', 0, type=int)
    logs = _get_task_logs_from_file(task_id, since)
    return jsonify({'logs': logs, 'count': len(logs)})


@app.route('/api/profile/start', methods=['POST'])
def start_profile():
    import sys
    from datetime import datetime
    import uuid
    import json
    import tempfile
    from core.task_manager import task_manager

    data = request.get_json()
    
    # ========== 1. 基础字段校验 ==========
    required = ['host', 'user', 'gpu_type']
    for key in required:
        if key not in data or not data[key]:
            return jsonify({'error': f'Missing or empty required field: {key}'}), 400
    
    # ========== 2. TP 校验 ==========
    tps = data.get('tps', [])
    if not tps:
        if data.get('tp'):
            tps = [data.get('tp')]
        else:
            return jsonify({'error': '至少选择一个 TP 配置'}), 400
    
    try:
        tps = sorted(set(int(t) for t in tps))
    except (ValueError, TypeError):
        return jsonify({'error': 'TP 值必须为数字'}), 400
    
    valid_tps = [1, 2, 4, 8, 16]
    for tp in tps:
        if tp not in valid_tps:
            return jsonify({'error': f'TP={tp} 不在支持的范围内: {valid_tps}'}), 400
    
    # ========== 3. 其他字段处理 ==========
    gpu_type = data['gpu_type'].strip()
    if not gpu_type:
        return jsonify({'error': 'GPU 型号不能为空'}), 400
    
    # ========== 新增：获取跳过层时间选项 ==========
    skip_layer_bench = data.get('skip_layer_bench', True)  # 默认跳过
    
    # ========== 4. 生成真实任务 ID ==========
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    # ========== 5. 创建任务对象 ==========
    task = {
        'id': task_id,
        'type': 'profile_batch',
        'status': 'pending',
        'params': {
            'gpu_type': gpu_type,
            'host': data['host'].strip(),
            'user': data['user'].strip(),
            'password': data.get('password'),
            'port': data.get('port', 22),
            'price': data.get('price'),
            'tps': tps,
            'total': len(tps),
            'skip_layer_bench': skip_layer_bench  # 新增
        },
        'created_at': datetime.now().isoformat(),
        'started_at': None,
        'completed_at': None,
        'logs': [],
        'result': None,
        'error': None
    }
    
    # ========== 6. 原子写入 tasks.json ==========
    try:
        if os.path.exists(TASKS_FILE):
            with open(TASKS_FILE, 'r') as f:
                tasks = json.load(f)
        else:
            tasks = []
        tasks.append(task)
        
        fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(TASKS_FILE), suffix='.json')
        with os.fdopen(fd, 'w') as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, TASKS_FILE)
        
        sys.stderr.write(f"[任务创建] task_id={task_id}\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[任务创建失败] {e}\n")
        sys.stderr.flush()
        return jsonify({'error': f'任务创建失败: {str(e)}'}), 500
    
    # ========== 7. 关键修复：同步更新 task_manager 内存缓存 ==========
    with task_manager._lock:
        task_manager.tasks.append(task)  # 确保 _run_task 能找到任务
    
    # ========== 8. 异步执行任务 ==========
    def background_executor():
        import sys
        sys.stderr.write("🔥🔥🔥 background_executor 线程已启动\n")
        sys.stderr.flush()
        try:
            task_manager._run_task(task_id)
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write(f"[后台执行失败] {e}\n")
            sys.stderr.flush()

    import sys
    sys.stderr.write(f"准备启动线程，task_id={task_id}\n")
    sys.stderr.flush()
    thread = threading.Thread(target=background_executor, daemon=True)
    thread.start()
    sys.stderr.write(f"✅ 线程已启动，task_id={task_id}\n")
    sys.stderr.flush()
    
    # ========== 9. 立即返回 ==========
    return jsonify({
        'task_id': task_id,
        'status': 'pending',
        'message': f'已提交 {len(tps)} 个 TP 配置的采集任务' +
                   (' (跳过层时间)' if skip_layer_bench else ' (包含层时间)')
    })

@app.route('/api/benchmark/run', methods=['POST'])
def run_benchmark():
    """仅运行评分引擎（不采集新数据）"""
    data = request.get_json()
    
    if 'gpu_type' not in data:
        return jsonify({'error': 'Missing gpu_type'}), 400
    
    task_id = task_manager.submit_task('benchmark', {
        'gpu_type': data['gpu_type'],
        'tp': data.get('tp', 1),
        'price': data.get('price')
    })
    
    return jsonify({
        'task_id': task_id,
        'status': 'pending',
        'message': f'Benchmark task {task_id} submitted'
    })


@app.route('/api/gpu-types')
def get_gpu_types():
    """获取所有已采集的 GPU 类型"""
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    gpu_dirs = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item not in ['benchmark_reports', 'ground_truth_pd', 'trace']:
            # 检查是否包含 Profiling 数据
            has_data = any(f.endswith('.json') for f in os.listdir(item_path))
            if has_data:
                gpu_dirs.append(item)
    return jsonify(gpu_dirs)


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Benchmark 平台启动")
    print("=" * 60)
    print(f"📊 访问: http://localhost:5000")
    print(f"📁 报告目录: {REPORT_DIR}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
