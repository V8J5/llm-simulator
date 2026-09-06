# scripts/generate_synthetic_data.py
import random
import json
import os

def generate_mock_requests(num_requests=100):
    """
    使用泊松分布模拟请求到达，正态分布模拟Token长度
    """
    data = []
    current_time = 0
    
    for _ in range(num_requests):
        # 模拟间隔 (秒) - 假设每秒约 2-5 个请求
        interval = random.expovariate(3.0) 
        current_time += interval * 1000 # 转毫秒
        
        # 模拟 Prompt 长度 (均值 500 tokens)
        prompt_len = max(10, int(random.gauss(500, 200)))
        # 模拟 Output 长度 (均值 200 tokens)
        output_len = max(10, int(random.gauss(200, 100)))
        
        data.append({
            "request_id": f"req_{_}",
            "arrival_time_ms": round(current_time, 2),
            "prompt_tokens": prompt_len,
            "output_tokens": output_len
        })
        
    return data

if __name__ == "__main__":
    mock_data = generate_mock_requests()
    
    # 动态获取项目根目录（即 scripts 的上一级）
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    save_path = os.path.join(project_root, 'data', 'mock_requests.json')
    
    with open(save_path, 'w') as f:
        json.dump(mock_data, f, indent=2)
    print(f"✅ 已生成 {len(mock_data)} 条模拟请求数据至: {save_path}")