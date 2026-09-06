#!/usr/bin/env python3
"""
测量真实模型的 Prefill 和 Decode 分别耗时
"""
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

MODEL_PATH = "/data/home/public/weight/Qwen3-32B"
DEVICE = "cuda"

def measure_prefill(model, tokenizer, prompt_text, warmup=True):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    
    # 预热：第一次跑不计时（消除冷启动）
    if warmup:
        with torch.no_grad():
            _ = model(input_ids=inputs.input_ids, use_cache=True)
        torch.cuda.synchronize()
    
    # 正式测量
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, use_cache=True)
        _ = outputs.logits
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000

def measure_decode(model, tokenizer, prompt_text, num_tokens=128):
    """只测 Decode（逐 token 生成，不测采样）"""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    
    # 先跑 Prefill
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values
    
    # 开始 Decode
    input_ids = inputs.input_ids
    next_token = input_ids[:, -1:]
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(num_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed

def main():
    print("🔄 加载真实模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("✅ 模型加载完成")
    
    cases = [
        {"name": "P_S", "prompt": "Hello " * 100, "tokens": 128},
        {"name": "P_M", "prompt": "Hello " * 200, "tokens": 256},
        {"name": "P_L", "prompt": "Hello " * 500, "tokens": 256},
        {"name": "D_S", "prompt": "Hello " * 100, "tokens": 128},
        {"name": "D_M", "prompt": "Hello " * 200, "tokens": 256},
    ]
    
    print("\n📊 Prefill 耗时:")
    for case in cases[:3]:
        # 先预热（不计时），再正式测量
        _ = measure_prefill(model, tokenizer, case["prompt"], warmup=True)
        elapsed = measure_prefill(model, tokenizer, case["prompt"], warmup=False)
        print(f"  {case['name']}: {elapsed:.2f} ms")
    
    print("\n📊 Decode 耗时 (per token):")
    for case in cases[3:]:  # Decode cases
        elapsed = measure_decode(model, tokenizer, case["prompt"], case["tokens"])
        per_token = elapsed / case["tokens"]
        print(f"  {case['name']}: {elapsed:.2f} ms total, {per_token:.2f} ms/token")

if __name__ == "__main__":
    main()