import torch
import torchvision
import time
import requests
import numpy as np
import json
import statistics
import concurrent.futures

MODEL_PATH_INFO = "ResNet50, opset 17, dynamic batch axis"
SERVER_URL = "http://localhost:8080"

def benchmark_pytorch(n=200, batch_size=1):
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.eval()
    torch.set_num_threads(4)  # match ONNX Runtime's thread count for fair comparison

    dummy = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy)

    latencies = []
    with torch.no_grad():
        for _ in range(n):
            start = time.perf_counter()
            _ = model(dummy)
            latencies.append((time.perf_counter() - start) * 1000)

    return latencies

def benchmark_server_single(n=200):
    dummy = np.random.randn(3, 224, 224).astype(np.float32)
    payload = {"data": dummy.flatten().tolist(), "shape": [3, 224, 224]}

    for _ in range(20):
        requests.post(f"{SERVER_URL}/infer", json=payload)

    latencies = []
    for _ in range(n):
        start = time.perf_counter()
        r = requests.post(f"{SERVER_URL}/infer", json=payload)
        r.raise_for_status()
        latencies.append((time.perf_counter() - start) * 1000)

    return latencies

def benchmark_server_throughput(num_requests=500, num_workers=16):
    dummy = np.random.randn(3, 224, 224).astype(np.float32)
    payload = {"data": dummy.flatten().tolist(), "shape": [3, 224, 224]}

    def send(_):
        return requests.post(f"{SERVER_URL}/infer", json=payload)

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(send, i) for i in range(num_requests)]
        results = [f.result() for f in futures]
    elapsed = time.perf_counter() - start

    resp = requests.get(f"{SERVER_URL}/metrics").json()
    return {
        "throughput_rps": num_requests / elapsed,
        "elapsed_s": elapsed,
        "avg_batch_size": resp["avg_batch_size"],
    }

def summarize(name, latencies):
    print(f"\n--- {name} ---")
    print(f"  mean:   {statistics.mean(latencies):.2f} ms")
    print(f"  median: {statistics.median(latencies):.2f} ms")
    print(f"  p95:    {sorted(latencies)[int(len(latencies)*0.95)]:.2f} ms")
    print(f"  p99:    {sorted(latencies)[int(len(latencies)*0.99)]:.2f} ms")
    print(f"  min:    {min(latencies):.2f} ms")
    print(f"  max:    {max(latencies):.2f} ms")

if __name__ == "__main__":
    print(f"Model: {MODEL_PATH_INFO}")

    pt_lat = benchmark_pytorch(n=200)
    summarize("PyTorch CPU baseline", pt_lat)

    onnx_lat = benchmark_server_single(n=200)
    summarize("Helios (single request, ONNX Runtime C++)", onnx_lat)

    throughput = benchmark_server_throughput(num_requests=500, num_workers=16)
    print("\n--- Concurrent throughput (16 threads, 500 requests) ---")
    print(json.dumps(throughput, indent=2))

    speedup = statistics.mean(pt_lat) / statistics.mean(onnx_lat)
    print(f"\nSpeedup vs PyTorch: {speedup:.2f}x")
