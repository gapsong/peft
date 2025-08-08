# Benchmark verschiedener Ansätze:
import time

import torch
from torch.nn.functional import F


def benchmark_pooling_methods(batch_size=32, seq_len=512, hidden_size=4096, group_size=32):
    x = torch.randn(batch_size * seq_len, hidden_size, device="cuda")
    pooled_dim = hidden_size // group_size

    # Method 1: F.avg_pool1d
    def method1():
        x_unsqueezed = x.unsqueeze(1)
        pooled = F.avg_pool1d(x_unsqueezed, kernel_size=group_size, stride=group_size)
        return pooled.squeeze(1) * pooled_dim

    # Method 2: view + mean
    def method2():
        return x.view(-1, pooled_dim, group_size).mean(dim=2) * pooled_dim

    # Method 3: Original (multiple reshapes)
    def method3():
        x_2d = x.reshape(-1, hidden_size)
        x_for_pooling = x_2d.unsqueeze(1)
        x_pooled = F.avg_pool1d(x_for_pooling, kernel_size=group_size, stride=group_size)
        return x_pooled.squeeze(1) * pooled_dim

    # Benchmark
    methods = [("F.avg_pool1d", method1), ("view+mean", method2), ("Original", method3)]

    for name, method in methods:
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            result = method()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"{name:15s}: {elapsed:.4f}s ({result.shape})")


# benchmark_pooling_methods()
