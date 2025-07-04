import time

import matplotlib.pyplot as plt
import psutil
import torch


def monitor_memory_usage():
    """Monitor both GPU and RAM usage"""
    memory_stats = {}

    # GPU Memory
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        memory_stats["gpu"] = {
            "allocated": gpu_allocated,
            "reserved": gpu_reserved,
            "total": gpu_total,
            "free": gpu_total - gpu_reserved,
        }

    # RAM Memory
    ram = psutil.virtual_memory()
    memory_stats["ram"] = {
        "used": ram.used / 1024**3,
        "total": ram.total / 1024**3,
        "free": ram.free / 1024**3,
        "percent": ram.percent,
    }

    return memory_stats


def print_memory_breakdown(label=""):
    """Print detailed memory breakdown"""
    stats = monitor_memory_usage()

    print(f"\n=== MEMORY BREAKDOWN {label} ===")
    print("GPU Memory:")
    if "gpu" in stats:
        print(f"  Allocated: {stats['gpu']['allocated']:.2f} GB")
        print(f"  Reserved:  {stats['gpu']['reserved']:.2f} GB")
        print(f"  Free:      {stats['gpu']['free']:.2f} GB")
        print(f"  Total:     {stats['gpu']['total']:.2f} GB")
        print(f"  Usage:     {stats['gpu']['allocated'] / stats['gpu']['total'] * 100:.1f}%")

    print("RAM Memory:")
    print(f"  Used:      {stats['ram']['used']:.2f} GB")
    print(f"  Free:      {stats['ram']['free']:.2f} GB")
    print(f"  Total:     {stats['ram']['total']:.2f} GB")
    print(f"  Usage:     {stats['ram']['percent']:.1f}%")


class MemoryTracker:
    """Track memory usage over time"""

    def __init__(self):
        self.timeline = []
        self.gpu_memory = []
        self.ram_memory = []
        self.labels = []

    def checkpoint(self, label):
        stats = monitor_memory_usage()
        self.timeline.append(time.time())
        self.labels.append(label)

        if "gpu" in stats:
            self.gpu_memory.append(stats["gpu"]["allocated"])
        else:
            self.gpu_memory.append(0)

        self.ram_memory.append(stats["ram"]["used"])

        print(f"[{label}] GPU: {self.gpu_memory[-1]:.2f}GB | RAM: {self.ram_memory[-1]:.2f}GB")

    def plot_memory_usage(self):
        """Plot memory usage over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # GPU Memory Plot
        ax1.plot(range(len(self.gpu_memory)), self.gpu_memory, "b-", marker="o")
        ax1.set_title("GPU Memory Usage Over Time")
        ax1.set_ylabel("GPU Memory (GB)")
        ax1.set_xticks(range(len(self.labels)))
        ax1.set_xticklabels(self.labels, rotation=45, ha="right")
        ax1.grid(True)

        # RAM Memory Plot
        ax2.plot(range(len(self.ram_memory)), self.ram_memory, "r-", marker="s")
        ax2.set_title("RAM Usage Over Time")
        ax2.set_ylabel("RAM (GB)")
        ax2.set_xlabel("Checkpoint")
        ax2.set_xticks(range(len(self.labels)))
        ax2.set_xticklabels(self.labels, rotation=45, ha="right")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("memory_usage.png", dpi=150, bbox_inches="tight")
        plt.show()

        # Print summary
        print("\n=== MEMORY USAGE SUMMARY ===")
        for i, label in enumerate(self.labels):
            print(f"{label:20s}: GPU {self.gpu_memory[i]:6.2f}GB | RAM {self.ram_memory[i]:6.2f}GB")
