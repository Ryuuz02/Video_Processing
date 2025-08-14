import sys
import subprocess
import re
import csv
from datetime import datetime

# Path to Python in your current venv
venv_python = sys.executable

# Output CSV file
csv_file = "benchmark_results.csv"

modes = ["cpu", "parallel", "gpu", "hybrid"]
results = {}

for mode in modes:
    print(f"\n=== Running {mode.upper()} benchmark ===")
    output = subprocess.check_output(
        [venv_python, "main.py", "--mode", mode], text=True
    )

    print(output)

    # Extract runtime from benchmark output
    match = re.search(r"\[BENCHMARK\] run_detection.*took ([\d.]+) seconds", output)
    if match:
        results[mode] = float(match.group(1))

# Create header if CSV does not exist
try:
    with open(csv_file, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "cpu", "parallel", "gpu", "gpu_speedup_vs_cpu", "gpu_speedup_vs_parallel"])
except FileExistsError:
    pass  # File already exists

# Append new results
with open(csv_file, "a", newline="") as f:
    writer = csv.writer(f)
    timestamp = datetime.now().isoformat(timespec='seconds')

    gpu_speedup_cpu = results["cpu"] / results["gpu"] if "cpu" in results and "gpu" in results else None
    gpu_speedup_parallel = results["parallel"] / results["gpu"] if "parallel" in results and "gpu" in results else None

    writer.writerow([
        timestamp,
        results.get("cpu"),
        results.get("parallel"),
        results.get("gpu"),
        gpu_speedup_cpu,
        gpu_speedup_parallel
    ])

# Print summary
print("\n=== Summary ===")
for mode, time in results.items():
    print(f"{mode:8} : {time:.2f} sec")

if gpu_speedup_cpu:
    print(f"GPU Speedup vs CPU: {gpu_speedup_cpu:.2f}x")
if gpu_speedup_parallel:
    print(f"GPU Speedup vs Parallel CPU: {gpu_speedup_parallel:.2f}x")

print(f"\nResults appended to {csv_file}")
