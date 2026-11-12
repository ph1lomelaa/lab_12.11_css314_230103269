from dataclasses import dataclass
import multiprocessing as mp
import time
from typing import Callable

@dataclass
class BenchStats:
    min_ms: float
    median_ms: float
    iqr_ms: float
    max_ms: float

def fnv1a_hash(s: str) -> int:
    h = 14695981039346656037
    for b in s.encode("utf -8"):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h



def run_benchmark(fn: Callable [[], None], warmup_runs: int = 3, timed_runs: int = 9) -> BenchStats:
    for _ in range(warmup_runs):
        fn()
        samples = []
    for _ in range(timed_runs):
        t0 = time.perf_counter_ns ()
        fn()
        t1 = time.perf_counter_ns ()
        samples.append ((t1 - t0) / 1e6)
    samples.sort()
    mid = timed_runs // 2
    med = samples[mid] if timed_runs % 2 == 1 else 0.5 * (samples[mid - 1] + samples[mid])
    q25 = samples[timed_runs // 4]
    q75 = samples [(3 * timed_runs) // 4]
    return BenchStats(samples [0], med , q75 - q25 , samples [-1])

def emit_verify_tag(task_id: str , student_seed: int ,
    primary_metric: float , secondary_metric: float) -> str:
    hw = mp.cpu_count ()
    payload = (
        f"{task_id }|{ student_seed }|{hw}|"
        f"{primary_metric :.4f}|{ secondary_metric :.4f}"
    )
    sig = fnv1a_hash(payload)
    return f"VERIFY_HASH [{ task_id }]: {sig :016X} (P_log ={hw})"