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


def emit_verify_tag(task_id: str , student_seed: int ,
    primary_metric: float , secondary_metric: float) -> str:
    hw = mp.cpu_count ()
    payload = (
        f"{task_id }|{ student_seed }|{hw}|"
        f"{primary_metric :.4f}|{ secondary_metric :.4f}"
    )
    sig = fnv1a_hash(payload)
    return f"VERIFY_HASH [{ task_id }]: {sig :016X} (P_log ={hw})"