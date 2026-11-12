"""Phase 1: False-sharing stride sweep + Karp-Flatt process-scaling diagnostics."""

import ctypes
import multiprocessing as mp
import subprocess
import time
from multiprocessing import shared_memory

from harness import emit_verify_tag, stats_from_samples

STUDENT_SEED = 230103269
SHM_BYTES = 4096 
N_SLOTS = SHM_BYTES // 8          # c_uint64 = 8 bytes -> 4096/8 = 512 slots
M_TOTAL = 16_000_000             # 1.6e7 total read-modify-write steps
STRIDES = (1, 2, 4, 8, 16, 32)
WARMUP_RUNS = 2
TIMED_RUNS = 7



def fs_worker(shm_name: str, slot_idx: int, iters: int, barrier) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * N_SLOTS).from_buffer(shm.buf)

    barrier.wait()

    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005
    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc         

    del arr                         
    shm.close()


def physical_cores() -> int:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                             capture_output=True, text=True, check=True)
        return int(out.stdout.strip())
    except Exception:
        return mp.cpu_count()


def one_trial(shm_name: str, p: int, stride: int) -> float:
    """Spawn p workers, time only the contended loop, return wall ms."""
    iters = M_TOTAL // p
    barrier = mp.Barrier(p + 1)
    procs = [
        mp.Process(target=fs_worker, args=(shm_name, w * stride, iters, barrier))
        for w in range(p)
    ]
    for proc in procs:
        proc.start()

    # A dead worker never reaches the barrier, so bound the wait instead of
    # blocking the parent forever.
    try:
        barrier.wait(timeout=60)
    except Exception:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join()
        codes = [proc.exitcode for proc in procs]
        raise RuntimeError(f"workers failed to reach barrier; exitcodes={codes}")

    t0 = time.perf_counter_ns()
    for proc in procs:
        proc.join()
    t1 = time.perf_counter_ns()

    for proc in procs:
        if proc.exitcode != 0:
            raise RuntimeError(f"worker exited with {proc.exitcode}")
    return (t1 - t0) / 1e6


def bench(shm_name: str, p: int, stride: int):
    for _ in range(WARMUP_RUNS):
        one_trial(shm_name, p, stride)
    samples = [one_trial(shm_name, p, stride) for _ in range(TIMED_RUNS)]
    return stats_from_samples(samples)


def karp_flatt(speedup: float, p: int) -> float:
    return ((1.0 / speedup) - (1.0 / p)) / (1.0 - (1.0 / p))


def stride_sweep(shm_name: str, p_phys: int) -> dict:
    print(f"--- 1. Stride sweep (p = P_phys = {p_phys}) ---")
    print(f"{'stride':>6} {'bytes':>6} {'median_ms':>10} {'IQR_ms':>8} {'Mops/s':>9} {'vs stride=1':>12}")
    results = {}
    base = None
    for stride in STRIDES:
        if (p_phys - 1) * stride >= N_SLOTS:
            print(f"{stride:>6}  skipped: would exceed {N_SLOTS} slots")
            continue
        st = bench(shm_name, p_phys, stride)
        mops = M_TOTAL / 1e6 / (st.median_ms / 1e3)
        if base is None:
            base = st.median_ms
        results[stride] = st
        print(f"{stride:>6} {stride * 8:>6} {st.median_ms:>10.2f} {st.iqr_ms:>8.2f} "
              f"{mops:>9.2f} {base / st.median_ms:>11.2f}x")
    print()
    return results


def process_scaling(shm_name: str, p_phys: int, p_log: int) -> dict:
    p_list = []
    q = 1
    while q <= p_log:
        p_list.append(q)
        q *= 2
    if p_log not in p_list:
        p_list.append(p_log)

    out = {}
    for stride in (1, 16):
        print(f"--- 2. Process scaling, stride = {stride} ({stride * 8} B spacing) ---")
        print(f"{'p':>3} {'median_ms':>10} {'S_p':>7} {'E_p':>7} {'e(p)':>8}")
        t1 = None
        rows = {}
        for p in p_list:
            if (p - 1) * stride >= N_SLOTS:
                continue
            st = bench(shm_name, p, stride)
            if t1 is None:
                t1 = st.median_ms
            sp = t1 / st.median_ms
            ep = sp / p
            e = karp_flatt(sp, p) if p >= 2 else float("nan")
            rows[p] = (st, sp, ep, e)
            e_txt = "     n/a" if p < 2 else f"{e:>8.4f}"
            print(f"{p:>3} {st.median_ms:>10.2f} {sp:>7.3f} {ep:>7.3f} {e_txt}")
        out[stride] = rows
        print()
    return out


def main() -> None:
    p_phys = physical_cores()
    p_log = mp.cpu_count()
    print("=== Phase 1: False Sharing Stride Sweep & Karp-Flatt ===")
    print(f"P_phys = {p_phys}, P_log = {p_log}, M = {M_TOTAL:,}, "
          f"shm = {SHM_BYTES} B ({N_SLOTS} uint64 slots)")
    print(f"{WARMUP_RUNS} warmup + {TIMED_RUNS} timed runs per point\n")

    shm = shared_memory.SharedMemory(create=True, size=SHM_BYTES)
    try:
        print(f"requested {SHM_BYTES} B, mapped {len(shm.buf)} B (page-rounded)\n")
        shm.buf[:SHM_BYTES] = bytes(SHM_BYTES)
        sweep = stride_sweep(shm.name, p_phys)
        scaling = process_scaling(shm.name, p_phys, p_log)
    finally:
        shm.close()
        shm.unlink()

    m1 = sweep[1].median_ms
    m16 = sweep[16].median_ms
    print(f"median_ms(stride=1)  @ p={p_phys}: {m1:.4f}")
    print(f"median_ms(stride=16) @ p={p_phys}: {m16:.4f}")
    print(f"contention relief factor: {m1 / m16:.2f}x\n")
    print(emit_verify_tag("PHASE1", STUDENT_SEED, m1, m16))


if __name__ == "__main__":
    main()
