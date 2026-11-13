import math
import multiprocessing as mp
import subprocess
import time

from harness import emit_verify_tag, stats_from_samples

STUDENT_SEED = 230103269
N = 8000
CHUNKS = (1, 8, 64, 512)
WARMUP_RUNS = 1
TIMED_RUNS = 3


def physical_cores() -> int:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                             capture_output=True, text=True, check=True)
        return int(out.stdout.strip())
    except Exception:
        return mp.cpu_count()


def work(i):
    x = i + 1.0
    for _ in range(1 + i * i // 1600):
        x = math.sqrt(abs(math.sin(x) * math.cos(x)) + 1.0)
    return x


def worker(barrier, t, p, mode, chunk, counter, times):
    barrier.wait()
    t0 = time.perf_counter_ns()
    if mode == "block":
        step = N // p
        for i in range(t * step, N if t == p - 1 else (t + 1) * step):
            work(i)
    elif mode == "cyclic":
        for i in range(t, N, p):
            work(i)
    else:
        while True:
            with counter.get_lock():
                start = counter.value
                counter.value += chunk
            if start >= N:
                break
            for i in range(start, min(start + chunk, N)):
                work(i)
    times[t] = (time.perf_counter_ns() - t0) / 1e6


def trial(p, mode, chunk):
    counter = mp.Value("i", 0)
    times = mp.Array("d", p)
    barrier = mp.Barrier(p + 1)
    procs = [mp.Process(target=worker, args=(barrier, t, p, mode, chunk, counter, times))
             for t in range(p)]
    for pr in procs:
        pr.start()
    barrier.wait()
    t0 = time.perf_counter_ns()
    for pr in procs:
        pr.join()
    wall = (time.perf_counter_ns() - t0) / 1e6
    tw = list(times)
    return wall, max(tw) / (sum(tw) / p)


def bench(p, mode, chunk=0):
    walls, lams = [], []
    for r in range(WARMUP_RUNS + TIMED_RUNS):
        w, lam = trial(p, mode, chunk)
        if r >= WARMUP_RUNS:
            walls.append(w)
            lams.append(lam)
    return stats_from_samples(walls), stats_from_samples(lams).median_ms


def main():
    p = physical_cores()
    print("=== Phase 4: Skewed Workloads & Dynamic Chunk Queue ===")
    print(f"P_phys = {p}, N = {N}, K(i) = 1 + floor((i/40)^2), "
          f"{WARMUP_RUNS} warmup + {TIMED_RUNS} timed runs\n")
    print(f"{'strategy':>16} {'T_wall_ms':>10} {'IQR_ms':>8} {'lambda':>7}")
    rows = {}
    for name, mode, c in [("block static", "block", 0), ("cyclic static", "cyclic", 0)] + \
            [(f"dynamic C={c}", "dynamic", c) for c in CHUNKS]:
        st, lam = bench(p, mode, c)
        rows[name] = st.median_ms
        print(f"{name:>16} {st.median_ms:>10.1f} {st.iqr_ms:>8.1f} {lam:>7.3f}")
    c_best = min(CHUNKS, key=lambda c: rows[f"dynamic C={c}"])
    t_block = rows["block static"]
    t_dyn = rows[f"dynamic C={c_best}"]
    print(f"\nC* = {c_best}, T_wall(block) = {t_block:.4f} ms, T_wall(dynamic C*) = {t_dyn:.4f} ms, "
          f"gain = {t_block / t_dyn:.2f}x\n")
    print(emit_verify_tag("PHASE4", STUDENT_SEED, t_block, t_dyn))


if __name__ == "__main__":
    main()
