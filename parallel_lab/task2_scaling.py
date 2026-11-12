"""Phase 2: Strong scaling (Amdahl) vs Weak scaling (Gustafson)."""

import math
import multiprocessing as mp
import subprocess
import time

from harness import emit_verify_tag, stats_from_samples

STUDENT_SEED = 230103269
N_STRONG = 1_200_000             # fixed problem size for strong scaling
N0_WEAK = 200_000                # per-worker load for weak scaling
SERIAL_FRACTION = 0.05           # Stage A covers the first 5% of elements
WARMUP_RUNS = 1
TIMED_RUNS = 3


def stage_a_element(x: float) -> float:
    """10 sin/sqrt operations on one element."""
    a = math.sin(x)
    b = math.sqrt(x)
    c = math.sin(a + b)
    d = math.sqrt(b + 1.0)
    e = math.sin(c * d)
    return a + b + c + d + e + math.sqrt(e * e + 1.0) + math.sin(d) + \
        math.sqrt(a * a + 1.0) + math.sin(b) + math.sqrt(c * c + 1.0)


def stage_b_element(x: float) -> float:
    """30 fused multiply-add / trigonometric operations on one element."""
    a = math.sin(x)
    b = math.cos(x)
    c = math.sqrt(x)
    t = a * b + c
    t = t * a + b
    t = t * c + a
    t = math.sin(t) * b + c
    t = math.cos(t) * c + a
    t = math.sqrt(abs(t) + 1.0) * a + b
    t = t * b + c
    t = math.sin(t) * a + b
    t = t * c + a
    t = math.cos(t) * b + c
    return t


def stage_a_serial(n_elements: int) -> float:
    """Runs in the parent process, single-threaded: the Amdahl serial region."""
    acc = 0.0
    for i in range(n_elements):
        acc += stage_a_element((i % 1000) + 1.0)
    return acc


def stage_b_chunk(bounds) -> float:
    """Pool worker: no shared writes, returns a private partial sum."""
    start, end = bounds
    acc = 0.0
    for i in range(start, end):
        acc += stage_b_element((i % 1000) + 1.0)
    return acc


def split_chunks(n: int, p: int):
    step = n // p
    bounds = [(w * step, (w + 1) * step) for w in range(p)]
    bounds[-1] = (bounds[-1][0], n)      # last worker absorbs the remainder
    return bounds


def one_trial(pool, n: int, p: int):
    """Returns (total_ms, stage_a_ms) for one Stage A -> Stage B pipeline run."""
    n_serial = int(n * SERIAL_FRACTION)
    chunks = split_chunks(n, p)

    t0 = time.perf_counter_ns()
    stage_a_serial(n_serial)
    t1 = time.perf_counter_ns()
    pool.map(stage_b_chunk, chunks)
    t2 = time.perf_counter_ns()

    return (t2 - t0) / 1e6, (t1 - t0) / 1e6


def bench(n: int, p: int):
    """Pool is created OUTSIDE the timed loop so spawn cost is never measured."""
    with mp.Pool(processes=p) as pool:
        pool.map(stage_b_chunk, split_chunks(1000, p))   # force worker start-up

        for _ in range(WARMUP_RUNS):
            one_trial(pool, n, p)

        totals, stage_as = [], []
        for _ in range(TIMED_RUNS):
            total_ms, a_ms = one_trial(pool, n, p)
            totals.append(total_ms)
            stage_as.append(a_ms)

    return stats_from_samples(totals), stats_from_samples(stage_as)


def sysctl_int(key: str, fallback: int) -> int:
    try:
        out = subprocess.run(["sysctl", "-n", key],
                             capture_output=True, text=True, check=True)
        return int(out.stdout.strip())
    except Exception:
        return fallback


def physical_cores() -> int:
    return sysctl_int("hw.physicalcpu", mp.cpu_count())


def performance_cores() -> int:
    """Count of the fast core cluster (perflevel0); equals P_phys if homogeneous."""
    return sysctl_int("hw.perflevel0.physicalcpu", physical_cores())


def p_ladder(p_fast: int, p_log: int):
    """Powers of two up to P_log, plus the points that bracket the core boundary."""
    ladder = []
    q = 1
    while q <= p_log:
        ladder.append(q)
        q *= 2
    # P-core -> E-core boundary and the oversubscription point
    for extra in (p_fast, p_fast + 1, p_log, p_log + 1):
        if extra not in ladder:
            ladder.append(extra)
    return sorted(ladder)


def strong_scaling(p_list):
    print(f"--- 1. Strong scaling (N = {N_STRONG:,} fixed) ---")
    print(f"{'p':>3} {'T_ms':>10} {'StageA_ms':>10} {'S_p':>7} {'E_p':>7}")
    t1 = None
    s_serial = None
    rows = {}
    for p in p_list:
        total, stage_a = bench(N_STRONG, p)
        if t1 is None:
            t1 = total.median_ms
            s_serial = stage_a.median_ms / t1       # Amdahl serial fraction s
        sp = t1 / total.median_ms
        rows[p] = (total, stage_a, sp, sp / p)
        print(f"{p:>3} {total.median_ms:>10.1f} {stage_a.median_ms:>10.1f} "
              f"{sp:>7.3f} {sp / p:>7.3f}")
    print(f"\nAmdahl serial fraction s = T_StageA(1)/T_1 = {s_serial:.4f}")
    print(f"Amdahl ceiling  S_inf = 1/s = {1.0 / s_serial:.2f}\n")
    return rows, s_serial


def weak_scaling(p_list):
    print(f"--- 2. Weak scaling (N_0 = {N0_WEAK:,} per worker) ---")
    print(f"{'p':>3} {'N':>10} {'T_ms':>10} {'StageA_ms':>10} {'E_weak':>8} {'alpha':>7} {'S_gust':>8}")
    t1 = None
    rows = {}
    for p in p_list:
        n = p * N0_WEAK
        total, stage_a = bench(n, p)
        if t1 is None:
            t1 = total.median_ms
        e_weak = t1 / total.median_ms
        alpha = stage_a.median_ms / total.median_ms
        s_gust = p - alpha * (p - 1)
        rows[p] = (total, stage_a, e_weak, alpha, s_gust)
        print(f"{p:>3} {n:>10,} {total.median_ms:>10.1f} {stage_a.median_ms:>10.1f} "
              f"{e_weak:>8.3f} {alpha:>7.4f} {s_gust:>8.3f}")
    print()
    return rows


def inflection(strong_rows, p_fast, p_log):
    print("--- 3. Architectural inflection (marginal speedup dS/dp) ---")
    print(f"{'p_from':>7} {'p_to':>5} {'dS':>8} {'dS/dp':>8}  note")
    ps = sorted(strong_rows)
    for a, b in zip(ps, ps[1:]):
        ds = strong_rows[b][2] - strong_rows[a][2]
        note = ""
        if a <= p_fast < b:
            note = "<- P-core pool exhausted, E-cores engage"
        if a <= p_log < b:
            note = "<- oversubscription past P_log"
        print(f"{a:>7} {b:>5} {ds:>8.3f} {ds / (b - a):>8.3f}  {note}")
    print()


def main() -> None:
    p_phys = physical_cores()
    p_fast = performance_cores()
    p_log = mp.cpu_count()
    p_list = p_ladder(p_fast, p_log)

    print("=== Phase 2: Strong Scaling vs Weak Scaling ===")
    print(f"P_phys = {p_phys}, P_log = {p_log}, "
          f"P-cores = {p_fast}, E-cores = {p_phys - p_fast}")
    print(f"Stage A: {SERIAL_FRACTION:.0%} of N, 10 sin/sqrt ops/element, serial in parent")
    print(f"Stage B: all N elements, 30 FMA/trig ops/element, mp.Pool(p), no shared writes")
    print(f"p ladder: {p_list}")
    print(f"{WARMUP_RUNS} warmup + {TIMED_RUNS} timed runs per point\n")

    strong_rows, s_serial = strong_scaling(p_list)
    weak_rows = weak_scaling(p_list)
    inflection(strong_rows, p_fast, p_log)

    s_strong = strong_rows[p_log][2]
    s_gust = weak_rows[p_log][4]
    print(f"S_strong(p={p_log})     = {s_strong:.4f}  (Amdahl cap {1.0 / s_serial:.2f})")
    print(f"S_Gustafson(p={p_log})  = {s_gust:.4f}\n")
    print(emit_verify_tag("PHASE2", STUDENT_SEED, s_strong, s_gust))


if __name__ == "__main__":
    main()
