import multiprocessing as mp
import subprocess
import time
from multiprocessing import shared_memory

from harness import emit_verify_tag, stats_from_samples

STUDENT_SEED = 230103269
BUF_BYTES = 64 * 10**6
FLOPS_PER_WORKER = 60_000_000
N = 256
BS_LIST = (8, 16, 32, 64, 96, 128)
WARMUP_RUNS = 1
TIMED_RUNS = 3


def physical_cores() -> int:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                             capture_output=True, text=True, check=True)
        return int(out.stdout.strip())
    except Exception:
        return mp.cpu_count()


def copy_worker(barrier, name_a, name_b, start, end):
    sa = shared_memory.SharedMemory(name=name_a)
    sb = shared_memory.SharedMemory(name=name_b)
    barrier.wait()
    sa.buf[start:end] = sb.buf[start:end]
    sa.close()
    sb.close()


def fma_worker(barrier, iters):
    a, b = 0.999999, 0.000001
    x0 = x1 = x2 = x3 = x4 = x5 = x6 = x7 = 1.0
    x8 = x9 = x10 = x11 = x12 = x13 = x14 = x15 = 1.0
    barrier.wait()
    for _ in range(iters):
        x0 = x0 * a + b
        x1 = x1 * a + b
        x2 = x2 * a + b
        x3 = x3 * a + b
        x4 = x4 * a + b
        x5 = x5 * a + b
        x6 = x6 * a + b
        x7 = x7 * a + b
        x8 = x8 * a + b
        x9 = x9 * a + b
        x10 = x10 * a + b
        x11 = x11 * a + b
        x12 = x12 * a + b
        x13 = x13 * a + b
        x14 = x14 * a + b
        x15 = x15 * a + b


def gemm_worker(barrier, names, n, r0, r1, variant, bs):
    shms = [shared_memory.SharedMemory(name=x) for x in names]
    a, b, c = [s.buf.cast("d") for s in shms]
    barrier.wait()
    if variant == 1:
        for i in range(r0, r1):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += a[i * n + k] * b[k * n + j]
                c[i * n + j] = s
    elif variant == 2:
        for i in range(r0, r1):
            ci = i * n
            for k in range(n):
                aik = a[ci + k]
                bk = k * n
                for j in range(n):
                    c[ci + j] += aik * b[bk + j]
    else:
        for ii in range(r0, r1, bs):
            for jj in range(0, n, bs):
                je = min(jj + bs, n)
                for kk in range(0, n, bs):
                    for i in range(ii, min(ii + bs, r1)):
                        ci = i * n
                        for k in range(kk, min(kk + bs, n)):
                            aik = a[ci + k]
                            bk = k * n
                            for j in range(jj, je):
                                c[ci + j] += aik * b[bk + j]
    for m in (a, b, c):
        m.release()
    for s in shms:
        s.close()


def run_parallel(target, args_list):
    barrier = mp.Barrier(len(args_list) + 1)
    procs = [mp.Process(target=target, args=(barrier, *args)) for args in args_list]
    for pr in procs:
        pr.start()
    barrier.wait()
    t0 = time.perf_counter_ns()
    for pr in procs:
        pr.join()
    return (time.perf_counter_ns() - t0) / 1e6


def bench(target, args_list, before=None):
    samples = []
    for r in range(WARMUP_RUNS + TIMED_RUNS):
        if before:
            before()
        t = run_parallel(target, args_list)
        if r >= WARMUP_RUNS:
            samples.append(t)
    return stats_from_samples(samples)


def split(n, p):
    step = -(-n // p)
    return [(w * step, min((w + 1) * step, n)) for w in range(p) if w * step < n]


def roofline(p):
    sa = shared_memory.SharedMemory(create=True, size=BUF_BYTES)
    sb = shared_memory.SharedMemory(create=True, size=BUF_BYTES)
    try:
        sb.buf[:BUF_BYTES] = bytes(range(256)) * (BUF_BYTES // 256)
        st = bench(copy_worker, [(sa.name, sb.name, s, e) for s, e in split(BUF_BYTES, p)])
    finally:
        for s in (sa, sb):
            s.close()
            s.unlink()
    bw = 2 * BUF_BYTES / (st.median_ms / 1e3) / 1e9

    iters = FLOPS_PER_WORKER // 32
    st2 = bench(fma_worker, [(iters,)] * p)
    gflops = p * iters * 32 / (st2.median_ms / 1e3) / 1e9

    print(f"--- 3A. Roofline (p = {p}) ---")
    print(f"copy 2 x 64 MB : median {st.median_ms:.2f} ms, IQR {st.iqr_ms:.2f} ms -> BW = {bw:.2f} GB/s")
    print(f"FMA 6e7/worker : median {st2.median_ms:.2f} ms, IQR {st2.iqr_ms:.2f} ms -> P = {gflops:.4f} GFLOP/s")
    print(f"ridge point I* = P/BW = {gflops / bw:.6f} FLOP/B\n")
    return bw, gflops


def gemm(p, bw, peak):
    size = N * N * 8
    shms = [shared_memory.SharedMemory(create=True, size=size) for _ in range(3)]
    names = [s.name for s in shms]
    a, b, c = [s.buf.cast("d") for s in shms]
    for i in range(N * N):
        a[i] = (i % 7) * 0.5 + 0.1
        b[i] = (i % 5) * 0.25 - 0.3

    def zero_c():
        shms[2].buf[:size] = bytes(size)

    rows = split(N, p)
    flops = 2 * N ** 3

    def run(variant, bs=0):
        st = bench(gemm_worker, [(names, N, r0, r1, variant, bs) for r0, r1 in rows], zero_c)
        return st, flops / (st.median_ms / 1e3) / 1e9

    print(f"--- 3B. GEMM N = {N}, p = {p} ---")
    print(f"{'variant':>12} {'median_ms':>10} {'IQR_ms':>8} {'GFLOP/s':>9} {'W_tile_KB':>10} {'I_FLOP/B':>9} {'%roof':>7} {'max_err':>9}")
    try:
        st1, g1 = run(1)
        st2, g2 = run(2)
        ref = c.tolist()
        att = min(peak, 0.125 * bw)
        print(f"{'1 naive ijk':>12} {st1.median_ms:>10.1f} {st1.iqr_ms:>8.1f} {g1:>9.4f} {'-':>10} {0.125:>9.3f} {100 * g1 / att:>6.1f}% {'-':>9}")
        print(f"{'2 ikj':>12} {st2.median_ms:>10.1f} {st2.iqr_ms:>8.1f} {g2:>9.4f} {'-':>10} {0.125:>9.3f} {100 * g2 / att:>6.1f}% {'-':>9}")
        best = None
        for bs in BS_LIST:
            st3, g3 = run(3, bs)
            err = max(abs(x - y) for x, y in zip(c.tolist(), ref))
            assert err < 1e-9
            inten = bs / 12
            att = min(peak, inten * bw)
            print(f"{'3 BS=' + str(bs):>12} {st3.median_ms:>10.1f} {st3.iqr_ms:>8.1f} {g3:>9.4f} "
                  f"{3 * bs * bs * 8 / 1024:>10.1f} {inten:>9.3f} {100 * g3 / att:>6.1f}% {err:>9.1e}")
            if best is None or g3 > best[1]:
                best = (bs, g3)
    finally:
        for m in (a, b, c):
            m.release()
        for s in shms:
            s.close()
            s.unlink()
    print(f"\nBS* = {best[0]} (W_tile = {3 * best[0] ** 2 * 8 / 1024:.1f} KB), GFLOP/s = {best[1]:.4f}")
    print(f"V3(BS*)/V1 = {best[1] / g1:.2f}x\n")
    return g1, best[1]


def main():
    p = physical_cores()
    print("=== Phase 3: Empirical Roofline & Cache-Blocked GEMM ===")
    print(f"P_phys = {p}, P_log = {mp.cpu_count()}, {WARMUP_RUNS} warmup + {TIMED_RUNS} timed runs\n")
    bw, peak = roofline(p)
    g1, g3 = gemm(p, bw, peak)
    print(emit_verify_tag("PHASE3", STUDENT_SEED, g1, g3))


if __name__ == "__main__":
    main()
