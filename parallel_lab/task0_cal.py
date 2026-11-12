
import multiprocessing as mp
import platform
import sys
import time

from harness import emit_verify_tag

STUDENT_SEED = 230103269
CALIB_CALLS = 1_000_000


def measure_clock_latency(n_calls: int = CALIB_CALLS) -> float:
    t0 = time.perf_counter_ns()
    for _ in range(n_calls):
        time.perf_counter_ns()
    t1 = time.perf_counter_ns()
    return (t1 - t0) / n_calls


def measure_clock_resolution(n_samples: int = 10_000) -> int:
    """Smallest non-zero delta the clock reports between back-to-back reads."""
    smallest = None
    for _ in range(n_samples):
        a = time.perf_counter_ns()
        b = time.perf_counter_ns()
        d = b - a
        if d > 0 and (smallest is None or d < smallest):
            smallest = d
    return smallest if smallest is not None else 0


def main() -> None:
    p_log = mp.cpu_count()

    print("=== Phase 0: Hardware Topology & Clock Calibration ===")
    print(f"Platform        : {platform.platform()}")
    print(f"Machine         : {platform.machine()}")
    print(f"Python          : {sys.version.split()[0]}")
    print(f"mp.cpu_count()  : {p_log}")
    print()

    measure_clock_latency(50_000)

    latency_ns = measure_clock_latency()
    resolution_ns = measure_clock_resolution()

    print(f"perf_counter_ns() avg latency : {latency_ns:.2f} ns over {CALIB_CALLS:,} calls")
    print(f"perf_counter_ns() resolution  : {resolution_ns} ns")
    print(f"time.get_clock_info()         : {time.get_clock_info('perf_counter')}")
    print()
    print("Timing floor: any phase whose measured region is shorter than")
    print(f"~{latency_ns * 2:.0f} ns is dominated by clock overhead, not work.")
    print()
    print(emit_verify_tag("PHASE0", STUDENT_SEED, latency_ns, float(p_log)))


if __name__ == "__main__":
    main()
