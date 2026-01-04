"""Micro-benchmark for Simulator implementations.

Runs both `Simulator.simulate` (baseline) and `Simulator.simulate_matrix` (matrix-based)
multiple times, computes summary statistics (mean, std, min, median) and writes a
Markdown report to `docs/benchmark_simulator.md`.
"""

import statistics
import time
import random
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so local package can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slotting_optimization.models import Order
from slotting_optimization.item_locations import ItemLocations
from slotting_optimization.order_book import OrderBook
from slotting_optimization.warehouse import Warehouse
from slotting_optimization.simulator import Simulator


def generate_problem(locations=1000, orders=5000, seed=42):
    random.seed(seed)
    LOCATIONS = locations
    ORDERS = orders

    locations_list = [f"L{i}" for i in range(LOCATIONS)]
    w = Warehouse(locations=["start", "end"] + locations_list, start_point_id="start", end_point_id="end")

    for i, loc in enumerate(locations_list):
        w.set_distance("start", loc, 1.0 + (i % 10) * 0.1)
        w.set_distance(loc, "end", 0.5 + (i % 7) * 0.07)
    w.set_distance("end", "start", 2.5)

    records = [{"item_id": f"sku{i}", "location_id": locations_list[i]} for i in range(LOCATIONS)]
    il = ItemLocations.from_records(records)

    orders_list = []
    base_ts = int(time.time())
    for i in range(ORDERS):
        item_idx = random.randrange(LOCATIONS)
        ts = base_ts + i
        orders_list.append(Order.from_dict({"order_id": f"o{i}", "item_id": f"sku{item_idx}", "timestamp": ts}))

    ob = OrderBook.from_orders(orders_list)
    return ob, w, il


def run_benchmark(reps=10, warmup=2, locations=1000, orders=5000):
    ob, w, il = generate_problem(locations=locations, orders=orders)
    sim = Simulator()

    def time_fn(fn):
        # warmup
        for _ in range(warmup):
            fn()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        return times

    def baseline_call():
        sim.simulate(ob, w, il)

    def matrix_call():
        sim.simulate_matrix(ob, w, il)

    baseline_times = time_fn(baseline_call)
    sparse_times = time_fn(lambda: sim.simulate_sparse_matrix(ob, w, il))

    def summarize(times):
        return {
            "mean": statistics.mean(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min": min(times),
            "median": statistics.median(times),
            "reps": len(times),
            "times": times,
        }

    return summarize(baseline_times), summarize(sparse_times)


def write_report(baseline_stats, sparse_stats, locations, orders, out_path="docs/benchmark_simulator.md"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Simulator Micro-benchmark\n")
    lines.append(f"**Problem size:** {orders} orders, {locations} locations\n")

    lines.append("## Summary\n")
    lines.append("| Implementation | mean (s) | stdev (s) | min (s) | median (s) | reps |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    lines.append(
        f"| baseline | {baseline_stats['mean']:.6f} | {baseline_stats['stdev']:.6f} | {baseline_stats['min']:.6f} | {baseline_stats['median']:.6f} | {baseline_stats['reps']} |\n"
    )
    lines.append(
        f"| sparse   | {sparse_stats['mean']:.6f} | {sparse_stats['stdev']:.6f} | {sparse_stats['min']:.6f} | {sparse_stats['median']:.6f} | {sparse_stats['reps']} |\n"
    )

    speedup_sparse = baseline_stats['mean'] / sparse_stats['mean'] if sparse_stats['mean'] > 0 else float('inf')
    lines.append(f"\n**Speedup (baseline / sparse):** {speedup_sparse:.3f}x\n")

    lines.append("## Raw timings (s)\n")
    lines.append("Baseline times:\n")
    lines.extend([f"- {t:.6f}\n" for t in baseline_stats['times']])
    lines.append("\nSparse times:\n")
    lines.extend([f"- {t:.6f}\n" for t in sparse_stats['times']])

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)

    print(f"Wrote benchmark report to {out_path}")


if __name__ == "__main__":
    # small default reps to keep runtime reasonable
    baseline_stats, sparse_stats = run_benchmark(reps=5, warmup=2, locations=1000, orders=5000)
    write_report(baseline_stats, sparse_stats, locations=1000, orders=5000)
    print("Done")
