"""Benchmark build_matrices vs build_matrices_fast.

Usage: python scripts/benchmark_matrices.py
"""
from time import perf_counter
import sys
import os
# Ensure package import works when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from slotting_optimization.models import Order
from slotting_optimization.order_book import OrderBook
from slotting_optimization.item_locations import ItemLocations
from slotting_optimization.warehouse import Warehouse
from slotting_optimization.simulator import build_matrices, build_matrices_fast


def make_problem(n_locs=50, n_items=100, n_orders=1000, max_items_per_order=4, seed=0):
    rng = np.random.default_rng(seed)
    # locations
    locs = [f"L{i}" for i in range(n_locs)]
    w = Warehouse(locations=locs, start_point_id=locs[0], end_point_id=locs[-1])

    # Fill a dense-ish distance map (avoid heavy loops in benchmark by not using full n^2 for huge sizes)
    for i, a in enumerate(locs):
        for j, b in enumerate(locs):
            w.set_distance(a, b, float(i + j + 1))

    # items -> locations
    items = [f"sku{i}" for i in range(n_items)]
    il_recs = []
    for i, it in enumerate(items):
        loc = locs[rng.integers(0, n_locs)]
        il_recs.append({"item_id": it, "location_id": loc})
    il = ItemLocations.from_records(il_recs)

    # orders: create n_orders, each with random len 1..max_items_per_order
    orders = []
    ts_base = 1600000000
    for oi in range(n_orders):
        k = rng.integers(1, max_items_per_order + 1)
        # sample with replacement
        chosen = rng.integers(0, n_items, size=k)
        for j, ci in enumerate(chosen):
            ts = f"2025-01-01T00:{(oi % 60):02d}:{j:02d}"
            orders.append(Order.from_dict({"order_id": f"o{oi}", "item_id": f"sku{ci}", "timestamp": ts}))

    ob = OrderBook.from_orders(orders)
    return ob, il, w


def bench_once(fn, ob, il, w):
    t0 = perf_counter()
    res = fn(ob, il, w)
    t1 = perf_counter()
    return t1 - t0, res


def run_bench():
    scenarios = [
        (50, 100, 2000),
        (100, 200, 5000),
        (200, 400, 10000),
    ]

    for (n_locs, n_items, n_orders) in scenarios:
        print(f"\nScenario: locs={n_locs}, items={n_items}, orders={n_orders}")
        ob, il, w = make_problem(n_locs, n_items, n_orders)

        # Warm-up
        build_matrices(ob, il, w)
        build_matrices_fast(ob, il, w)

        # Timed runs
        reps = 3
        t1 = min(bench_once(build_matrices, ob, il, w)[0] for _ in range(reps))
        t2 = min(bench_once(build_matrices_fast, ob, il, w)[0] for _ in range(reps))

        print(f"build_matrices: {t1:.4f}s, build_matrices_fast: {t2:.4f}s, speedup: {t1 / t2:.2f}x")


if __name__ == "__main__":
    run_bench()
