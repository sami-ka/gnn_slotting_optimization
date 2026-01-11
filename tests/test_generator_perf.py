"""Performance benchmarks for DataGenerator optimizations.

These tests validate that optimizations achieve expected speedups.
Run with: uv run pytest tests/test_generator_perf.py -v
"""

import time
import pytest

from slotting_optimization.generator import DataGenerator
from slotting_optimization.warehouse import Warehouse


def test_warehouse_bulk_performance():
    """Benchmark: bulk method should be >10x faster for 1000 locations"""
    locations = [f"L{i}" for i in range(1000)]
    nodes = ["start", "end"] + locations

    # Create large distance map (full directed graph excluding self-loops)
    dist_map = {}
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if a != b:
                dist_map[(a, b)] = 1.5

    # Test individual calls
    w1 = Warehouse(locations=["start", "end"] + locations,
                   start_point_id="start", end_point_id="end")
    t0 = time.perf_counter()
    for (a, b), d in dist_map.items():
        w1.set_distance(a, b, d)
    dur_individual = time.perf_counter() - t0

    # Test bulk call
    w2 = Warehouse(locations=["start", "end"] + locations,
                   start_point_id="start", end_point_id="end")
    t0 = time.perf_counter()
    w2.set_distances_bulk(dist_map)
    dur_bulk = time.perf_counter() - t0

    # Verify correctness
    assert w1._distance_map == w2._distance_map

    # Expect >10x speedup
    speedup = dur_individual / dur_bulk
    print(f"\n  Individual calls: {dur_individual:.3f}s")
    print(f"  Bulk method: {dur_bulk:.3f}s")
    print(f"  Speedup: {speedup:.1f}x")

    assert speedup > 10, f"Expected >10x speedup, got {speedup:.1f}x"


def test_orderbook_creation_performance():
    """Benchmark: from_dicts_direct should be faster than from_orders"""
    from datetime import datetime, timedelta
    from slotting_optimization.order_book import OrderBook
    from slotting_optimization.models import Order

    N = 10000
    base_time = datetime(2025, 1, 1, 10, 0, 0)

    # Test from_orders()
    orders = [
        Order(order_id=f"o{i}", item_id=f"sku{i%100}",
              timestamp=base_time + timedelta(seconds=i))
        for i in range(N)
    ]
    t0 = time.perf_counter()
    ob1 = OrderBook.from_orders(orders)
    dur_orders = time.perf_counter() - t0

    # Test from_dicts_direct()
    records = [
        {"order_id": f"o{i}", "item_id": f"sku{i%100}",
         "timestamp": base_time + timedelta(seconds=i)}
        for i in range(N)
    ]
    t0 = time.perf_counter()
    ob2 = OrderBook.from_dicts_direct(records)
    dur_direct = time.perf_counter() - t0

    speedup = dur_orders / dur_direct
    print(f"\n  from_orders(): {dur_orders:.3f}s")
    print(f"  from_dicts_direct(): {dur_direct:.3f}s")
    print(f"  Speedup: {speedup:.1f}x")

    # Should be at least slightly faster (>1.2x)
    assert speedup > 1.2, f"Expected >1.2x speedup, got {speedup:.1f}x"


def test_generate_samples_small_scale_performance():
    """Benchmark: Small scale (10 samples × 100 locations)"""
    gen = DataGenerator()

    t0 = time.perf_counter()
    samples = gen.generate_samples(
        n_locations=100,
        nb_items=100,
        n_orders=100,
        min_items_per_order=2,
        max_items_per_order=5,
        n_samples=10,
        distances_fixed=True,
        seed=42
    )
    duration = time.perf_counter() - t0

    # Verify correctness
    assert len(samples) == 10

    per_sample_ms = (duration / 10) * 1000
    print(f"\n  Total: {duration:.3f}s")
    print(f"  Per sample: {per_sample_ms:.1f}ms")

    # Should complete quickly (<1s)
    assert duration < 1.0, f"Small scale generation too slow: {duration:.2f}s"


def test_generate_samples_medium_scale_performance():
    """Benchmark: Medium scale (100 samples × 500 locations)"""
    gen = DataGenerator()

    t0 = time.perf_counter()
    samples = gen.generate_samples(
        n_locations=500,
        nb_items=500,
        n_orders=100,
        min_items_per_order=2,
        max_items_per_order=5,
        n_samples=100,
        distances_fixed=True,
        seed=42
    )
    duration = time.perf_counter() - t0

    # Verify correctness
    assert len(samples) == 100

    per_sample_ms = (duration / 100) * 1000
    print(f"\n  Total: {duration:.3f}s")
    print(f"  Per sample: {per_sample_ms:.1f}ms")

    # Target: <5s for medium scale
    assert duration < 5.0, f"Medium scale generation too slow: {duration:.2f}s"


def test_generate_samples_large_scale_performance():
    """Benchmark: Large scale (100 samples × 1000 locations)"""
    gen = DataGenerator()

    t0 = time.perf_counter()
    samples = gen.generate_samples(
        n_locations=1000,
        nb_items=1000,
        n_orders=100,
        min_items_per_order=2,
        max_items_per_order=5,
        n_samples=100,
        distances_fixed=True,
        seed=42
    )
    duration = time.perf_counter() - t0

    # Verify correctness
    assert len(samples) == 100

    per_sample_ms = (duration / 100) * 1000
    print(f"\n  Total: {duration:.3f}s")
    print(f"  Per sample: {per_sample_ms:.1f}ms")

    # Target: <30s for large scale
    assert duration < 30.0, f"Large scale generation too slow: {duration:.2f}s"


def test_warehouse_reuse_memory_efficiency():
    """Verify warehouse reuse doesn't create unnecessary copies"""
    import sys

    gen = DataGenerator()

    samples = gen.generate_samples(
        n_locations=1000,
        nb_items=1000,
        n_orders=10,
        min_items_per_order=1,
        max_items_per_order=2,
        n_samples=100,
        distances_fixed=True,
        seed=42
    )

    # Check that all warehouses share the same distance_map reference
    dist_maps = [id(w._distance_map) for _, _, w in samples]
    unique_dist_maps = set(dist_maps)

    # Should only have one unique distance_map object
    assert len(unique_dist_maps) == 1, \
        f"Expected all warehouses to share distance_map, got {len(unique_dist_maps)} unique objects"

    # Estimate memory savings
    _, _, first_warehouse = samples[0]
    dist_map_size = sys.getsizeof(first_warehouse._distance_map)

    # Each entry is roughly 8 bytes (float) + overhead
    n_distances = len(first_warehouse._distance_map)
    savings_mb = (dist_map_size * 99) / (1024 * 1024)  # 99 copies avoided

    print(f"\n  Distance map entries: {n_distances}")
    print(f"  Distance map size: {dist_map_size / 1024:.1f} KB")
    print(f"  Memory saved by reuse: ~{savings_mb:.1f} MB")


def test_distances_fixed_vs_variable_performance():
    """Compare performance of distances_fixed=True vs False"""
    gen = DataGenerator()

    # Test with distances_fixed=True
    t0 = time.perf_counter()
    samples_fixed = gen.generate_samples(
        n_locations=500,
        nb_items=500,
        n_orders=50,
        min_items_per_order=2,
        max_items_per_order=5,
        n_samples=50,
        distances_fixed=True,
        seed=42
    )
    dur_fixed = time.perf_counter() - t0

    # Test with distances_fixed=False
    t0 = time.perf_counter()
    samples_variable = gen.generate_samples(
        n_locations=500,
        nb_items=500,
        n_orders=50,
        min_items_per_order=2,
        max_items_per_order=5,
        n_samples=50,
        distances_fixed=False,
        seed=42
    )
    dur_variable = time.perf_counter() - t0

    speedup = dur_variable / dur_fixed
    print(f"\n  distances_fixed=True: {dur_fixed:.3f}s")
    print(f"  distances_fixed=False: {dur_variable:.3f}s")
    print(f"  Speedup with reuse: {speedup:.1f}x")

    # Fixed should be significantly faster (>5x) due to warehouse reuse
    assert speedup > 5.0, f"Expected >5x speedup with warehouse reuse, got {speedup:.1f}x"
