"""Tests for DataGenerator.generate_optimal_samples().

Verifies that the generated (OrderBook, ItemLocations, Warehouse) tuples have
the property that the returned ItemLocations assignment is provably optimal
for the given orderbook and warehouse.
"""

import pytest
import numpy as np

from slotting_optimization.generator import DataGenerator
from slotting_optimization.order_book import OrderBook
from slotting_optimization.item_locations import ItemLocations
from slotting_optimization.warehouse import Warehouse
from slotting_optimization.simulator import Simulator, build_matrices_fast
from slotting_optimization.validation.ground_truth import compute_brute_force_optimal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unpack(sample):
    ob, il, w, meta = sample
    return ob, il, w, meta


# ---------------------------------------------------------------------------
# Test 1: shapes and types
# ---------------------------------------------------------------------------


def test_optimal_shapes_and_types():
    gen = DataGenerator()
    result = gen.generate_optimal_samples(
        n_items=4,
        n_locations=4,
        n_orders=50,
        n_samples=2,
        min_items_per_order=1,
        max_items_per_order=2,
        noise_level=0.0,
        seed=42,
    )

    assert len(result) == 2

    for sample in result:
        assert len(sample) == 4
        ob, il, w, meta = sample

        assert isinstance(ob, OrderBook)
        assert isinstance(il, ItemLocations)
        assert isinstance(w, Warehouse)
        assert isinstance(meta, dict)

        # Metadata keys
        for key in (
            "items",
            "storage_locations",
            "optimal_assignment",
            "optimal_distance",
            "item_frequencies",
            "location_costs",
            "noise_level",
        ):
            assert key in meta, f"Missing metadata key: {key}"

        assert len(meta["items"]) == 4
        assert len(meta["storage_locations"]) == 4
        assert len(meta["optimal_assignment"]) == 4
        assert meta["optimal_distance"] > 0
        assert meta["noise_level"] == 0.0

        # OrderBook has correct number of logical orders
        order_ids = set(ob.to_df().get_column("order_id").to_list())
        assert len(order_ids) == 50

        # Each order has between min and max line items
        counts = ob.to_df().group_by("order_id").len().get_column("len").to_list()
        for c in counts:
            assert 1 <= c <= 2

        # Warehouse has start and end
        assert w.start_point is not None
        assert w.end_point is not None


# ---------------------------------------------------------------------------
# Test 2: optimal assignment is truly optimal (noise=0, brute force check)
# ---------------------------------------------------------------------------


def test_optimal_assignment_is_truly_optimal_small():
    gen = DataGenerator()
    result = gen.generate_optimal_samples(
        n_items=4,
        n_locations=4,
        n_orders=300,
        n_samples=1,
        noise_level=0.0,
        seed=7,
    )
    ob, il, w, meta = result[0]

    items = meta["items"]
    storage_locations = meta["storage_locations"]

    bf_assignment, bf_dist = compute_brute_force_optimal(
        ob, w, items, storage_locations
    )

    # Brute force should match or beat our claimed optimal distance
    assert bf_dist <= meta["optimal_distance"] + 1e-6, (
        f"Brute force found better: {bf_dist:.4f} vs claimed {meta['optimal_distance']:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3: optimal assignment is truly optimal with noise=0.5
# ---------------------------------------------------------------------------


def test_optimal_assignment_is_truly_optimal_with_noise():
    gen = DataGenerator()
    result = gen.generate_optimal_samples(
        n_items=4,
        n_locations=4,
        n_orders=500,
        n_samples=1,
        noise_level=0.5,
        seed=13,
    )
    ob, il, w, meta = result[0]

    items = meta["items"]
    storage_locations = meta["storage_locations"]

    bf_assignment, bf_dist = compute_brute_force_optimal(
        ob, w, items, storage_locations
    )

    assert bf_dist <= meta["optimal_distance"] + 1e-6, (
        f"Brute force found better: {bf_dist:.4f} vs claimed {meta['optimal_distance']:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: full noise (noise_level=1.0) — all assignments equally good
# ---------------------------------------------------------------------------


def test_optimal_assignment_at_full_noise():
    gen = DataGenerator()
    result = gen.generate_optimal_samples(
        n_items=3,
        n_locations=3,
        n_orders=600,
        n_samples=1,
        noise_level=1.0,
        seed=99,
    )
    ob, il, w, meta = result[0]

    items = meta["items"]
    storage_locations = meta["storage_locations"]

    bf_assignment, bf_dist = compute_brute_force_optimal(
        ob, w, items, storage_locations
    )

    # With uniform frequencies, brute force should match our assignment (within tolerance)
    assert bf_dist <= meta["optimal_distance"] + 1e-6


# ---------------------------------------------------------------------------
# Test 5: noise=0 has steeper frequency gradient than noise=0.8
# ---------------------------------------------------------------------------


def test_noise_level_zero_has_steep_gradient():
    gen = DataGenerator()

    result_low = gen.generate_optimal_samples(
        n_items=5,
        n_locations=5,
        n_orders=200,
        n_samples=1,
        noise_level=0.0,
        seed=21,
    )
    result_high = gen.generate_optimal_samples(
        n_items=5,
        n_locations=5,
        n_orders=200,
        n_samples=1,
        noise_level=0.8,
        seed=21,
    )

    freqs_low = list(result_low[0][3]["item_frequencies"].values())
    freqs_high = list(result_high[0][3]["item_frequencies"].values())

    # Standard deviation of frequencies should be larger at low noise
    std_low = np.std(freqs_low)
    std_high = np.std(freqs_high)
    assert std_low >= std_high, (
        f"Expected steeper gradient at noise=0, but std_low={std_low:.2f} < std_high={std_high:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 6: multi-item orders preserve frequency ordering (noise=0)
# ---------------------------------------------------------------------------


def test_multi_item_orders_frequency_preservation():
    gen = DataGenerator()
    result = gen.generate_optimal_samples(
        n_items=5,
        n_locations=5,
        n_orders=500,
        n_samples=1,
        min_items_per_order=3,
        max_items_per_order=5,
        noise_level=0.0,
        seed=55,
    )
    ob, il, w, meta = result[0]

    items = meta["items"]
    freqs = meta["item_frequencies"]

    # item_0 should have highest frequency, item_{n-1} lowest
    # Allow for sampling noise in edge cases, but overall trend should hold
    freq_values = [freqs[item] for item in items]
    for i in range(len(freq_values) - 1):
        assert freq_values[i] >= freq_values[i + 1], (
            f"Frequency ordering violated: freq({items[i]})={freq_values[i]} "
            f"< freq({items[i + 1]})={freq_values[i + 1]}"
        )


# ---------------------------------------------------------------------------
# Test 7: multi-item orders populate seq_mat
# ---------------------------------------------------------------------------


def test_multi_item_orders_seq_mat_populated():
    gen = DataGenerator()
    result = gen.generate_optimal_samples(
        n_items=4,
        n_locations=4,
        n_orders=50,
        n_samples=1,
        min_items_per_order=2,
        max_items_per_order=4,
        noise_level=0.0,
        seed=77,
    )
    ob, il, w, meta = result[0]

    _, seq_mat, _, _, _ = build_matrices_fast(ob, il, w)

    # seq_mat should have non-zero entries from item co-occurrences
    assert seq_mat.sum() > 0, (
        "seq_mat should have non-zero entries for multi-item orders"
    )


# ---------------------------------------------------------------------------
# Test 8: seed reproducibility
# ---------------------------------------------------------------------------


def test_seed_reproducibility():
    gen = DataGenerator()
    params = dict(
        n_items=4,
        n_locations=4,
        n_orders=50,
        n_samples=2,
        min_items_per_order=1,
        max_items_per_order=2,
        noise_level=0.3,
        seed=42,
    )
    result_a = gen.generate_optimal_samples(**params)
    result_b = gen.generate_optimal_samples(**params)

    for (ob_a, il_a, w_a, meta_a), (ob_b, il_b, w_b, meta_b) in zip(result_a, result_b):
        df_a = ob_a.to_df().sort(["order_id", "item_id", "timestamp"])
        df_b = ob_b.to_df().sort(["order_id", "item_id", "timestamp"])
        assert df_a.equals(df_b)

        assert il_a.to_dict() == il_b.to_dict()
        assert w_a._distance_map == w_b._distance_map
        assert meta_a["optimal_distance"] == meta_b["optimal_distance"]


# ---------------------------------------------------------------------------
# Test 9: multiple samples are independent (different warehouses)
# ---------------------------------------------------------------------------


def test_multiple_samples_are_independent():
    gen = DataGenerator()
    result = gen.generate_optimal_samples(
        n_items=3,
        n_locations=3,
        n_orders=30,
        n_samples=3,
        noise_level=0.0,
        seed=11,
    )

    assert len(result) == 3

    warehouses = [w for _, _, w, _ in result]
    distances = [meta["optimal_distance"] for _, _, _, meta in result]

    # Each sample should have a different warehouse
    assert warehouses[0]._distance_map != warehouses[1]._distance_map
    assert warehouses[1]._distance_map != warehouses[2]._distance_map

    # Optimal distances should differ (different warehouse structures)
    assert len(set(distances)) > 1, (
        "All samples have the same optimal distance — likely sharing state"
    )


# ---------------------------------------------------------------------------
# Test 10: validation errors
# ---------------------------------------------------------------------------


def test_validation_errors():
    gen = DataGenerator()

    with pytest.raises(ValueError, match="n_items must be positive"):
        gen.generate_optimal_samples(n_items=0, n_locations=5, n_orders=10)

    with pytest.raises(ValueError, match="n_items must be positive"):
        gen.generate_optimal_samples(n_items=-1, n_locations=5, n_orders=10)

    with pytest.raises(ValueError, match="cannot exceed n_locations"):
        gen.generate_optimal_samples(n_items=6, n_locations=5, n_orders=10)

    with pytest.raises(ValueError, match="noise_level must be in"):
        gen.generate_optimal_samples(
            n_items=3, n_locations=3, n_orders=10, noise_level=-0.1
        )

    with pytest.raises(ValueError, match="noise_level must be in"):
        gen.generate_optimal_samples(
            n_items=3, n_locations=3, n_orders=10, noise_level=1.1
        )

    with pytest.raises(ValueError, match="cost_progression"):
        gen.generate_optimal_samples(
            n_items=3, n_locations=3, n_orders=10, cost_progression="invalid"
        )


# ---------------------------------------------------------------------------
# Test 11: integration with simulator — distance matches metadata
# ---------------------------------------------------------------------------


def test_integration_with_simulator():
    gen = DataGenerator()
    result = gen.generate_optimal_samples(
        n_items=4,
        n_locations=4,
        n_orders=80,
        n_samples=1,
        noise_level=0.0,
        seed=33,
    )
    ob, il, w, meta = result[0]

    sim = Simulator()
    actual_dist, per_order = sim.simulate(ob, w, il)

    assert abs(actual_dist - meta["optimal_distance"]) < 1e-6, (
        f"Simulator distance {actual_dist:.4f} != metadata {meta['optimal_distance']:.4f}"
    )
    assert all(d > 0 for d in per_order)


# ---------------------------------------------------------------------------
# Test 12: integration with build_matrices_fast
# ---------------------------------------------------------------------------


def test_integration_with_build_matrices():
    gen = DataGenerator()
    result = gen.generate_optimal_samples(
        n_items=5,
        n_locations=5,
        n_orders=50,
        n_samples=1,
        noise_level=0.0,
        seed=44,
    )
    ob, il, w, meta = result[0]

    loc_mat, seq_mat, item_loc_mat, locs, items = build_matrices_fast(ob, il, w)

    # Shapes: L = n_locations + 2 (start + end)
    n_locs = 5 + 2
    assert loc_mat.shape == (n_locs, n_locs)
    assert item_loc_mat.shape == (5, n_locs)
    assert seq_mat.shape == (5, 5)

    # Each item assigned to exactly one location
    assert np.all(item_loc_mat.sum(axis=1) == 1), (
        "Each item should be assigned to exactly one location"
    )


# ---------------------------------------------------------------------------
# Test 13: pairwise swap condition (direct local optimality check)
# ---------------------------------------------------------------------------


def test_pairwise_swap_condition():
    gen = DataGenerator()
    result = gen.generate_optimal_samples(
        n_items=5,
        n_locations=5,
        n_orders=300,
        n_samples=1,
        noise_level=0.0,
        seed=88,
    )
    ob, il, w, meta = result[0]

    items = meta["items"]
    location_costs = meta["location_costs"]
    item_frequencies = meta["item_frequencies"]
    optimal_assignment = meta["optimal_assignment"]

    for a_idx in range(len(items)):
        for b_idx in range(a_idx + 1, len(items)):
            item_a = items[a_idx]
            item_b = items[b_idx]

            freq_diff = item_frequencies[item_a] - item_frequencies[item_b]
            cost_diff = (
                location_costs[optimal_assignment[item_a]]
                - location_costs[optimal_assignment[item_b]]
            )

            # Optimality: higher freq items must be at lower cost locations
            assert freq_diff * cost_diff <= 1e-9, (
                f"Swap violation: items ({item_a}, {item_b}): "
                f"freq_diff={freq_diff}, cost_diff={cost_diff:.4f}"
            )


# ---------------------------------------------------------------------------
# Test 14: cost progression variants all produce valid optimal samples
# ---------------------------------------------------------------------------


def test_cost_progression_variants():
    gen = DataGenerator()

    for progression in ("linear", "quadratic", "exponential"):
        result = gen.generate_optimal_samples(
            n_items=4,
            n_locations=4,
            n_orders=200,
            n_samples=1,
            noise_level=0.0,
            cost_progression=progression,
            seed=66,
        )
        ob, il, w, meta = result[0]

        # Location costs must be strictly increasing
        storage_locs = meta["storage_locations"]
        costs = [meta["location_costs"][loc] for loc in storage_locs]
        for i in range(len(costs) - 1):
            assert costs[i] < costs[i + 1], (
                f"[{progression}] Location costs not strictly increasing: "
                f"cost[{i}]={costs[i]:.4f} >= cost[{i + 1}]={costs[i + 1]:.4f}"
            )

        # Cross-validate with brute force
        bf_assignment, bf_dist = compute_brute_force_optimal(
            ob, w, meta["items"], storage_locs
        )
        assert bf_dist <= meta["optimal_distance"] + 1e-6, (
            f"[{progression}] Brute force found better: {bf_dist:.4f} vs {meta['optimal_distance']:.4f}"
        )
