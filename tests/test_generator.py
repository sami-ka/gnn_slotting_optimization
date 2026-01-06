import copy
import pytest

from slotting_optimization.generator import DataGenerator
from slotting_optimization.order_book import OrderBook
from slotting_optimization.item_locations import ItemLocations
from slotting_optimization.warehouse import Warehouse


def test_generate_shapes_and_types():
    gen = DataGenerator()
    samples = gen.generate_samples(
        n_locations=10, nb_items=10, n_orders=20, min_items_per_order=1, max_items_per_order=3, n_samples=3, distances_fixed=True, seed=42
    )
    assert len(samples) == 3
    for ob, il, w in samples:
        assert isinstance(ob, OrderBook)
        assert isinstance(il, ItemLocations)
        assert isinstance(w, Warehouse)

        # Item locations should cover all skus referenced in the orderbook
        item_ids = set(ob.to_df().get_column("item_id").to_list())
        for item in item_ids:
            assert il.get_location(item) is not None

        # unique logical orders
        order_ids = set(ob.to_df().get_column("order_id").to_list())
        assert len(order_ids) == 20

        # number of rows >= n_orders and <= n_orders * max_items
        nrows = ob.to_df().height
        assert nrows >= 20 and nrows <= 20 * 3

        # Warehouse must have start and end
        assert w.start_point is not None and w.end_point is not None


def test_distances_fixed_flag():
    gen = DataGenerator()
    samples_fixed = gen.generate_samples(5, 5, 10, 1, 2, n_samples=2, distances_fixed=True, seed=1)
    samples_var = gen.generate_samples(5, 5, 10, 1, 2, n_samples=2, distances_fixed=False, seed=1)

    _, _, w0 = samples_fixed[0]
    _, _, w1 = samples_fixed[1]
    # same distances when fixed
    assert w0._distance_map == w1._distance_map

    _, _, w0v = samples_var[0]
    _, _, w1v = samples_var[1]
    # likely different distances when not fixed
    assert w0v._distance_map != w1v._distance_map


def test_seed_reproducibility():
    gen = DataGenerator()
    a = gen.generate_samples(5, 5, 5, 1, 2, n_samples=1, distances_fixed=False, seed=123)
    b = gen.generate_samples(5, 5, 5, 1, 2, n_samples=1, distances_fixed=False, seed=123)

    # Compare warehouses and item locations and orderbooks
    ob_a, il_a, w_a = a[0]
    ob_b, il_b, w_b = b[0]

    # Compare orderbooks by sorted to_dicts for determinism
    assert sorted(ob_a.to_df().to_dicts(), key=lambda d: (d['order_id'], d['item_id'], d['timestamp'])) == sorted(ob_b.to_df().to_dicts(), key=lambda d: (d['order_id'], d['item_id'], d['timestamp']))
    assert il_a.to_dict() == il_b.to_dict()
    assert w_a._distance_map == w_b._distance_map


def test_min_max_items_range():
    gen = DataGenerator()
    samples = gen.generate_samples(5, 5, 20, 2, 4, n_samples=1, distances_fixed=True, seed=5)
    ob, il, w = samples[0]
    # each logical order id should appear between min and max times
    df = ob.to_df()
    counts = df.group_by("order_id").len().get_column("len").to_list()
    for c in counts:
        assert 2 <= c <= 4


# Validation Tests

def test_nb_items_validation_zero():
    """Test that nb_items = 0 raises ValueError"""
    gen = DataGenerator()
    with pytest.raises(ValueError, match="nb_items must be positive"):
        gen.generate_samples(n_locations=10, nb_items=0, n_orders=5,
                            min_items_per_order=1, max_items_per_order=2, seed=42)


def test_nb_items_validation_negative():
    """Test that negative nb_items raises ValueError"""
    gen = DataGenerator()
    with pytest.raises(ValueError, match="nb_items must be positive"):
        gen.generate_samples(n_locations=10, nb_items=-1, n_orders=5,
                            min_items_per_order=1, max_items_per_order=2, seed=42)


def test_nb_items_validation_exceeds_locations():
    """Test that nb_items > n_locations raises ValueError"""
    gen = DataGenerator()
    with pytest.raises(ValueError, match="nb_items .* cannot exceed n_locations"):
        gen.generate_samples(n_locations=5, nb_items=10, n_orders=5,
                            min_items_per_order=1, max_items_per_order=2, seed=42)


# Edge Case Tests

def test_nb_items_one():
    """Test edge case: nb_items=1 creates single item mapping at randomly selected location"""
    gen = DataGenerator()
    samples = gen.generate_samples(n_locations=10, nb_items=1, n_orders=5,
                                   min_items_per_order=1, max_items_per_order=2,
                                   n_samples=1, seed=42)
    ob, il, w = samples[0]

    # Exactly one item in mapping
    assert len(il) == 1
    # Item should be sku0
    assert "sku0" in il.to_dict()
    # The location should be one of L0-L9
    location = il.to_dict()["sku0"]
    assert location in [f"L{i}" for i in range(10)]
    # All orders should reference sku0
    item_ids = set(ob.to_df().get_column("item_id").to_list())
    assert item_ids == {"sku0"}


def test_nb_items_equals_n_locations():
    """Test that nb_items=n_locations uses all locations with random assignment (not deterministic)"""
    gen = DataGenerator()
    samples = gen.generate_samples(n_locations=10, nb_items=10, n_orders=20,
                                   min_items_per_order=1, max_items_per_order=3,
                                   n_samples=1, seed=42)
    ob, il, w = samples[0]

    # All items should be present
    assert len(il) == 10
    # All SKUs should be present (sku0 through sku9)
    items = set(il.to_dict().keys())
    expected_skus = {f"sku{i}" for i in range(10)}
    assert items == expected_skus
    # All locations should be used (L0 through L9)
    locations = set(il.to_dict().values())
    expected_locations = {f"L{i}" for i in range(10)}
    assert locations == expected_locations


# Functional Tests

def test_nb_items_partial_random_selection():
    """Test that nb_items < n_locations creates partial mapping with correct counts"""
    gen = DataGenerator()
    samples = gen.generate_samples(n_locations=20, nb_items=10, n_orders=50,
                                   min_items_per_order=2, max_items_per_order=5,
                                   n_samples=1, seed=123)
    ob, il, w = samples[0]

    # Exactly nb_items items in mapping
    assert len(il) == 10
    # SKUs should be sku0 through sku9
    items = set(il.to_dict().keys())
    expected_skus = {f"sku{i}" for i in range(10)}
    assert items == expected_skus
    # Only 10 of the 20 locations should be used
    used_locations = set(il.to_dict().values())
    assert len(used_locations) == 10
    # All used locations should be from the valid set
    all_locations = {f"L{i}" for i in range(20)}
    assert used_locations.issubset(all_locations)


def test_nb_items_reproducibility_with_seed():
    """Test that same seed produces same random location selection"""
    gen = DataGenerator()
    samples_a = gen.generate_samples(n_locations=15, nb_items=8, n_orders=10,
                                     min_items_per_order=1, max_items_per_order=2,
                                     n_samples=1, seed=999)
    samples_b = gen.generate_samples(n_locations=15, nb_items=8, n_orders=10,
                                     min_items_per_order=1, max_items_per_order=2,
                                     n_samples=1, seed=999)

    _, il_a, _ = samples_a[0]
    _, il_b, _ = samples_b[0]

    # Mappings should be identical
    assert il_a.to_dict() == il_b.to_dict()


def test_nb_items_different_seeds_different_selections():
    """Test that different seeds produce different location selections"""
    gen = DataGenerator()
    samples_a = gen.generate_samples(n_locations=20, nb_items=10, n_orders=10,
                                     min_items_per_order=1, max_items_per_order=2,
                                     n_samples=1, seed=111)
    samples_b = gen.generate_samples(n_locations=20, nb_items=10, n_orders=10,
                                     min_items_per_order=1, max_items_per_order=2,
                                     n_samples=1, seed=222)

    _, il_a, _ = samples_a[0]
    _, il_b, _ = samples_b[0]

    # Different seeds should produce different location selections
    # (with high probability for this size)
    assert il_a.to_dict() != il_b.to_dict()


# Integration Tests

def test_nb_items_integration_with_simulator():
    """Test that generated samples with nb_items < n_locations work with Simulator"""
    from slotting_optimization.simulator import Simulator

    gen = DataGenerator()
    samples = gen.generate_samples(n_locations=15, nb_items=8, n_orders=30,
                                   min_items_per_order=2, max_items_per_order=4,
                                   n_samples=1, seed=777)
    ob, il, w = samples[0]

    # Simulator should work without errors
    sim = Simulator()
    total_dist, per_order = sim.simulate(ob, w, il)

    # Sanity checks
    assert total_dist > 0
    # per_order has one entry per order line (item), not per logical order
    # With 30 orders and 2-4 items per order, expect 60-120 entries
    assert 60 <= len(per_order) <= 120
    assert all(d > 0 for d in per_order)


def test_nb_items_integration_with_build_matrices():
    """Test that generated samples work with build_matrices_fast"""
    from slotting_optimization.simulator import build_matrices_fast

    gen = DataGenerator()
    samples = gen.generate_samples(n_locations=12, nb_items=6, n_orders=20,
                                   min_items_per_order=1, max_items_per_order=3,
                                   n_samples=1, seed=555)
    ob, il, w = samples[0]

    # build_matrices_fast should work without errors
    loc_mat, seq_mat, item_loc_mat, locs, items = build_matrices_fast(ob, il, w)

    # Validate matrix shapes and properties
    assert loc_mat.shape == (14, 14)  # 12 locations + start + end
    assert item_loc_mat.shape[0] == len(items)
    assert item_loc_mat.shape[1] == 14
    # Each item should be assigned to exactly one location
    assert all(item_loc_mat.sum(axis=1) == 1)
