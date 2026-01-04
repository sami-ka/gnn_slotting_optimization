import copy

from slotting_optimization.generator import DataGenerator
from slotting_optimization.order_book import OrderBook
from slotting_optimization.item_locations import ItemLocations
from slotting_optimization.warehouse import Warehouse


def test_generate_shapes_and_types():
    gen = DataGenerator()
    samples = gen.generate_samples(
        n_locations=10, n_orders=20, min_items_per_order=1, max_items_per_order=3, n_samples=3, distances_fixed=True, seed=42
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
    samples_fixed = gen.generate_samples(5, 10, 1, 2, n_samples=2, distances_fixed=True, seed=1)
    samples_var = gen.generate_samples(5, 10, 1, 2, n_samples=2, distances_fixed=False, seed=1)

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
    a = gen.generate_samples(5, 5, 1, 2, n_samples=1, distances_fixed=False, seed=123)
    b = gen.generate_samples(5, 5, 1, 2, n_samples=1, distances_fixed=False, seed=123)

    # Compare warehouses and item locations and orderbooks
    ob_a, il_a, w_a = a[0]
    ob_b, il_b, w_b = b[0]

    # Compare orderbooks by sorted to_dicts for determinism
    assert sorted(ob_a.to_df().to_dicts(), key=lambda d: (d['order_id'], d['item_id'], d['timestamp'])) == sorted(ob_b.to_df().to_dicts(), key=lambda d: (d['order_id'], d['item_id'], d['timestamp']))
    assert il_a.to_dict() == il_b.to_dict()
    assert w_a._distance_map == w_b._distance_map


def test_min_max_items_range():
    gen = DataGenerator()
    samples = gen.generate_samples(5, 20, 2, 4, n_samples=1, distances_fixed=True, seed=5)
    ob, il, w = samples[0]
    # each logical order id should appear between min and max times
    df = ob.to_df()
    counts = df.group_by("order_id").len().get_column("len").to_list()
    for c in counts:
        assert 2 <= c <= 4
