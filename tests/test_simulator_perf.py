import random
import time

from slotting_optimization.models import Order
from slotting_optimization.item_locations import ItemLocations
from slotting_optimization.order_book import OrderBook
from slotting_optimization.warehouse import Warehouse
from slotting_optimization.simulator import Simulator


def test_simulator_large_scale():
    random.seed(42)
    LOCATIONS = 1000
    ORDERS = 5000

    # Build warehouse with many locations
    locations = [f"L{i}" for i in range(LOCATIONS)]
    w = Warehouse(locations=["start", "end"] + locations, start_point_id="start", end_point_id="end")

    # Distances: start->Li and Li->end, and end->start
    for i, loc in enumerate(locations):
        w.set_distance("start", loc, 1.0 + (i % 10) * 0.1)
        w.set_distance(loc, "end", 0.5 + (i % 7) * 0.07)
    w.set_distance("end", "start", 2.5)

    # Item locations: map sku{i} -> Li
    records = [{"item_id": f"sku{i}", "location_id": locations[i]} for i in range(LOCATIONS)]
    il = ItemLocations.from_records(records)

    # Generate many orders, random distribution over the skus
    orders = []
    import datetime as _dt
    base_ts = int(_dt.datetime.fromisoformat("2026-01-01T00:00:00").timestamp())
    for i in range(ORDERS):
        item_idx = random.randrange(LOCATIONS)
        ts = base_ts + i  # seconds since epoch
        orders.append(Order.from_dict({"order_id": f"o{i}", "item_id": f"sku{item_idx}", "timestamp": ts}))

    ob = OrderBook.from_orders(orders)

    sim = Simulator()

    t0 = time.perf_counter()
    total_a, per_order_a = sim.simulate(ob, w, il)
    dur_a = time.perf_counter() - t0

    t0 = time.perf_counter()
    total_b, per_order_b = sim.simulate_sparse_matrix(ob, w, il)
    dur_b = time.perf_counter() - t0

    # Basic correctness checks
    assert len(per_order_a) == ORDERS
    assert len(per_order_b) == ORDERS
    assert abs(total_a - total_b) < 1e-9
    for x, y in zip(per_order_a, per_order_b):
        assert abs(x - y) < 1e-9

    # Each per_order entry should be start->loc + loc->end for that order
    expected_total = sum(per_order_a) + (ORDERS - 1) * 2.5
    assert abs(total_a - expected_total) < 1e-6

    # Print duration comparison for informational purposes
    faster = "sparse" if dur_b < dur_a else "baseline"
    print(f"Baseline duration={dur_a:.3f}s, sparse duration={dur_b:.3f}s => faster: {faster}")
