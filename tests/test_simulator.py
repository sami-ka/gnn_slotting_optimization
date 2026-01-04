from datetime import datetime

from slotting_optimization.models import Order
from slotting_optimization.item_locations import ItemLocations
from slotting_optimization.order_book import OrderBook
from slotting_optimization.warehouse import Warehouse
from slotting_optimization.simulator import Simulator


def test_simulator_single_and_multiple_orders():
    # Setup warehouse
    w = Warehouse(locations=["start", "L1", "end"], start_point_id="start", end_point_id="end")
    w.set_distance("start", "L1", 5.0)
    w.set_distance("L1", "end", 3.0)
    w.set_distance("end", "start", 2.0)

    # Item locations
    il = ItemLocations.from_records([{"item_id": "sku1", "location_id": "L1"}])

    # Orders: two sequential orders
    o1 = Order.from_dict({"order_id": "o1", "item_id": "sku1", "timestamp": "2026-01-01T08:30:00"})
    o2 = Order.from_dict({"order_id": "o2", "item_id": "sku1", "timestamp": "2026-01-01T09:00:00"})
    ob = OrderBook.from_orders([o1, o2])

    sim = Simulator()
    total, per_order = sim.simulate(ob, w, il)

    # Each order: start->L1 (5) + L1->end (3) = 8
    # Between orders: end->start = 2
    assert per_order == [8.0, 8.0]
    assert total == 8.0 + 2.0 + 8.0


def test_simulator_order_sorting():
    w = Warehouse(locations=["start", "L1", "end"], start_point_id="start", end_point_id="end")
    w.set_distance("start", "L1", 1.0)
    w.set_distance("L1", "end", 1.0)
    w.set_distance("end", "start", 1.0)

    il = ItemLocations.from_records([{"item_id": "sku1", "location_id": "L1"}])

    # Provide orders out-of-order; simulator should sort by timestamp
    o1 = Order.from_dict({"order_id": "o1", "item_id": "sku1", "timestamp": "2026-01-01T09:00:00"})
    o2 = Order.from_dict({"order_id": "o2", "item_id": "sku1", "timestamp": "2026-01-01T08:00:00"})
    ob = OrderBook.from_orders([o1, o2])

    sim = Simulator()
    total, per_order = sim.simulate(ob, w, il)

    # Each order distance = 1 + 1 = 2; plus between orders 1
    assert per_order == [2.0, 2.0]
    assert total == 2.0 + 1.0 + 2.0


def test_simulator_missing_distances_raise():
    w = Warehouse(locations=["start", "L1", "end"], start_point_id="start", end_point_id="end")
    # Intentionally leave out end->start distance
    w.set_distance("start", "L1", 5.0)
    w.set_distance("L1", "end", 3.0)

    il = ItemLocations.from_records([{"item_id": "sku1", "location_id": "L1"}])
    o1 = Order.from_dict({"order_id": "o1", "item_id": "sku1", "timestamp": "2026-01-01T08:30:00"})
    o2 = Order.from_dict({"order_id": "o2", "item_id": "sku1", "timestamp": "2026-01-01T09:00:00"})
    ob = OrderBook.from_orders([o1, o2])

    sim = Simulator()
    try:
        sim.simulate(ob, w, il)
        assert False, "Expected ValueError due to missing return distance"
    except ValueError as e:
        assert "Missing distance from end to start" in str(e)
