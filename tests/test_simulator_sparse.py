from slotting_optimization.models import Order
from slotting_optimization.item_locations import ItemLocations
from slotting_optimization.order_book import OrderBook
from slotting_optimization.warehouse import Warehouse
from slotting_optimization.simulator import Simulator


def test_simulator_sparse_matches_other_methods():
    w = Warehouse(locations=["start", "L1", "end"], start_point_id="start", end_point_id="end")
    w.set_distance("start", "L1", 5.0)
    w.set_distance("L1", "end", 3.0)
    w.set_distance("end", "start", 2.0)

    il = ItemLocations.from_records([{"item_id": "sku1", "location_id": "L1"}])

    o1 = Order.from_dict({"order_id": "o1", "item_id": "sku1", "timestamp": "2026-01-01T08:30:00"})
    o2 = Order.from_dict({"order_id": "o2", "item_id": "sku1", "timestamp": "2026-01-01T09:00:00"})
    ob = OrderBook.from_orders([o1, o2])

    sim = Simulator()
    a_total, a_per = sim.simulate(ob, w, il)
    s_total, s_per = sim.simulate_sparse_matrix(ob, w, il)

    assert abs(a_total - s_total) < 1e-9
    assert a_per == s_per
