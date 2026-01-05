from datetime import datetime

import numpy as np

from slotting_optimization.models import Order
from slotting_optimization.order_book import OrderBook
from slotting_optimization.item_locations import ItemLocations
from slotting_optimization.warehouse import Warehouse
from slotting_optimization.simulator import build_matrices


def make_order(order_id, item_id, ts):
    return Order.from_dict({"order_id": order_id, "item_id": item_id, "timestamp": ts})


def test_build_matrices_basic():
    # Warehouse with 4 locations
    w = Warehouse(locations=["start", "A", "B", "end"], start_point_id="start", end_point_id="end")
    w.set_distance("start", "A", 1.0)
    w.set_distance("A", "end", 1.5)
    w.set_distance("start", "B", 2.0)
    w.set_distance("B", "end", 2.5)
    w.set_distance("end", "start", 3.0)

    # Item locations
    il = ItemLocations.from_records([
        {"item_id": "sku1", "location_id": "A"},
        {"item_id": "sku2", "location_id": "B"},
        {"item_id": "sku3", "location_id": "A"},
    ])

    # Orders: two orders with sequences
    orders = [
        make_order("o1", "sku1", "2025-01-01T00:00:00"),
        make_order("o1", "sku2", "2025-01-01T00:01:00"),
        make_order("o1", "sku3", "2025-01-01T00:02:00"),
        make_order("o2", "sku2", "2025-01-02T00:00:00"),
        make_order("o2", "sku1", "2025-01-02T00:01:00"),
    ]
    ob = OrderBook.from_orders(orders)

    loc_mat, seq_mat, item_loc_mat, locs, items = build_matrices(ob, il, w)

    # Location matrix checks
    assert loc_mat.shape == (4, 4)
    assert locs == ["start", "A", "B", "end"]
    # start -> A == 1.0
    assert np.isclose(loc_mat[locs.index("start"), locs.index("A")], 1.0)
    # A -> B is missing -> nan
    assert np.isnan(loc_mat[locs.index("A"), locs.index("B")])

    # Item sequence matrix checks
    assert seq_mat.shape == (3, 3)
    # Items sorted alphabetically
    assert items == ["sku1", "sku2", "sku3"]

    # Expected sequences: sku1->sku2 (1), sku2->sku3 (1), sku2->sku1 (1)
    expected_seq = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=np.int64)
    np.testing.assert_array_equal(seq_mat, expected_seq)

    # Item-location matrix checks
    assert item_loc_mat.shape == (3, 4)
    # sku1 in A
    assert item_loc_mat[items.index("sku1"), locs.index("A")] == 1
    # sku2 in B
    assert item_loc_mat[items.index("sku2"), locs.index("B")] == 1
    # sku3 in A
    assert item_loc_mat[items.index("sku3"), locs.index("A")] == 1
