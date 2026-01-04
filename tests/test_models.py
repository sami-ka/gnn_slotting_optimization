from datetime import datetime
from pathlib import Path

import polars as pl

from slotting_optimization.models import Order
from slotting_optimization.item_locations import ItemLocations
from slotting_optimization.order_book import OrderBook
from slotting_optimization.warehouse import Warehouse


DATA_DIR = Path(__file__).parent / ".." / "slotting_optimization" / "data"
SAMPLE_ORDERS = (DATA_DIR / "sample_orders.csv").resolve()
SAMPLE_LOCATIONS = (DATA_DIR / "sample_item_locations.csv").resolve()


def test_order_parse_and_to_dict():
    d = {"order_id": "o1", "item_id": "sku1", "timestamp": "2026-01-01T08:30:00"}
    o = Order.from_dict(d)
    assert isinstance(o.timestamp, datetime)
    assert o.order_id == "o1"
    assert o.item_id == "sku1"
    assert o.to_dict()["timestamp"] == "2026-01-01T08:30:00"


def test_itemlocations_load():
    il = ItemLocations.load_csv(str(SAMPLE_LOCATIONS))
    assert len(il) == 3
    assert il.get_location("sku1") == "A1"
    assert il.get_location("nonexistent") is None


def test_orderbook_load_and_filter_and_add():
    ob = OrderBook.load_csv(str(SAMPLE_ORDERS))
    assert len(ob) == 4

    start = datetime.fromisoformat("2026-01-01T08:30:00")
    end = datetime.fromisoformat("2026-01-01T09:00:00")
    filtered = ob.filter_by_time(start, end)
    # orders o1 (08:30), o2 (08:35), o3 (09:00) => 3 rows
    assert filtered.shape[0] == 3

    # Test adding an Order
    o = Order.from_dict({"order_id": "o5", "item_id": "sku2", "timestamp": "2026-01-03T11:00:00"})
    ob.add(o)
    assert len(ob) == 5


def test_warehouse_basic():
    w = Warehouse(locations=["A1", "B2"], start_point_id="A1")
    assert w.start_point == "A1"
    assert w.location_exists("B2")
    w.set_distance("A1", "B2", 12.5)
    assert w.get_distance("A1", "B2") == 12.5
