from datetime import datetime
import pytest
import polars as pl

from slotting_optimization.order_book import OrderBook
from slotting_optimization.models import Order


def test_orderbook_from_dicts_direct_basic():
    """Test that from_dicts_direct produces correct OrderBook"""
    records = [
        {"order_id": "o1", "item_id": "sku1", "timestamp": datetime(2025, 1, 1, 12, 0, 0)},
        {"order_id": "o1", "item_id": "sku2", "timestamp": datetime(2025, 1, 1, 12, 0, 1)},
        {"order_id": "o2", "item_id": "sku1", "timestamp": datetime(2025, 1, 1, 12, 5, 0)},
    ]

    ob = OrderBook.from_dicts_direct(records)

    assert len(ob) == 3
    df = ob.to_df()

    # Check schema
    assert df.schema["order_id"] == pl.Utf8
    assert df.schema["item_id"] == pl.Utf8
    assert df.schema["timestamp"] == pl.Datetime

    # Verify data
    assert set(df["order_id"].to_list()) == {"o1", "o2"}
    assert set(df["item_id"].to_list()) == {"sku1", "sku2"}


def test_orderbook_from_dicts_direct_empty():
    """Test from_dicts_direct with empty list"""
    ob = OrderBook.from_dicts_direct([])

    assert len(ob) == 0
    df = ob.to_df()

    # Check schema even for empty DataFrame
    assert df.schema["order_id"] == pl.Utf8
    assert df.schema["item_id"] == pl.Utf8
    assert df.schema["timestamp"] == pl.Datetime


def test_orderbook_from_dicts_direct_equivalent_to_from_orders():
    """Test that from_dicts_direct is equivalent to from_orders()"""
    # Create via from_orders()
    orders = [
        Order(order_id="o1", item_id="sku1", timestamp=datetime(2025, 1, 1, 10, 0, 0)),
        Order(order_id="o2", item_id="sku2", timestamp=datetime(2025, 1, 1, 11, 0, 0)),
    ]
    ob1 = OrderBook.from_orders(orders)

    # Create via from_dicts_direct()
    records = [
        {"order_id": "o1", "item_id": "sku1", "timestamp": datetime(2025, 1, 1, 10, 0, 0)},
        {"order_id": "o2", "item_id": "sku2", "timestamp": datetime(2025, 1, 1, 11, 0, 0)},
    ]
    ob2 = OrderBook.from_dicts_direct(records)

    # Compare dataframes
    df1 = ob1.to_df().sort("order_id")
    df2 = ob2.to_df().sort("order_id")
    assert df1.equals(df2)


def test_orderbook_from_dicts_direct_preserves_order():
    """Test that from_dicts_direct preserves record order"""
    records = [
        {"order_id": "o3", "item_id": "sku3", "timestamp": datetime(2025, 1, 3, 10, 0, 0)},
        {"order_id": "o1", "item_id": "sku1", "timestamp": datetime(2025, 1, 1, 10, 0, 0)},
        {"order_id": "o2", "item_id": "sku2", "timestamp": datetime(2025, 1, 2, 10, 0, 0)},
    ]

    ob = OrderBook.from_dicts_direct(records)
    df = ob.to_df()

    # Order should be preserved (not sorted)
    order_ids = df["order_id"].to_list()
    assert order_ids == ["o3", "o1", "o2"]


def test_orderbook_from_dicts_direct_handles_microseconds():
    """Test that from_dicts_direct preserves datetime precision"""
    records = [
        {"order_id": "o1", "item_id": "sku1",
         "timestamp": datetime(2025, 1, 1, 10, 0, 0, 123456)},
        {"order_id": "o2", "item_id": "sku2",
         "timestamp": datetime(2025, 1, 1, 10, 0, 0, 654321)},
    ]

    ob = OrderBook.from_dicts_direct(records)
    df = ob.to_df()

    # Verify microseconds are preserved
    timestamps = df["timestamp"].to_list()
    assert timestamps[0].microsecond == 123456
    assert timestamps[1].microsecond == 654321


def test_orderbook_from_dicts_direct_type_casting():
    """Test that from_dicts_direct casts types correctly"""
    records = [
        # Pass integers for IDs (should be cast to strings)
        {"order_id": 123, "item_id": 456, "timestamp": datetime(2025, 1, 1, 10, 0, 0)},
    ]

    ob = OrderBook.from_dicts_direct(records)
    df = ob.to_df()

    # IDs should be strings
    assert df["order_id"][0] == "123"
    assert df["item_id"][0] == "456"
    assert df.schema["order_id"] == pl.Utf8
    assert df.schema["item_id"] == pl.Utf8


def test_orderbook_from_dicts_direct_large_batch():
    """Test from_dicts_direct with large number of records"""
    from datetime import timedelta

    n_records = 10000
    base_time = datetime(2025, 1, 1, 10, 0, 0)
    records = [
        {
            "order_id": f"o{i // 5}",  # 5 items per order
            "item_id": f"sku{i % 100}",
            "timestamp": base_time + timedelta(seconds=i)
        }
        for i in range(n_records)
    ]

    ob = OrderBook.from_dicts_direct(records)

    assert len(ob) == n_records
    df = ob.to_df()

    # Verify schema
    assert df.schema["order_id"] == pl.Utf8
    assert df.schema["item_id"] == pl.Utf8
    assert df.schema["timestamp"] == pl.Datetime

    # Verify unique orders (should have 2000 unique orders)
    unique_orders = df["order_id"].n_unique()
    assert unique_orders == 2000


def test_orderbook_from_dicts_direct_duplicate_records():
    """Test from_dicts_direct with duplicate records (should be preserved)"""
    records = [
        {"order_id": "o1", "item_id": "sku1", "timestamp": datetime(2025, 1, 1, 10, 0, 0)},
        {"order_id": "o1", "item_id": "sku1", "timestamp": datetime(2025, 1, 1, 10, 0, 0)},
        {"order_id": "o1", "item_id": "sku1", "timestamp": datetime(2025, 1, 1, 10, 0, 0)},
    ]

    ob = OrderBook.from_dicts_direct(records)

    # All three duplicate records should be preserved
    assert len(ob) == 3


def test_orderbook_from_dicts_direct_with_filter_by_time():
    """Test that from_dicts_direct works with filter_by_time()"""
    records = [
        {"order_id": "o1", "item_id": "sku1", "timestamp": datetime(2025, 1, 1, 10, 0, 0)},
        {"order_id": "o2", "item_id": "sku2", "timestamp": datetime(2025, 1, 1, 12, 0, 0)},
        {"order_id": "o3", "item_id": "sku3", "timestamp": datetime(2025, 1, 1, 14, 0, 0)},
    ]

    ob = OrderBook.from_dicts_direct(records)

    # Filter for records between 11:00 and 13:00
    filtered = ob.filter_by_time(
        start=datetime(2025, 1, 1, 11, 0, 0),
        end=datetime(2025, 1, 1, 13, 0, 0)
    )

    # Should only get o2
    assert len(filtered) == 1
    assert filtered["order_id"][0] == "o2"


def test_orderbook_from_dicts_direct_integration_with_simulator():
    """Test that OrderBook created with from_dicts_direct works with Simulator"""
    from slotting_optimization.item_locations import ItemLocations
    from slotting_optimization.warehouse import Warehouse
    from slotting_optimization.simulator import Simulator

    # Create test data
    records = [
        {"order_id": "o1", "item_id": "sku1", "timestamp": datetime(2025, 1, 1, 10, 0, 0)},
        {"order_id": "o1", "item_id": "sku2", "timestamp": datetime(2025, 1, 1, 10, 0, 1)},
        {"order_id": "o2", "item_id": "sku1", "timestamp": datetime(2025, 1, 1, 10, 1, 0)},
    ]

    ob = OrderBook.from_dicts_direct(records)

    # Create warehouse and item locations
    il = ItemLocations.from_records([
        {"item_id": "sku1", "location_id": "L0"},
        {"item_id": "sku2", "location_id": "L1"},
    ])

    w = Warehouse(locations=["start", "end", "L0", "L1"],
                  start_point_id="start", end_point_id="end")
    w.set_distance("start", "L0", 1.0)
    w.set_distance("L0", "end", 1.0)
    w.set_distance("start", "L1", 2.0)
    w.set_distance("L1", "end", 2.0)
    w.set_distance("end", "start", 0.5)

    # Simulator should work without errors
    sim = Simulator()
    total_dist, per_order = sim.simulate(ob, w, il)

    assert total_dist > 0
    assert len(per_order) == 3  # 3 order lines
