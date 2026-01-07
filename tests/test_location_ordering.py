"""Test suite for warehouse location ordering in matrix building functions."""

import numpy as np
import pytest

from slotting_optimization.models import Order
from slotting_optimization.warehouse import Warehouse
from slotting_optimization.order_book import OrderBook
from slotting_optimization.item_locations import ItemLocations
from slotting_optimization.simulator import build_matrices, build_matrices_fast


def make_order(order_id, item_id, ts):
    """Helper to create Order from dict."""
    return Order.from_dict({"order_id": order_id, "item_id": item_id, "timestamp": ts})


def test_location_ordering_basic():
    """Test basic location ordering: storage locations first, then start, then end."""
    # Create warehouse with 3 storage locations + start + end
    w = Warehouse(
        locations=["start", "L0", "L1", "L2", "end"],
        start_point_id="start",
        end_point_id="end"
    )
    # Set some distances
    for loc in ["L0", "L1", "L2"]:
        w.set_distance("start", loc, 5.0)
        w.set_distance(loc, "end", 3.0)
    w.set_distance("end", "start", 2.0)

    # Create minimal order book and item locations
    ob = OrderBook.from_orders([
        make_order("O1", "I1", "2024-01-01T10:00:00")
    ])
    il = ItemLocations.from_records([{"item_id": "I1", "location_id": "L0"}])

    # Build matrices
    loc_mat, seq_mat, item_loc_mat, locs, items = build_matrices_fast(ob, il, w)

    # Verify ordering: storage locations first, then start, then end
    assert locs == ["L0", "L1", "L2", "start", "end"]

    # Verify loc_mat indices match ordering
    start_idx = locs.index("start")
    end_idx = locs.index("end")
    l0_idx = locs.index("L0")

    assert start_idx == 3
    assert end_idx == 4
    assert l0_idx == 0

    # Verify distances are at correct indices
    assert loc_mat[start_idx, l0_idx] == 5.0
    assert loc_mat[l0_idx, end_idx] == 3.0


def test_location_ordering_preserves_storage_order():
    """Test that storage locations maintain their insertion order."""
    # Add storage locations in specific order: L2, L0, L1
    w = Warehouse(locations=[])
    w.add_location("L2")
    w.add_location("L0")
    w.add_location("L1")
    w.set_start_point("start")
    w.set_end_point("end")

    # Set minimal distances
    for loc in ["L2", "L0", "L1"]:
        w.set_distance("start", loc, 5.0)
        w.set_distance(loc, "end", 3.0)
    w.set_distance("end", "start", 2.0)

    ob = OrderBook.from_orders([
        make_order("O1", "I1", "2024-01-01T10:00:00")
    ])
    il = ItemLocations.from_records([{"item_id": "I1", "location_id": "L2"}])

    loc_mat, seq_mat, item_loc_mat, locs, items = build_matrices_fast(ob, il, w)

    # Verify storage order is preserved (L2, L0, L1 come before start/end)
    assert locs == ["L2", "L0", "L1", "start", "end"]


def test_location_ordering_matrix_slicing():
    """Test that matrix slicing works correctly with new ordering."""
    w = Warehouse(
        locations=["start", "A", "B", "C", "end"],
        start_point_id="start",
        end_point_id="end"
    )

    # Set distances with distinct values for testing
    storage_locs = ["A", "B", "C"]
    for i, loc1 in enumerate(storage_locs):
        w.set_distance("start", loc1, float(10 + i))  # 10, 11, 12
        w.set_distance(loc1, "end", float(20 + i))    # 20, 21, 22
        for j, loc2 in enumerate(storage_locs):
            if loc1 != loc2:
                w.set_distance(loc1, loc2, float(i * 10 + j))  # Unique values
    w.set_distance("end", "start", 99.0)

    ob = OrderBook.from_orders([
        make_order("O1", "I1", "2024-01-01T10:00:00")
    ])
    il = ItemLocations.from_records([{"item_id": "I1", "location_id": "A"}])

    loc_mat, seq_mat, item_loc_mat, locs, items = build_matrices_fast(ob, il, w)

    # Verify n_storage calculation
    n_storage = len(locs) - 2
    assert n_storage == 3

    # Test slicing: storage-only submatrix
    storage_mat = loc_mat[:n_storage, :n_storage]
    assert storage_mat.shape == (3, 3)
    # Diagonal should be 0
    assert storage_mat[0, 0] == 0.0

    # Test slicing: start→all distances
    start_distances = loc_mat[n_storage, :]
    assert start_distances[0] == 10.0  # start→A
    assert start_distances[1] == 11.0  # start→B
    assert start_distances[2] == 12.0  # start→C

    # Test slicing: end→all distances
    end_distances = loc_mat[n_storage + 1, :]
    assert end_distances[n_storage] == 99.0  # end→start


def test_location_ordering_missing_start_raises_error():
    """Test that missing start_point raises ValueError."""
    w = Warehouse(locations=["L0", "L1", "end"], start_point_id=None, end_point_id="end")

    ob = OrderBook.from_orders([
        make_order("O1", "I1", "2024-01-01T10:00:00")
    ])
    il = ItemLocations.from_records([{"item_id": "I1", "location_id": "L0"}])

    with pytest.raises(ValueError, match="must define start_point and end_point"):
        build_matrices_fast(ob, il, w)


def test_location_ordering_missing_end_raises_error():
    """Test that missing end_point raises ValueError."""
    w = Warehouse(locations=["start", "L0", "L1"], start_point_id="start", end_point_id=None)

    ob = OrderBook.from_orders([
        make_order("O1", "I1", "2024-01-01T10:00:00")
    ])
    il = ItemLocations.from_records([{"item_id": "I1", "location_id": "L0"}])

    with pytest.raises(ValueError, match="must define start_point and end_point"):
        build_matrices_fast(ob, il, w)


def test_location_ordering_start_equals_end_raises_error():
    """Test that start_point == end_point raises ValueError."""
    w = Warehouse(
        locations=["depot", "L0", "L1"],
        start_point_id="depot",
        end_point_id="depot"  # Same as start!
    )

    ob = OrderBook.from_orders([
        make_order("O1", "I1", "2024-01-01T10:00:00")
    ])
    il = ItemLocations.from_records([{"item_id": "I1", "location_id": "L0"}])

    with pytest.raises(ValueError, match="must be distinct"):
        build_matrices_fast(ob, il, w)


def test_location_ordering_consistency_between_functions():
    """Test that build_matrices and build_matrices_fast return same ordering."""
    w = Warehouse(
        locations=["start", "A", "B", "C", "end"],
        start_point_id="start",
        end_point_id="end"
    )

    # Set all required distances
    for loc in ["A", "B", "C"]:
        w.set_distance("start", loc, 5.0)
        w.set_distance(loc, "end", 3.0)
        for loc2 in ["A", "B", "C"]:
            if loc != loc2:
                w.set_distance(loc, loc2, 1.0)
    w.set_distance("end", "start", 2.0)

    ob = OrderBook.from_orders([
        make_order("O1", "I1", "2024-01-01T10:00:00"),
        make_order("O1", "I2", "2024-01-01T10:01:00")
    ])
    il = ItemLocations.from_records([
        {"item_id": "I1", "location_id": "A"},
        {"item_id": "I2", "location_id": "B"}
    ])

    # Build with both functions
    loc_mat1, seq_mat1, item_loc_mat1, locs1, items1 = build_matrices(ob, il, w)
    loc_mat2, seq_mat2, item_loc_mat2, locs2, items2 = build_matrices_fast(ob, il, w)

    # Verify locs ordering is identical
    assert locs1 == locs2
    assert locs1 == ["A", "B", "C", "start", "end"]

    # Verify items ordering is identical (should be sorted)
    assert items1 == items2

    # Verify matrix values at corresponding indices match
    assert np.allclose(loc_mat1, loc_mat2, equal_nan=True)
    assert np.array_equal(seq_mat1, seq_mat2)
    assert np.array_equal(item_loc_mat1, item_loc_mat2)


def test_location_ordering_block_matrix_construction():
    """Test block matrix construction with new ordering (notebook use case)."""
    w = Warehouse(
        locations=["start", "L0", "L1", "end"],
        start_point_id="start",
        end_point_id="end"
    )

    for loc in ["L0", "L1"]:
        w.set_distance("start", loc, 5.0)
        w.set_distance(loc, "end", 3.0)
    w.set_distance("L0", "L1", 1.0)
    w.set_distance("L1", "L0", 1.0)
    w.set_distance("end", "start", 2.0)

    ob = OrderBook.from_orders([
        make_order("O1", "I1", "2024-01-01T10:00:00"),
        make_order("O1", "I2", "2024-01-01T10:01:00")
    ])
    il = ItemLocations.from_records([
        {"item_id": "I1", "location_id": "L0"},
        {"item_id": "I2", "location_id": "L1"}
    ])

    loc_mat, seq_mat, item_loc_mat, locs, items = build_matrices_fast(ob, il, w)

    # Construct block matrix as in notebook
    block_matrix = np.block([
        [loc_mat, item_loc_mat.T],
        [item_loc_mat, seq_mat]
    ])

    # Verify dimensions
    L = len(locs)  # 4 (L0, L1, start, end)
    I = len(items)  # 2 (I1, I2)
    assert block_matrix.shape == (L + I, L + I)
    assert block_matrix.shape == (6, 6)

    # Verify we can slice storage-only block
    n_storage = L - 2  # 2
    storage_block = block_matrix[:n_storage, :n_storage]
    assert storage_block.shape == (2, 2)

    # Verify start/end rows are at predictable positions
    start_row = block_matrix[n_storage, :]  # Row 2
    end_row = block_matrix[n_storage + 1, :]  # Row 3
    assert start_row.shape == (6,)
    assert end_row.shape == (6,)


def test_location_ordering_no_storage_locations():
    """Test edge case where warehouse has only start and end (no storage)."""
    w = Warehouse(
        locations=["start", "end"],
        start_point_id="start",
        end_point_id="end"
    )
    w.set_distance("start", "end", 10.0)
    w.set_distance("end", "start", 10.0)

    # Create order with item at start location (edge case but valid)
    ob = OrderBook.from_orders([
        make_order("O1", "I1", "2024-01-01T10:00:00")
    ])
    il = ItemLocations.from_records([{"item_id": "I1", "location_id": "start"}])

    loc_mat, seq_mat, item_loc_mat, locs, items = build_matrices_fast(ob, il, w)

    # Verify ordering
    assert locs == ["start", "end"]

    # Verify n_storage calculation
    n_storage = len(locs) - 2
    assert n_storage == 0

    # Verify matrices have correct shape
    assert loc_mat.shape == (2, 2)
    assert item_loc_mat.shape == (1, 2)  # 1 item, 2 locations
