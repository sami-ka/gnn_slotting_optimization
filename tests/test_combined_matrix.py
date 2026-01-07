"""Test suite for combined matrix function."""

import numpy as np
import pytest

from slotting_optimization.models import Order
from slotting_optimization.warehouse import Warehouse
from slotting_optimization.order_book import OrderBook
from slotting_optimization.item_locations import ItemLocations
from slotting_optimization.simulator import (
    build_combined_matrix,
    extract_submatrices,
    build_matrices,
    build_matrices_fast
)


def make_order(order_id, item_id, ts):
    """Helper to create Order from dict."""
    return Order.from_dict({"order_id": order_id, "item_id": item_id, "timestamp": ts})


def test_combined_matrix_basic_structure():
    """Test basic combined matrix structure with 2 storage + start + end (4 locs), 2 items."""
    # Create warehouse with 2 storage locations + start + end
    w = Warehouse(
        locations=["L0", "L1", "start", "end"],
        start_point_id="start",
        end_point_id="end"
    )

    # Set distances
    for loc in ["L0", "L1"]:
        w.set_distance("start", loc, 5.0)
        w.set_distance(loc, "end", 3.0)
    w.set_distance("L0", "L1", 1.0)
    w.set_distance("L1", "L0", 1.0)
    w.set_distance("end", "start", 2.0)

    # Create orders with 2 items
    ob = OrderBook.from_orders([
        make_order("O1", "I1", "2024-01-01T10:00:00"),
        make_order("O1", "I2", "2024-01-01T10:01:00")
    ])

    # Assign items to storage locations
    il = ItemLocations.from_records([
        {"item_id": "I1", "location_id": "L0"},
        {"item_id": "I2", "location_id": "L1"}
    ])

    # Build combined matrix
    combined, metadata = build_combined_matrix(ob, il, w)

    # Verify shape: (2 items + 4 locs, 2 items + 4 locs) = (6, 6)
    assert combined.shape == (6, 6)

    # Extract quadrants using metadata
    I = metadata['n_items']  # 2
    L = metadata['n_locs']    # 4

    # Verify each quadrant has correct shape
    item_loc_quadrant = combined[:I, :L]  # Top-left
    assert item_loc_quadrant.shape == (2, 4)

    seq_quadrant = combined[:I, L:]  # Top-right
    assert seq_quadrant.shape == (2, 2)

    loc_quadrant = combined[I:, :L]  # Bottom-left
    assert loc_quadrant.shape == (4, 4)

    zeros_quadrant = combined[I:, L:]  # Bottom-right
    assert zeros_quadrant.shape == (4, 2)

    # Verify bottom-right is all zeros
    assert np.all(zeros_quadrant == 0)


def test_combined_matrix_start_end_zeroed():
    """Test that start and end location columns are zeroed in item_loc_mat."""
    w = Warehouse(
        locations=["A", "B", "start", "end"],
        start_point_id="start",
        end_point_id="end"
    )

    for loc in ["A", "B"]:
        w.set_distance("start", loc, 5.0)
        w.set_distance(loc, "end", 3.0)
    w.set_distance("A", "B", 1.0)
    w.set_distance("B", "A", 1.0)
    w.set_distance("end", "start", 2.0)

    ob = OrderBook.from_orders([
        make_order("O1", "I1", "2024-01-01T10:00:00")
    ])

    il = ItemLocations.from_records([
        {"item_id": "I1", "location_id": "A"}
    ])

    combined, metadata = build_combined_matrix(ob, il, w)

    # Get item_loc_mat quadrant (top-left)
    I = metadata['n_items']
    L = metadata['n_locs']
    n_storage = metadata['n_storage']

    item_loc_quadrant = combined[:I, :L]

    # Verify start and end columns (last 2 columns of item_loc_mat) are all zeros
    start_col = item_loc_quadrant[:, n_storage]      # Column for start
    end_col = item_loc_quadrant[:, n_storage + 1]    # Column for end

    assert np.all(start_col == 0), "Start location column should be all zeros"
    assert np.all(end_col == 0), "End location column should be all zeros"

    # Verify storage location columns can have non-zero values
    storage_cols = item_loc_quadrant[:, :n_storage]
    assert np.any(storage_cols == 1), "Storage location columns should have item assignments"


def test_combined_matrix_metadata():
    """Test that metadata dict contains all expected keys and correct values."""
    w = Warehouse(
        locations=["L0", "L1", "L2", "start", "end"],
        start_point_id="start",
        end_point_id="end"
    )

    for loc in ["L0", "L1", "L2"]:
        w.set_distance("start", loc, 5.0)
        w.set_distance(loc, "end", 3.0)
    w.set_distance("end", "start", 2.0)

    ob = OrderBook.from_orders([
        make_order("O1", "I1", "2024-01-01T10:00:00"),
        make_order("O1", "I2", "2024-01-01T10:01:00"),
        make_order("O1", "I3", "2024-01-01T10:02:00")
    ])

    il = ItemLocations.from_records([
        {"item_id": "I1", "location_id": "L0"},
        {"item_id": "I2", "location_id": "L1"},
        {"item_id": "I3", "location_id": "L2"}
    ])

    combined, metadata = build_combined_matrix(ob, il, w)

    # Verify all required keys exist
    required_keys = [
        'n_items', 'n_locs', 'n_storage',
        'items_slice', 'locs_slice', 'storage_slice',
        'start_idx', 'end_idx', 'locs', 'items'
    ]
    for key in required_keys:
        assert key in metadata, f"Metadata missing key: {key}"

    # Verify values
    assert metadata['n_items'] == 3
    assert metadata['n_locs'] == 5
    assert metadata['n_storage'] == 3

    # Verify slice objects
    assert metadata['items_slice'] == slice(0, 3)
    assert metadata['locs_slice'] == slice(3, 8)
    assert metadata['storage_slice'] == slice(0, 3)

    # Verify start and end indices
    assert metadata['start_idx'] == 3  # After 3 storage locations
    assert metadata['end_idx'] == 4    # After start

    # Verify locs and items lists
    assert metadata['locs'] == ["L0", "L1", "L2", "start", "end"]
    assert sorted(metadata['items']) == ["I1", "I2", "I3"]


def test_combined_matrix_slicing_with_metadata():
    """Test that metadata slices correctly extract each quadrant."""
    w = Warehouse(
        locations=["A", "B", "start", "end"],
        start_point_id="start",
        end_point_id="end"
    )

    for loc in ["A", "B"]:
        w.set_distance("start", loc, 5.0)
        w.set_distance(loc, "end", 3.0)
    w.set_distance("A", "B", 1.0)
    w.set_distance("B", "A", 1.0)
    w.set_distance("end", "start", 2.0)

    ob = OrderBook.from_orders([
        make_order("O1", "X", "2024-01-01T10:00:00"),
        make_order("O1", "Y", "2024-01-01T10:01:00")
    ])

    il = ItemLocations.from_records([
        {"item_id": "X", "location_id": "A"},
        {"item_id": "Y", "location_id": "B"}
    ])

    combined, metadata = build_combined_matrix(ob, il, w)

    # Extract quadrants using metadata
    items_slice = metadata['items_slice']
    locs_slice = metadata['locs_slice']

    # Extract item rows
    item_rows = combined[items_slice, :]
    assert item_rows.shape[0] == metadata['n_items']

    # Extract location rows
    loc_rows = combined[locs_slice, :]
    assert loc_rows.shape[0] == metadata['n_locs']

    # Use extract_submatrices helper
    submatrices = extract_submatrices(combined, metadata)

    assert submatrices['item_loc_mat'].shape == (2, 4)
    assert submatrices['seq_mat'].shape == (2, 2)
    assert submatrices['loc_mat'].shape == (4, 4)
    assert submatrices['zeros'].shape == (4, 2)


def test_combined_matrix_consistency_with_both_build_functions():
    """Test that results are identical whether using build_matrices or build_matrices_fast."""
    w = Warehouse(
        locations=["L0", "L1", "start", "end"],
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

    # Build with both methods
    combined_fast, meta_fast = build_combined_matrix(ob, il, w, use_fast=True)
    combined_slow, meta_slow = build_combined_matrix(ob, il, w, use_fast=False)

    # Verify matrices are identical
    assert np.allclose(combined_fast, combined_slow, equal_nan=True)

    # Verify metadata matches
    assert meta_fast['n_items'] == meta_slow['n_items']
    assert meta_fast['n_locs'] == meta_slow['n_locs']
    assert meta_fast['n_storage'] == meta_slow['n_storage']
    assert meta_fast['locs'] == meta_slow['locs']
    assert meta_fast['items'] == meta_slow['items']


def test_combined_matrix_edge_case_no_items():
    """Test edge case where order book has no items."""
    w = Warehouse(
        locations=["L0", "L1", "start", "end"],
        start_point_id="start",
        end_point_id="end"
    )

    for loc in ["L0", "L1"]:
        w.set_distance("start", loc, 5.0)
        w.set_distance(loc, "end", 3.0)
    w.set_distance("L0", "L1", 1.0)
    w.set_distance("L1", "L0", 1.0)
    w.set_distance("end", "start", 2.0)

    # Empty order book
    ob = OrderBook.from_orders([])

    # Empty item locations
    il = ItemLocations.from_records([])

    combined, metadata = build_combined_matrix(ob, il, w)

    # With no items, matrix should be (L, L)
    L = metadata['n_locs']
    assert combined.shape == (L, L)
    assert metadata['n_items'] == 0

    # Should consist only of loc_mat (bottom-left quadrant fills entire matrix)
    # No item rows, so the entire matrix is just loc_mat
    assert combined.shape == (4, 4)


def test_combined_matrix_values_preservation():
    """Test that actual values from original matrices are preserved in correct positions."""
    w = Warehouse(
        locations=["A", "B", "start", "end"],
        start_point_id="start",
        end_point_id="end"
    )

    # Set specific distances for testing
    w.set_distance("start", "A", 10.0)
    w.set_distance("start", "B", 11.0)
    w.set_distance("A", "end", 20.0)
    w.set_distance("B", "end", 21.0)
    w.set_distance("A", "B", 1.5)
    w.set_distance("B", "A", 2.5)
    w.set_distance("end", "start", 99.0)

    ob = OrderBook.from_orders([
        make_order("O1", "X", "2024-01-01T10:00:00"),
        make_order("O1", "Y", "2024-01-01T10:01:00")
    ])

    il = ItemLocations.from_records([
        {"item_id": "X", "location_id": "A"},
        {"item_id": "Y", "location_id": "B"}
    ])

    # Get original matrices
    loc_mat_orig, seq_mat_orig, item_loc_mat_orig, locs, items = build_matrices_fast(ob, il, w)

    # Build combined matrix
    combined, metadata = build_combined_matrix(ob, il, w)

    # Extract submatrices
    submat = extract_submatrices(combined, metadata)

    # Verify loc_mat values are preserved
    assert np.allclose(submat['loc_mat'], loc_mat_orig, equal_nan=True)

    # Verify seq_mat values are preserved
    assert np.array_equal(submat['seq_mat'], seq_mat_orig)

    # Verify item_loc_mat values are preserved (except start/end columns which are zeroed)
    n_storage = metadata['n_storage']
    # Check only storage columns
    original_storage_cols = item_loc_mat_orig[:, :n_storage]
    combined_storage_cols = submat['item_loc_mat'][:, :n_storage]
    assert np.array_equal(combined_storage_cols, original_storage_cols)

    # Verify specific distance values appear in loc_mat portion
    I = metadata['n_items']
    L = metadata['n_locs']

    # Find index of location "A" and "B" in locs
    a_idx = locs.index("A")
    b_idx = locs.index("B")

    # Check specific distances in combined matrix (loc_mat portion)
    loc_mat_portion = combined[I:, :L]
    assert loc_mat_portion[a_idx, b_idx] == 1.5
    assert loc_mat_portion[b_idx, a_idx] == 2.5

    # Check sequence count (X -> Y) in seq_mat portion
    x_idx = items.index("X")
    y_idx = items.index("Y")
    seq_mat_portion = combined[:I, L:]
    assert seq_mat_portion[x_idx, y_idx] == 1  # X followed by Y in order O1
