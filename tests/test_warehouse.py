import pytest

from slotting_optimization.warehouse import Warehouse


def test_warehouse_set_distances_bulk_basic():
    """Test that set_distances_bulk produces same result as individual calls"""
    w1 = Warehouse(locations=["start", "end"], start_point_id="start", end_point_id="end")
    w2 = Warehouse(locations=["start", "end"], start_point_id="start", end_point_id="end")

    dist_map = {("start", "L0"): 1.5, ("L0", "end"): 2.0, ("start", "L1"): 3.0}

    # Method 1: Individual calls
    for (a, b), d in dist_map.items():
        w1.set_distance(a, b, d)

    # Method 2: Bulk call
    w2.set_distances_bulk(dist_map)

    # Verify equivalence
    assert set(w1._locations) == set(w2._locations)
    assert w1._distance_map == w2._distance_map


def test_warehouse_bulk_adds_new_locations():
    """Test that bulk method adds locations not in initial list"""
    w = Warehouse(locations=["start"], start_point_id="start", end_point_id="end")

    dist_map = {("start", "L0"): 1.0, ("L0", "L1"): 2.0}
    w.set_distances_bulk(dist_map)

    assert "L0" in w._locations
    assert "L1" in w._locations
    assert w.get_distance("start", "L0") == 1.0
    assert w.get_distance("L0", "L1") == 2.0


def test_warehouse_bulk_empty_map():
    """Test bulk method with empty distance map"""
    w = Warehouse(locations=["start", "end"], start_point_id="start", end_point_id="end")
    initial_locations = set(w._locations)

    w.set_distances_bulk({})

    # Locations should be unchanged
    assert set(w._locations) == initial_locations
    # No distances should be set
    assert len(w._distance_map) == 0


def test_warehouse_bulk_with_existing_locations():
    """Test bulk method when locations already exist"""
    w = Warehouse(locations=["start", "end", "L0", "L1", "L2"],
                  start_point_id="start", end_point_id="end")

    dist_map = {
        ("start", "L0"): 1.0,
        ("L0", "L1"): 2.0,
        ("L1", "L2"): 3.0,
        ("L2", "end"): 4.0,
    }
    w.set_distances_bulk(dist_map)

    # All distances should be set correctly
    assert w.get_distance("start", "L0") == 1.0
    assert w.get_distance("L0", "L1") == 2.0
    assert w.get_distance("L1", "L2") == 3.0
    assert w.get_distance("L2", "end") == 4.0

    # No duplicate locations should be added
    location_counts = {}
    for loc in w._locations:
        location_counts[loc] = location_counts.get(loc, 0) + 1
    assert all(count == 1 for count in location_counts.values()), "Duplicate locations detected"


def test_warehouse_bulk_deterministic_ordering():
    """Test that bulk method produces deterministic location ordering"""
    # Create two warehouses with same initial state
    w1 = Warehouse(locations=["start", "end"], start_point_id="start", end_point_id="end")
    w2 = Warehouse(locations=["start", "end"], start_point_id="start", end_point_id="end")

    # Same distance map but test multiple times
    dist_map = {
        ("start", "L5"): 1.0,
        ("L2", "L3"): 2.0,
        ("L1", "L4"): 3.0,
        ("L0", "end"): 4.0,
    }

    w1.set_distances_bulk(dist_map)
    w2.set_distances_bulk(dist_map)

    # Location lists should be identical (same order)
    assert w1._locations == w2._locations


def test_warehouse_bulk_large_distance_map():
    """Test bulk method with large distance map"""
    n_locs = 100
    locations = [f"L{i}" for i in range(n_locs)]

    w = Warehouse(locations=["start", "end"], start_point_id="start", end_point_id="end")

    # Create full directed graph
    dist_map = {}
    nodes = ["start", "end"] + locations
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if a != b:
                dist_map[(a, b)] = float(i + j + 1)

    w.set_distances_bulk(dist_map)

    # Verify all distances are set
    assert len(w._distance_map) == len(dist_map)

    # Verify all locations are present
    assert set(w._locations) == set(nodes)

    # Spot check some distances
    assert w.get_distance("start", "L0") == 3.0  # i=0 (start), j=2 (L0): 0+2+1=3
    assert w.get_distance("L0", "L1") == 6.0     # i=2 (L0), j=3 (L1): 2+3+1=6


def test_warehouse_bulk_mixed_with_individual():
    """Test mixing set_distance() and set_distances_bulk()"""
    w = Warehouse(locations=["start"], start_point_id="start", end_point_id="end")

    # Use bulk API first
    w.set_distances_bulk({("start", "L0"): 1.0, ("L0", "L1"): 2.0})

    # Then use individual API
    w.set_distance("L1", "end", 3.0)
    w.set_distance("L0", "L2", 4.0)

    # Both should work
    assert w.get_distance("start", "L0") == 1.0
    assert w.get_distance("L0", "L1") == 2.0
    assert w.get_distance("L1", "end") == 3.0
    assert w.get_distance("L0", "L2") == 4.0


def test_warehouse_bulk_overwrite_existing_distance():
    """Test that bulk method can overwrite existing distances"""
    w = Warehouse(locations=["start", "L0", "end"],
                  start_point_id="start", end_point_id="end")

    # Set initial distance
    w.set_distance("start", "L0", 10.0)
    assert w.get_distance("start", "L0") == 10.0

    # Overwrite with bulk method
    w.set_distances_bulk({("start", "L0"): 5.0})
    assert w.get_distance("start", "L0") == 5.0


def test_warehouse_bulk_float_conversion():
    """Test that bulk method converts distances to float"""
    w = Warehouse(locations=["start", "end"], start_point_id="start", end_point_id="end")

    # Pass integers and ensure they're converted to float
    dist_map = {("start", "L0"): 1, ("L0", "end"): 2}
    w.set_distances_bulk(dist_map)

    # Verify distances are stored as floats
    assert isinstance(w.get_distance("start", "L0"), float)
    assert isinstance(w.get_distance("L0", "end"), float)
    assert w.get_distance("start", "L0") == 1.0
    assert w.get_distance("L0", "end") == 2.0


def test_warehouse_set_distance_backward_compatibility():
    """Test that old set_distance() API still works correctly"""
    w = Warehouse(locations=["start", "end"], start_point_id="start", end_point_id="end")

    # Old API should still work
    w.set_distance("start", "L0", 1.5)
    w.set_distance("L0", "end", 2.0)

    assert w.get_distance("start", "L0") == 1.5
    assert w.get_distance("L0", "end") == 2.0
    assert "L0" in w._locations
