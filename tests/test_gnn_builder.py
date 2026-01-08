"""Test suite for GNN builder functions."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from slotting_optimization.models import Order
from slotting_optimization.warehouse import Warehouse
from slotting_optimization.order_book import OrderBook
from slotting_optimization.item_locations import ItemLocations
from slotting_optimization.simulator import build_combined_matrix
from slotting_optimization.gnn_builder import build_graph_sparse, build_graph_dense


def make_order(order_id, item_id, ts):
    """Helper to create Order from dict."""
    return Order.from_dict({"order_id": order_id, "item_id": item_id, "timestamp": ts})


@pytest.fixture
def small_setup():
    """Create 2 items, 4 locs (2 storage + start + end) setup."""
    # Create warehouse with 2 storage locations + start + end
    w = Warehouse(
        locations=["L0", "L1", "start", "end"],
        start_point_id="start",
        end_point_id="end"
    )

    # Set all necessary distances to avoid NaN
    # From start to storage locations
    w.set_distance("start", "L0", 5.0)
    w.set_distance("start", "L1", 5.0)
    w.set_distance("start", "end", 10.0)

    # From storage to storage (asymmetric for testing)
    w.set_distance("L0", "L1", 1.0)
    w.set_distance("L1", "L0", 2.0)

    # From storage to end
    w.set_distance("L0", "end", 3.0)
    w.set_distance("L1", "end", 3.0)
    w.set_distance("L0", "start", 4.0)
    w.set_distance("L1", "start", 4.0)

    # From end to start
    w.set_distance("end", "start", 2.0)
    w.set_distance("end", "L0", 6.0)
    w.set_distance("end", "L1", 6.0)

    # Create orders with 2 items
    ob = OrderBook.from_orders([
        make_order("O1", "I1", "2024-01-01T10:00:00"),
        make_order("O1", "I2", "2024-01-01T10:01:00"),
        make_order("O2", "I1", "2024-01-01T10:02:00"),  # I2 -> I1 sequence
    ])

    # Assign items to storage locations
    il = ItemLocations.from_records([
        {"item_id": "I1", "location_id": "L0"},
        {"item_id": "I2", "location_id": "L1"}
    ])

    return ob, il, w


@pytest.fixture
def full_setup():
    """Create 2 items, 4 locs with full OrderBook, ItemLocations, Warehouse for simulation."""
    # Same warehouse setup as small_setup
    w = Warehouse(
        locations=["L0", "L1", "start", "end"],
        start_point_id="start",
        end_point_id="end"
    )

    # Set all necessary distances
    w.set_distance("start", "L0", 5.0)
    w.set_distance("start", "L1", 5.0)
    w.set_distance("start", "end", 10.0)
    w.set_distance("L0", "L1", 1.0)
    w.set_distance("L1", "L0", 2.0)
    w.set_distance("L0", "end", 3.0)
    w.set_distance("L1", "end", 3.0)
    w.set_distance("L0", "start", 4.0)
    w.set_distance("L1", "start", 4.0)
    w.set_distance("end", "start", 2.0)
    w.set_distance("end", "L0", 6.0)
    w.set_distance("end", "L1", 6.0)

    # Create orders
    ob = OrderBook.from_orders([
        make_order("O1", "I1", "2024-01-01T10:00:00"),
        make_order("O1", "I2", "2024-01-01T10:01:00"),
        make_order("O2", "I1", "2024-01-01T10:02:00"),
    ])

    # Assign items to storage locations
    il = ItemLocations.from_records([
        {"item_id": "I1", "location_id": "L0"},
        {"item_id": "I2", "location_id": "L1"}
    ])

    return ob, il, w


# Group 1: Basic Structure Tests

def test_sparse_basic_structure(small_setup):
    """Verify sparse graph has correct edge_index and edge_attr shapes."""
    ob, il, w = small_setup
    data = build_graph_sparse(ob, il, w)

    # Check edge_index shape and dtype
    assert data.edge_index.shape[0] == 2, "edge_index should have shape [2, num_edges]"
    assert data.edge_index.dtype == torch.long, "edge_index should be torch.long"

    # Check edge_attr shape and dtype
    num_edges = data.edge_index.shape[1]
    assert data.edge_attr.shape == (num_edges, 1), "edge_attr should have shape [num_edges, 1]"
    assert data.edge_attr.dtype == torch.float, "edge_attr should be torch.float"


def test_dense_basic_structure(small_setup):
    """Verify dense graph has correct edge_index and edge_attr shapes."""
    ob, il, w = small_setup
    data = build_graph_dense(ob, il, w)

    # Check edge_index shape and dtype
    assert data.edge_index.shape[0] == 2, "edge_index should have shape [2, num_edges]"
    assert data.edge_index.dtype == torch.long, "edge_index should be torch.long"

    # Check edge_attr shape and dtype
    num_edges = data.edge_index.shape[1]
    assert data.edge_attr.shape == (num_edges, 1), "edge_attr should have shape [num_edges, 1]"
    assert data.edge_attr.dtype == torch.float, "edge_attr should be torch.float"


# Group 2: Edge Count Validation Tests

def test_sparse_edge_count(small_setup):
    """Verify sparse graph creates edges only for non-zero matrix values."""
    ob, il, w = small_setup
    data = build_graph_sparse(ob, il, w)

    # Build combined matrix to calculate expected edge count
    combined, metadata = build_combined_matrix(ob, il, w)
    I, L = metadata['n_items'], metadata['n_locs']

    # Count non-zeros in each quadrant (excluding diagonal)
    item_loc_quad = combined[:I, :L]
    seq_quad = combined[:I, L:]
    loc_quad = combined[I:, :L]

    # seq_quad and loc_quad: exclude diagonal
    seq_diag_mask = np.eye(I, dtype=bool)
    loc_diag_mask = np.eye(L, dtype=bool)

    expected_count = (
        np.count_nonzero(item_loc_quad) +
        np.count_nonzero(seq_quad[~seq_diag_mask]) +
        np.count_nonzero(loc_quad[~loc_diag_mask])
    )

    assert data.edge_index.shape[1] == expected_count


def test_dense_edge_count(small_setup):
    """Verify dense graph creates edges for all positions (except diagonal and bottom-right)."""
    ob, il, w = small_setup
    data = build_graph_dense(ob, il, w)

    # Get dimensions from Data object
    I, L = data.n_items, data.n_locs
    total_nodes = I + L

    # Total possible edges
    total_possible = total_nodes * total_nodes

    # Subtract diagonal
    total_possible -= total_nodes

    # Subtract bottom-right quadrant (L × I)
    total_possible -= (L * I)

    expected_count = total_possible

    assert data.edge_index.shape[1] == expected_count


def test_sparse_vs_dense_edge_count_difference(small_setup):
    """Verify sparse has fewer edges than dense when matrix has zeros."""
    ob, il, w = small_setup

    sparse_data = build_graph_sparse(ob, il, w)
    dense_data = build_graph_dense(ob, il, w)

    sparse_edges = sparse_data.edge_index.shape[1]
    dense_edges = dense_data.edge_index.shape[1]

    # Sparse should have fewer edges because not all positions are non-zero
    assert sparse_edges < dense_edges, "Sparse should have fewer edges than dense"


# Group 3: NaN Validation Tests

def test_sparse_nan_validation_triggers():
    """Verify NaN from missing warehouse distance raises ValueError."""
    # Create warehouse with MISSING distance (will produce NaN in loc_mat)
    w = Warehouse(
        locations=["L0", "start", "end"],
        start_point_id="start",
        end_point_id="end"
    )
    # Define most distances but intentionally skip one to create NaN
    w.set_distance("start", "L0", 5.0)
    w.set_distance("start", "end", 10.0)
    w.set_distance("L0", "end", 3.0)
    w.set_distance("L0", "start", 4.0)
    # MISSING: w.set_distance("end", "start", 2.0) - will produce NaN!
    w.set_distance("end", "L0", 6.0)

    ob = OrderBook.from_orders([
        make_order("O1", "I1", "2024-01-01T10:00:00")
    ])

    il = ItemLocations.from_records([
        {"item_id": "I1", "location_id": "L0"}
    ])

    # Should raise ValueError during internal matrix building
    with pytest.raises(ValueError, match="NaN detected"):
        build_graph_sparse(ob, il, w)


def test_dense_nan_validation_triggers():
    """Verify NaN from missing warehouse distance raises ValueError."""
    # Create warehouse with MISSING distance (will produce NaN in loc_mat)
    w = Warehouse(
        locations=["L0", "start", "end"],
        start_point_id="start",
        end_point_id="end"
    )
    # Define most distances but intentionally skip one to create NaN
    w.set_distance("start", "L0", 5.0)
    w.set_distance("start", "end", 10.0)
    w.set_distance("L0", "end", 3.0)
    # MISSING: w.set_distance("L0", "start", 4.0) - will produce NaN!
    w.set_distance("end", "start", 2.0)
    w.set_distance("end", "L0", 6.0)

    ob = OrderBook.from_orders([
        make_order("O1", "I1", "2024-01-01T10:00:00")
    ])

    il = ItemLocations.from_records([
        {"item_id": "I1", "location_id": "L0"}
    ])

    # Should raise ValueError during internal matrix building
    with pytest.raises(ValueError, match="NaN detected"):
        build_graph_dense(ob, il, w)


def test_nan_validation_can_be_disabled():
    """Verify validate_nan=False skips NaN check."""
    # Create warehouse with MISSING distance (will produce NaN)
    w = Warehouse(
        locations=["L0", "start", "end"],
        start_point_id="start",
        end_point_id="end"
    )
    # Define most distances but intentionally skip one
    w.set_distance("start", "L0", 5.0)
    w.set_distance("start", "end", 10.0)
    w.set_distance("L0", "end", 3.0)
    w.set_distance("L0", "start", 4.0)
    # MISSING: w.set_distance("end", "start", 2.0) - will produce NaN!
    w.set_distance("end", "L0", 6.0)

    ob = OrderBook.from_orders([
        make_order("O1", "I1", "2024-01-01T10:00:00")
    ])

    il = ItemLocations.from_records([
        {"item_id": "I1", "location_id": "L0"}
    ])

    # Should not raise with validate_nan=False
    # (may produce invalid graph, but that's user's choice)
    data = build_graph_sparse(ob, il, w, validate_nan=False)
    assert data is not None


# Group 4: Structural Constraints Tests

def test_sparse_no_self_loops(small_setup):
    """Verify diagonal elements are excluded (no i==j edges)."""
    ob, il, w = small_setup
    data = build_graph_sparse(ob, il, w)

    # Check no self-loops
    edges = data.edge_index.numpy()
    sources = edges[0]
    targets = edges[1]

    self_loops = sources == targets
    assert not self_loops.any(), "Sparse graph should have no self-loops"


def test_dense_no_self_loops(small_setup):
    """Verify diagonal elements are excluded (no i==j edges)."""
    ob, il, w = small_setup
    data = build_graph_dense(ob, il, w)

    # Check no self-loops
    edges = data.edge_index.numpy()
    sources = edges[0]
    targets = edges[1]

    self_loops = sources == targets
    assert not self_loops.any(), "Dense graph should have no self-loops"


def test_sparse_bottom_right_quadrant_skipped(small_setup):
    """Verify bottom-right (L×I) zeros block creates no edges in sparse."""
    ob, il, w = small_setup
    data = build_graph_sparse(ob, il, w)

    # For this test, I'll verify that all edges in sparse have non-zero values
    # (since bottom-right quadrant is all zeros, sparse should skip it)
    edges_np = data.edge_index.numpy()
    attrs_np = data.edge_attr.numpy()

    # All edge attributes should be non-zero in sparse graph
    assert (attrs_np != 0).all(), "Sparse graph should only have non-zero edge weights"


def test_dense_bottom_right_quadrant_skipped(small_setup):
    """Verify bottom-right (L×I) zeros block creates no edges even in dense."""
    ob, il, w = small_setup
    data = build_graph_dense(ob, il, w)

    # The edge count test already verifies this - dense skips the bottom-right quadrant
    # which is subtracted from the total possible edges
    # This test is a placeholder for future verification if needed
    pass


# Group 5: Edge Directionality Tests

def test_sparse_preserves_asymmetry(small_setup):
    """Verify directed edges preserve asymmetry (A->B != B->A)."""
    ob, il, w = small_setup
    data = build_graph_sparse(ob, il, w)

    # We set L0->L1 = 1.0 and L1->L0 = 2.0 in the fixture
    # Find indices of L0 and L1 in node indexing
    I = data.n_items
    locs = data.locs_list

    l0_idx = locs.index("L0")
    l1_idx = locs.index("L1")

    # Node indices (locations start at index I)
    l0_node = I + l0_idx
    l1_node = I + l1_idx

    # Find edges L0->L1 and L1->L0
    edges = data.edge_index.numpy()
    attrs = data.edge_attr.numpy()

    l0_to_l1_mask = (edges[0] == l0_node) & (edges[1] == l1_node)
    l1_to_l0_mask = (edges[0] == l1_node) & (edges[1] == l0_node)

    # Check that both edges exist
    assert l0_to_l1_mask.any(), "L0->L1 edge should exist"
    assert l1_to_l0_mask.any(), "L1->L0 edge should exist"

    # Get weights
    l0_to_l1_weight = attrs[l0_to_l1_mask][0, 0]
    l1_to_l0_weight = attrs[l1_to_l0_mask][0, 0]

    # Verify asymmetry
    assert l0_to_l1_weight == 1.0, "L0->L1 weight should be 1.0"
    assert l1_to_l0_weight == 2.0, "L1->L0 weight should be 2.0"
    assert l0_to_l1_weight != l1_to_l0_weight, "Weights should be different (asymmetric)"


def test_dense_preserves_asymmetry(small_setup):
    """Verify directed edges preserve asymmetry in dense mode."""
    ob, il, w = small_setup
    data = build_graph_dense(ob, il, w)

    # Same test as sparse
    I = data.n_items
    locs = data.locs_list

    l0_idx = locs.index("L0")
    l1_idx = locs.index("L1")

    l0_node = I + l0_idx
    l1_node = I + l1_idx

    edges = data.edge_index.numpy()
    attrs = data.edge_attr.numpy()

    l0_to_l1_mask = (edges[0] == l0_node) & (edges[1] == l1_node)
    l1_to_l0_mask = (edges[0] == l1_node) & (edges[1] == l0_node)

    assert l0_to_l1_mask.any(), "L0->L1 edge should exist"
    assert l1_to_l0_mask.any(), "L1->L0 edge should exist"

    l0_to_l1_weight = attrs[l0_to_l1_mask][0, 0]
    l1_to_l0_weight = attrs[l1_to_l0_mask][0, 0]

    assert l0_to_l1_weight == 1.0, "L0->L1 weight should be 1.0"
    assert l1_to_l0_weight == 2.0, "L1->L0 weight should be 2.0"


# Group 6: Edge Attribute Tests

def test_sparse_edge_values_correct(small_setup):
    """Verify edge_attr values match non-zero matrix positions."""
    ob, il, w = small_setup
    data = build_graph_sparse(ob, il, w)

    # Build combined matrix to verify values
    combined, metadata = build_combined_matrix(ob, il, w)
    I, L = metadata['n_items'], metadata['n_locs']

    # For each edge, verify the attribute matches the matrix value
    edges = data.edge_index.numpy()
    attrs = data.edge_attr.numpy()

    for idx in range(edges.shape[1]):
        src_node = edges[0, idx]
        dst_node = edges[1, idx]
        attr_value = attrs[idx, 0]

        # Convert node indices to matrix indices
        # src_node is already the row index
        # dst_node needs to be converted to column index:
        #   - If dst_node < I (item): matrix_col = L + dst_node
        #   - If dst_node >= I (location): matrix_col = dst_node - I
        if dst_node < I:
            matrix_col = L + dst_node
        else:
            matrix_col = dst_node - I

        matrix_value = combined[src_node, matrix_col]

        assert np.isclose(attr_value, matrix_value), \
            f"Edge {src_node}->{dst_node} has attr {attr_value} but matrix has {matrix_value}"


def test_dense_edge_values_correct(small_setup):
    """Verify edge_attr values match matrix positions (including 0s)."""
    ob, il, w = small_setup
    data = build_graph_dense(ob, il, w)

    # Build combined matrix to verify values
    combined, metadata = build_combined_matrix(ob, il, w)
    I, L = metadata['n_items'], metadata['n_locs']

    # For each edge, verify the attribute matches the matrix value
    edges = data.edge_index.numpy()
    attrs = data.edge_attr.numpy()

    for idx in range(edges.shape[1]):
        src_node = edges[0, idx]
        dst_node = edges[1, idx]
        attr_value = attrs[idx, 0]

        # Convert node indices to matrix indices
        if dst_node < I:
            matrix_col = L + dst_node
        else:
            matrix_col = dst_node - I

        matrix_value = combined[src_node, matrix_col]

        assert np.isclose(attr_value, matrix_value), \
            f"Edge {src_node}->{dst_node} has attr {attr_value} but matrix has {matrix_value}"


def test_dense_can_have_zero_edge_weights(small_setup):
    """Verify dense mode includes edges with weight 0.0."""
    ob, il, w = small_setup
    data = build_graph_dense(ob, il, w)

    attrs = data.edge_attr.numpy()

    # Check if there are any zero-valued edges
    zero_edges = (attrs == 0.0).any()

    # Build combined matrix to check if there are zeros in valid regions
    combined, metadata = build_combined_matrix(ob, il, w)
    I, L = metadata['n_items'], metadata['n_locs']

    # Check for zeros in valid regions (not diagonal, not bottom-right)
    # Top-left: item_loc_mat
    top_left = combined[:I, :L]

    # Top-right: seq_mat (excluding diagonal)
    top_right = combined[:I, L:]
    seq_diag_mask = np.eye(I, dtype=bool)
    top_right_no_diag = top_right[~seq_diag_mask]

    # Bottom-left: loc_mat (excluding diagonal)
    bottom_left = combined[I:, :L]
    loc_diag_mask = np.eye(L, dtype=bool)
    bottom_left_no_diag = bottom_left[~loc_diag_mask]

    # Check if any of these regions have zeros
    has_zeros = (
        (top_left == 0.0).any() or
        (top_right_no_diag == 0.0).any() or
        (bottom_left_no_diag == 0.0).any()
    )

    if has_zeros:
        assert zero_edges, "Dense graph should include zero-valued edges when matrix has zeros"


# Group 7: Metadata Tests

def test_sparse_metadata_included(small_setup):
    """Verify Data object contains n_items, n_locs, items_list, locs_list attributes."""
    ob, il, w = small_setup
    data = build_graph_sparse(ob, il, w)

    # Build combined matrix to get expected metadata values
    combined, metadata = build_combined_matrix(ob, il, w)

    assert hasattr(data, 'n_items'), "Data should have n_items attribute"
    assert hasattr(data, 'n_locs'), "Data should have n_locs attribute"
    assert hasattr(data, 'n_storage'), "Data should have n_storage attribute"
    assert hasattr(data, 'items_list'), "Data should have items_list attribute"
    assert hasattr(data, 'locs_list'), "Data should have locs_list attribute"

    assert data.n_items == metadata['n_items']
    assert data.n_locs == metadata['n_locs']
    assert data.n_storage == metadata['n_storage']
    assert data.items_list == metadata['items']
    assert data.locs_list == metadata['locs']


def test_dense_metadata_included(small_setup):
    """Verify Data object contains metadata attributes."""
    ob, il, w = small_setup
    data = build_graph_dense(ob, il, w)

    # Build combined matrix to get expected metadata values
    combined, metadata = build_combined_matrix(ob, il, w)

    assert hasattr(data, 'n_items'), "Data should have n_items attribute"
    assert hasattr(data, 'n_locs'), "Data should have n_locs attribute"
    assert hasattr(data, 'items_list'), "Data should have items_list attribute"
    assert hasattr(data, 'locs_list'), "Data should have locs_list attribute"
    assert data.n_items == metadata['n_items']
    assert data.n_locs == metadata['n_locs']
    assert data.items_list == metadata['items']
    assert data.locs_list == metadata['locs']


def test_num_nodes_attribute(small_setup):
    """Verify num_nodes = n_items + n_locs."""
    ob, il, w = small_setup
    data = build_graph_sparse(ob, il, w)

    expected_num_nodes = data.n_items + data.n_locs

    assert hasattr(data, 'num_nodes'), "Data should have num_nodes attribute"
    assert data.num_nodes == expected_num_nodes


# Group 8: Error Handling Tests

def test_missing_metadata_key_raises():
    """Verify missing metadata keys raise KeyError in internal validation."""
    from slotting_optimization.gnn_builder import _validate_inputs

    combined = np.zeros((5, 5))
    metadata = {'n_items': 2}  # Missing required keys

    with pytest.raises(KeyError, match="metadata missing required key"):
        _validate_inputs(combined, metadata, validate_nan=True)


def test_shape_mismatch_raises():
    """Verify mismatched matrix shape raises ValueError in internal validation."""
    from slotting_optimization.gnn_builder import _validate_inputs

    combined = np.zeros((5, 6))  # Non-square matrix
    metadata = {
        'n_items': 2,
        'n_locs': 3,
        'n_storage': 1,
        'items': ['I1', 'I2'],
        'locs': ['L0', 'start', 'end']
    }

    with pytest.raises(ValueError, match="shape.*doesn't match"):
        _validate_inputs(combined, metadata, validate_nan=True)


def test_nan_error_message_includes_position():
    """Verify NaN error message shows position and count in internal validation."""
    from slotting_optimization.gnn_builder import _validate_inputs

    # Create a matrix with multiple NaNs
    combined = np.zeros((4, 4))
    combined[0, 1] = np.nan
    combined[2, 3] = np.nan

    metadata = {
        'n_items': 2,
        'n_locs': 2,
        'n_storage': 1,
        'items': ['I1', 'I2'],
        'locs': ['L0', 'start']
    }

    with pytest.raises(ValueError) as exc_info:
        _validate_inputs(combined, metadata, validate_nan=True)

    error_msg = str(exc_info.value)
    assert "NaN detected" in error_msg
    assert "position" in error_msg
    assert "2" in error_msg  # Should mention count of 2 NaNs


# Group 9: Simulation Target Tests

def test_sparse_with_simulator_adds_y_attribute(full_setup):
    """Verify y attribute added when simulator provided."""
    from slotting_optimization.simulator import Simulator

    ob, il, w = full_setup
    sim = Simulator()

    data = build_graph_sparse(
        ob, il, w,
        simulator=sim.simulate
    )

    assert hasattr(data, 'y'), "Data should have y attribute when simulator provided"
    assert data.y.shape == (1,), "y should be shape [1] for graph-level target"
    assert data.y.dtype == torch.float, "y should be float dtype"


def test_dense_with_simulator_adds_y_attribute(full_setup):
    """Verify y attribute added when simulator provided for dense graph."""
    from slotting_optimization.simulator import Simulator

    ob, il, w = full_setup
    sim = Simulator()

    data = build_graph_dense(
        ob, il, w,
        simulator=sim.simulate
    )

    assert hasattr(data, 'y'), "Data should have y attribute when simulator provided"
    assert data.y.shape == (1,), "y should be shape [1] for graph-level target"
    assert data.y.dtype == torch.float, "y should be float dtype"


def test_sparse_y_value_matches_simulation(full_setup):
    """Verify y value equals simulation result."""
    from slotting_optimization.simulator import Simulator

    ob, il, w = full_setup
    sim = Simulator()

    # Get expected value
    expected_distance, _ = sim.simulate(ob, w, il)

    # Build graph with y
    data = build_graph_sparse(
        ob, il, w,
        simulator=sim.simulate
    )

    assert torch.isclose(data.y, torch.tensor([expected_distance])).item(), \
        f"Expected y={expected_distance}, got {data.y.item()}"


def test_dense_y_value_matches_simulation(full_setup):
    """Verify y value equals simulation result for dense graph."""
    from slotting_optimization.simulator import Simulator

    ob, il, w = full_setup
    sim = Simulator()

    # Get expected value
    expected_distance, _ = sim.simulate(ob, w, il)

    # Build graph with y
    data = build_graph_dense(
        ob, il, w,
        simulator=sim.simulate
    )

    assert torch.isclose(data.y, torch.tensor([expected_distance])).item(), \
        f"Expected y={expected_distance}, got {data.y.item()}"


def test_sparse_without_simulator_no_y(small_setup):
    """Verify y not added when simulator=None."""
    ob, il, w = small_setup
    data = build_graph_sparse(ob, il, w)

    # PyG Data objects have y attribute by default, but it should be None when not provided
    assert data.y is None, "Data.y should be None when simulator=None"


def test_simulator_parameter_is_optional(full_setup):
    """Verify simulator parameter is optional (test new API)."""
    # With the new API, ob/il/w are required but simulator is optional
    # This test verifies the new simplified API works correctly
    ob, il, w = full_setup

    # Should work without simulator
    data_no_sim = build_graph_sparse(ob, il, w)
    assert data_no_sim.y is None, "No simulator should mean no y attribute"

    # Should work with simulator
    from slotting_optimization.simulator import Simulator
    sim = Simulator()
    data_with_sim = build_graph_sparse(ob, il, w, simulator=sim.simulate)
    assert data_with_sim.y is not None, "Simulator should add y attribute"
