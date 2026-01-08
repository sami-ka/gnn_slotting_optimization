"""Test suite for 3D edge attribute GNN builder functions."""

import numpy as np
import pytest
import torch

from slotting_optimization.models import Order
from slotting_optimization.warehouse import Warehouse
from slotting_optimization.order_book import OrderBook
from slotting_optimization.item_locations import ItemLocations
from slotting_optimization.simulator import (
    build_combined_matrix,
    extract_submatrices,
    Simulator,
)


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
        end_point_id="end",
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
    ob = OrderBook.from_orders(
        [
            make_order("O1", "I1", "2024-01-01T10:00:00"),
            make_order("O1", "I2", "2024-01-01T10:01:00"),
            make_order("O2", "I1", "2024-01-01T10:02:00"),  # I2 -> I1 sequence
        ]
    )

    # Assign items to storage locations
    il = ItemLocations.from_records(
        [{"item_id": "I1", "location_id": "L0"}, {"item_id": "I2", "location_id": "L1"}]
    )

    return ob, il, w


@pytest.fixture
def full_setup():
    """Create 2 items, 4 locs with full OrderBook, ItemLocations, Warehouse for simulation."""
    # Same warehouse setup as small_setup
    w = Warehouse(
        locations=["L0", "L1", "start", "end"],
        start_point_id="start",
        end_point_id="end",
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
    ob = OrderBook.from_orders(
        [
            make_order("O1", "I1", "2024-01-01T10:00:00"),
            make_order("O1", "I2", "2024-01-01T10:01:00"),
            make_order("O2", "I1", "2024-01-01T10:02:00"),
        ]
    )

    # Assign items to storage locations
    il = ItemLocations.from_records(
        [{"item_id": "I1", "location_id": "L0"}, {"item_id": "I2", "location_id": "L1"}]
    )

    return ob, il, w


@pytest.fixture
def asymmetric_setup():
    """Setup with clearly asymmetric distances for directionality testing."""
    w = Warehouse(
        locations=["L0", "L1", "L2", "start", "end"],
        start_point_id="start",
        end_point_id="end",
    )

    # Asymmetric distances (deliberately different in each direction)
    w.set_distance("start", "L0", 5.0)
    w.set_distance("start", "L1", 6.0)
    w.set_distance("start", "L2", 7.0)
    w.set_distance("start", "end", 10.0)

    w.set_distance("L0", "L1", 1.0)
    w.set_distance("L1", "L0", 10.0)  # Very different
    w.set_distance("L0", "L2", 2.0)
    w.set_distance("L2", "L0", 20.0)  # Very different
    w.set_distance("L1", "L2", 3.0)
    w.set_distance("L2", "L1", 30.0)  # Very different

    w.set_distance("L0", "end", 3.0)
    w.set_distance("L1", "end", 4.0)
    w.set_distance("L2", "end", 5.0)

    w.set_distance("L0", "start", 8.0)
    w.set_distance("L1", "start", 9.0)
    w.set_distance("L2", "start", 10.0)

    w.set_distance("end", "start", 2.0)
    w.set_distance("end", "L0", 6.0)
    w.set_distance("end", "L1", 7.0)
    w.set_distance("end", "L2", 8.0)

    # Create orders with 3 items (more sequences)
    ob = OrderBook.from_orders(
        [
            make_order("O1", "I1", "2024-01-01T10:00:00"),
            make_order("O1", "I2", "2024-01-01T10:01:00"),
            make_order("O1", "I3", "2024-01-01T10:02:00"),
            make_order("O2", "I2", "2024-01-01T10:03:00"),  # I3 -> I2 sequence
            make_order("O2", "I1", "2024-01-01T10:04:00"),  # I2 -> I1 sequence
            make_order("O3", "I3", "2024-01-01T10:05:00"),  # I1 -> I3 sequence
        ]
    )

    # Assign items to storage locations
    il = ItemLocations.from_records(
        [
            {"item_id": "I1", "location_id": "L0"},
            {"item_id": "I2", "location_id": "L1"},
            {"item_id": "I3", "location_id": "L2"},
        ]
    )

    return ob, il, w


# =============================================================================
# Group 1: Basic Structure Validation (4 tests)
# =============================================================================


def test_3d_sparse_edge_attr_shape(small_setup):
    """Verify sparse graph has edge_attr shape [num_edges, 3]."""
    from slotting_optimization.gnn_builder import build_graph_3d_sparse

    ob, il, w = small_setup
    data = build_graph_3d_sparse(ob, il, w)

    # Check edge_attr shape and dtype
    num_edges = data.edge_index.shape[1]
    assert data.edge_attr.shape == (num_edges, 3), (
        "edge_attr should have shape [num_edges, 3]"
    )
    assert data.edge_attr.dtype == torch.float, "edge_attr should be torch.float"


def test_3d_dense_edge_attr_shape(small_setup):
    """Verify dense graph has edge_attr shape [num_edges, 3]."""
    from slotting_optimization.gnn_builder import build_graph_3d_dense

    ob, il, w = small_setup
    data = build_graph_3d_dense(ob, il, w)

    # Check edge_attr shape and dtype
    num_edges = data.edge_index.shape[1]
    assert data.edge_attr.shape == (num_edges, 3), (
        "edge_attr should have shape [num_edges, 3]"
    )
    assert data.edge_attr.dtype == torch.float, "edge_attr should be torch.float"


def test_3d_sparse_edge_type_mask_shape(small_setup):
    """Verify sparse graph has edge_type_mask shape [num_edges, 3]."""
    from slotting_optimization.gnn_builder import build_graph_3d_sparse

    ob, il, w = small_setup
    data = build_graph_3d_sparse(ob, il, w)

    # Check edge_type_mask exists, shape and dtype
    assert hasattr(data, "edge_type_mask"), "Data should have edge_type_mask attribute"
    num_edges = data.edge_index.shape[1]
    assert data.edge_type_mask.shape == (num_edges, 3), (
        "edge_type_mask should have shape [num_edges, 3]"
    )
    assert data.edge_type_mask.dtype == torch.bool, (
        "edge_type_mask should be torch.bool"
    )


def test_3d_dense_edge_type_mask_shape(small_setup):
    """Verify dense graph has edge_type_mask shape [num_edges, 3]."""
    from slotting_optimization.gnn_builder import build_graph_3d_dense

    ob, il, w = small_setup
    data = build_graph_3d_dense(ob, il, w)

    # Check edge_type_mask exists, shape and dtype
    assert hasattr(data, "edge_type_mask"), "Data should have edge_type_mask attribute"
    num_edges = data.edge_index.shape[1]
    assert data.edge_type_mask.shape == (num_edges, 3), (
        "edge_type_mask should have shape [num_edges, 3]"
    )
    assert data.edge_type_mask.dtype == torch.bool, (
        "edge_type_mask should be torch.bool"
    )


# =============================================================================
# Group 2: Edge Type Classification (4 tests)
# =============================================================================


def test_3d_item_to_loc_edges_have_assignment_only(small_setup):
    """Verify Item→Location edges have only dimension 2 non-zero."""
    from slotting_optimization.gnn_builder import build_graph_3d_sparse

    ob, il, w = small_setup
    data = build_graph_3d_sparse(ob, il, w)

    edges = data.edge_index.numpy()
    attrs = data.edge_attr.numpy()
    masks = data.edge_type_mask.numpy()
    I = data.n_items

    # Find Item→Location edges (src < I and dst >= I)
    src_is_item = edges[0] < I
    dst_is_loc = edges[1] >= I
    item_to_loc_mask = src_is_item & dst_is_loc

    if not item_to_loc_mask.any():
        pytest.skip("No Item→Location edges in this setup")

    # Check each Item→Location edge
    for idx in np.where(item_to_loc_mask)[0]:
        attr_vec = attrs[idx]
        mask_vec = masks[idx]

        # Dimension 2 should be active
        assert mask_vec[2], f"Item→Location edge {idx} should have mask[2]=True"
        # Dimensions 0 and 1 should be zero
        assert attr_vec[0] == 0.0, f"Item→Location edge {idx} should have attr[0]=0.0"
        assert attr_vec[1] == 0.0, f"Item→Location edge {idx} should have attr[1]=0.0"


def test_3d_item_to_item_edges_have_sequence_only(small_setup):
    """Verify Item→Item edges have only dimension 1 non-zero."""
    from slotting_optimization.gnn_builder import build_graph_3d_sparse

    ob, il, w = small_setup
    data = build_graph_3d_sparse(ob, il, w)

    edges = data.edge_index.numpy()
    attrs = data.edge_attr.numpy()
    masks = data.edge_type_mask.numpy()
    I = data.n_items

    # Find Item→Item edges (src < I and dst < I)
    src_is_item = edges[0] < I
    dst_is_item = edges[1] < I
    item_to_item_mask = src_is_item & dst_is_item

    if not item_to_item_mask.any():
        pytest.skip("No Item→Item edges in this setup")

    # Check each Item→Item edge
    for idx in np.where(item_to_item_mask)[0]:
        attr_vec = attrs[idx]
        mask_vec = masks[idx]

        # Dimension 1 should be active
        assert mask_vec[1], f"Item→Item edge {idx} should have mask[1]=True"
        # Dimensions 0 and 2 should be zero
        assert attr_vec[0] == 0.0, f"Item→Item edge {idx} should have attr[0]=0.0"
        assert attr_vec[2] == 0.0, f"Item→Item edge {idx} should have attr[2]=0.0"


def test_3d_loc_to_loc_edges_have_distance_only(small_setup):
    """Verify Location→Location edges have only dimension 0 non-zero."""
    from slotting_optimization.gnn_builder import build_graph_3d_sparse

    ob, il, w = small_setup
    data = build_graph_3d_sparse(ob, il, w)

    edges = data.edge_index.numpy()
    attrs = data.edge_attr.numpy()
    masks = data.edge_type_mask.numpy()
    I = data.n_items

    # Find Location→Location edges (src >= I and dst >= I)
    src_is_loc = edges[0] >= I
    dst_is_loc = edges[1] >= I
    loc_to_loc_mask = src_is_loc & dst_is_loc

    if not loc_to_loc_mask.any():
        pytest.skip("No Location→Location edges in this setup")

    # Check each Location→Location edge
    for idx in np.where(loc_to_loc_mask)[0]:
        attr_vec = attrs[idx]
        mask_vec = masks[idx]

        # Dimension 0 should be active
        assert mask_vec[0], f"Location→Location edge {idx} should have mask[0]=True"
        # Dimensions 1 and 2 should be zero
        assert attr_vec[1] == 0.0, (
            f"Location→Location edge {idx} should have attr[1]=0.0"
        )
        assert attr_vec[2] == 0.0, (
            f"Location→Location edge {idx} should have attr[2]=0.0"
        )


def test_3d_edge_type_mask_mutually_exclusive(small_setup):
    """Verify edge type mask is mutually exclusive (exactly one True per edge)."""
    from slotting_optimization.gnn_builder import build_graph_3d_sparse

    ob, il, w = small_setup
    data = build_graph_3d_sparse(ob, il, w)

    masks = data.edge_type_mask.numpy()

    # Each edge should have exactly one True in mask
    for idx in range(masks.shape[0]):
        mask_vec = masks[idx]
        num_true = mask_vec.sum()
        assert num_true == 1, (
            f"Edge {idx} should have exactly one True in mask, got {num_true}"
        )


# =============================================================================
# Group 3: Value Correctness (3 tests)
# =============================================================================


def test_3d_sparse_values_match_submatrices(small_setup):
    """Verify 3D edge values come from correct submatrices."""
    from slotting_optimization.gnn_builder import build_graph_3d_sparse

    ob, il, w = small_setup

    # Get submatrices
    combined, metadata = build_combined_matrix(ob, il, w)
    submat = extract_submatrices(combined, metadata)
    I = metadata["n_items"]

    # Build 3D graph
    data = build_graph_3d_sparse(ob, il, w)

    edges = data.edge_index.numpy()
    attrs = data.edge_attr.numpy()
    masks = data.edge_type_mask.numpy()

    # Verify each edge's non-zero dimension matches the original matrix
    for idx in range(edges.shape[1]):
        src = edges[0, idx]
        dst = edges[1, idx]
        attr_vec = attrs[idx]
        mask_vec = masks[idx]

        # Verify exactly one dimension is active
        assert mask_vec.sum() == 1, f"Edge {idx} should have exactly one type"

        if mask_vec[2]:  # Item→Location
            item_idx = src
            loc_idx = dst - I
            expected = submat["item_loc_mat"][item_idx, loc_idx]
            assert np.isclose(attr_vec[2], expected), (
                f"Edge {idx} (Item→Loc): expected attr[2]={expected}, got {attr_vec[2]}"
            )
            assert attr_vec[0] == 0.0 and attr_vec[1] == 0.0

        elif mask_vec[1]:  # Item→Item
            expected = submat["seq_mat"][src, dst]
            assert np.isclose(attr_vec[1], expected), (
                f"Edge {idx} (Item→Item): expected attr[1]={expected}, got {attr_vec[1]}"
            )
            assert attr_vec[0] == 0.0 and attr_vec[2] == 0.0

        elif mask_vec[0]:  # Location→Location
            loc_src = src - I
            loc_dst = dst - I
            expected = submat["loc_mat"][loc_src, loc_dst]
            assert np.isclose(attr_vec[0], expected), (
                f"Edge {idx} (Loc→Loc): expected attr[0]={expected}, got {attr_vec[0]}"
            )
            assert attr_vec[1] == 0.0 and attr_vec[2] == 0.0


def test_3d_dense_values_match_submatrices(small_setup):
    """Verify 3D dense edge values come from correct submatrices."""
    from slotting_optimization.gnn_builder import build_graph_3d_dense

    ob, il, w = small_setup

    # Get submatrices
    combined, metadata = build_combined_matrix(ob, il, w)
    submat = extract_submatrices(combined, metadata)
    I = metadata["n_items"]

    # Build 3D graph
    data = build_graph_3d_dense(ob, il, w)

    edges = data.edge_index.numpy()
    attrs = data.edge_attr.numpy()
    masks = data.edge_type_mask.numpy()

    # Verify each edge (same logic as sparse)
    for idx in range(edges.shape[1]):
        src = edges[0, idx]
        dst = edges[1, idx]
        attr_vec = attrs[idx]
        mask_vec = masks[idx]

        assert mask_vec.sum() == 1, f"Edge {idx} should have exactly one type"

        if mask_vec[2]:  # Item→Location
            item_idx = src
            loc_idx = dst - I
            expected = submat["item_loc_mat"][item_idx, loc_idx]
            assert np.isclose(attr_vec[2], expected)
            assert attr_vec[0] == 0.0 and attr_vec[1] == 0.0

        elif mask_vec[1]:  # Item→Item
            expected = submat["seq_mat"][src, dst]
            assert np.isclose(attr_vec[1], expected)
            assert attr_vec[0] == 0.0 and attr_vec[2] == 0.0

        elif mask_vec[0]:  # Location→Location
            loc_src = src - I
            loc_dst = dst - I
            expected = submat["loc_mat"][loc_src, loc_dst]
            assert np.isclose(attr_vec[0], expected)
            assert attr_vec[1] == 0.0 and attr_vec[2] == 0.0


def test_3d_zero_padding_correct(small_setup):
    """Verify zero-padding is correct for all edge types."""
    from slotting_optimization.gnn_builder import build_graph_3d_sparse

    ob, il, w = small_setup
    data = build_graph_3d_sparse(ob, il, w)

    attrs = data.edge_attr.numpy()
    masks = data.edge_type_mask.numpy()

    # For each edge, verify that non-active dimensions are exactly 0.0
    for idx in range(attrs.shape[0]):
        attr_vec = attrs[idx]
        mask_vec = masks[idx]

        # Check each dimension
        for dim in range(3):
            if not mask_vec[dim]:
                assert attr_vec[dim] == 0.0, (
                    f"Edge {idx} dimension {dim} should be 0.0 (not active), got {attr_vec[dim]}"
                )


# =============================================================================
# Group 4: Consistency with 2D Variants (3 tests)
# =============================================================================


def test_3d_sparse_has_same_edge_count_as_2d(small_setup):
    """Verify 3D sparse has same number of edges as 2D sparse."""
    from slotting_optimization.gnn_builder import (
        build_graph_sparse,
        build_graph_3d_sparse,
    )

    ob, il, w = small_setup

    data_2d = build_graph_sparse(ob, il, w)
    data_3d = build_graph_3d_sparse(ob, il, w)

    assert data_2d.edge_index.shape[1] == data_3d.edge_index.shape[1], (
        "2D and 3D sparse graphs should have same edge count"
    )


def test_3d_dense_has_same_edge_count_as_2d(small_setup):
    """Verify 3D dense has same number of edges as 2D dense."""
    from slotting_optimization.gnn_builder import (
        build_graph_dense,
        build_graph_3d_dense,
    )

    ob, il, w = small_setup

    data_2d = build_graph_dense(ob, il, w)
    data_3d = build_graph_3d_dense(ob, il, w)

    assert data_2d.edge_index.shape[1] == data_3d.edge_index.shape[1], (
        "2D and 3D dense graphs should have same edge count"
    )


def test_3d_nonzero_dimension_matches_2d_value(small_setup):
    """Verify the non-zero dimension in 3D matches the 2D edge value."""
    from slotting_optimization.gnn_builder import (
        build_graph_sparse,
        build_graph_3d_sparse,
    )

    ob, il, w = small_setup

    data_2d = build_graph_sparse(ob, il, w)
    data_3d = build_graph_3d_sparse(ob, il, w)

    # Both should have same edges in same order
    assert torch.equal(data_2d.edge_index, data_3d.edge_index), (
        "2D and 3D should have same edge_index"
    )

    attrs_2d = data_2d.edge_attr.numpy()
    attrs_3d = data_3d.edge_attr.numpy()
    masks_3d = data_3d.edge_type_mask.numpy()

    # For each edge, the non-zero dimension in 3D should match 2D value
    for idx in range(attrs_2d.shape[0]):
        val_2d = attrs_2d[idx, 0]
        vec_3d = attrs_3d[idx]
        mask_3d = masks_3d[idx]

        # Find the active dimension
        active_dim = np.argmax(mask_3d)  # Works because exactly one True
        val_3d = vec_3d[active_dim]

        assert np.isclose(val_2d, val_3d), (
            f"Edge {idx}: 2D value {val_2d} should match 3D non-zero dimension value {val_3d}"
        )


# =============================================================================
# Group 5: Edge Cases (4 tests)
# =============================================================================


def test_3d_asymmetric_distances_preserved(asymmetric_setup):
    """Verify asymmetric distances are preserved in 3D representation."""
    from slotting_optimization.gnn_builder import build_graph_3d_sparse

    ob, il, w = asymmetric_setup
    data = build_graph_3d_sparse(ob, il, w)

    # Find L0 and L1 node indices
    I = data.n_items
    locs = data.locs_list

    l0_idx = locs.index("L0")
    l1_idx = locs.index("L1")
    l0_node = I + l0_idx
    l1_node = I + l1_idx

    edges = data.edge_index.numpy()
    attrs = data.edge_attr.numpy()
    masks = data.edge_type_mask.numpy()

    # Find L0→L1 and L1→L0 edges
    l0_to_l1_mask = (edges[0] == l0_node) & (edges[1] == l1_node)
    l1_to_l0_mask = (edges[0] == l1_node) & (edges[1] == l0_node)

    assert l0_to_l1_mask.any(), "L0→L1 edge should exist"
    assert l1_to_l0_mask.any(), "L1→L0 edge should exist"

    # Get distance values (dimension 0)
    l0_to_l1_idx = np.where(l0_to_l1_mask)[0][0]
    l1_to_l0_idx = np.where(l1_to_l0_mask)[0][0]

    # Both should be Location→Location edges (mask[0]=True)
    assert masks[l0_to_l1_idx][0], "L0→L1 should be Location→Location edge"
    assert masks[l1_to_l0_idx][0], "L1→L0 should be Location→Location edge"

    l0_to_l1_dist = attrs[l0_to_l1_idx][0]
    l1_to_l0_dist = attrs[l1_to_l0_idx][0]

    # Verify asymmetry (from fixture: L0→L1=1.0, L1→L0=10.0)
    assert np.isclose(l0_to_l1_dist, 1.0), (
        f"L0→L1 distance should be 1.0, got {l0_to_l1_dist}"
    )
    assert np.isclose(l1_to_l0_dist, 10.0), (
        f"L1→L0 distance should be 10.0, got {l1_to_l0_dist}"
    )
    assert l0_to_l1_dist != l1_to_l0_dist, "Distances should be asymmetric"


def test_3d_single_item_scenario():
    """Verify 3D builder works with minimal graph (single item)."""
    from slotting_optimization.gnn_builder import build_graph_3d_sparse

    # Create minimal setup with 1 item, 3 locations (1 storage + start + end)
    w = Warehouse(
        locations=["L0", "start", "end"], start_point_id="start", end_point_id="end"
    )

    w.set_distance("start", "L0", 5.0)
    w.set_distance("start", "end", 10.0)
    w.set_distance("L0", "end", 3.0)
    w.set_distance("L0", "start", 4.0)
    w.set_distance("end", "start", 2.0)
    w.set_distance("end", "L0", 6.0)

    ob = OrderBook.from_orders([make_order("O1", "I1", "2024-01-01T10:00:00")])

    il = ItemLocations.from_records([{"item_id": "I1", "location_id": "L0"}])

    data = build_graph_3d_sparse(ob, il, w)

    # Should have 1 item node + 3 location nodes = 4 nodes
    assert data.num_nodes == 4
    assert data.n_items == 1
    assert data.n_locs == 3

    # Verify edge attributes are 3D
    assert data.edge_attr.shape[1] == 3
    assert data.edge_type_mask.shape[1] == 3


def test_3d_empty_sequence_matrix():
    """Verify 3D builder handles orders with no repeated items (zero seq_mat)."""
    from slotting_optimization.gnn_builder import build_graph_3d_sparse

    # Create setup where no item follows another (all separate orders)
    w = Warehouse(
        locations=["L0", "L1", "start", "end"],
        start_point_id="start",
        end_point_id="end",
    )

    # Set distances
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

    # Each order has only one item (no sequences)
    ob = OrderBook.from_orders(
        [
            make_order("O1", "I1", "2024-01-01T10:00:00"),
            make_order("O2", "I2", "2024-01-01T10:01:00"),
        ]
    )

    il = ItemLocations.from_records(
        [{"item_id": "I1", "location_id": "L0"}, {"item_id": "I2", "location_id": "L1"}]
    )

    data = build_graph_3d_sparse(ob, il, w)

    # Verify no Item→Item edges (seq_mat is all zeros)
    edges = data.edge_index.numpy()
    I = data.n_items

    src_is_item = edges[0] < I
    dst_is_item = edges[1] < I
    item_to_item_mask = src_is_item & dst_is_item

    # Should have no Item→Item edges in sparse mode
    assert not item_to_item_mask.any(), (
        "Should have no Item→Item edges when seq_mat is zero"
    )


def test_3d_dense_includes_zero_sequence_edges():
    """Verify 3D dense mode includes Item→Item edges even with zero counts."""
    from slotting_optimization.gnn_builder import build_graph_3d_dense

    # Same setup as above (no sequences)
    w = Warehouse(
        locations=["L0", "L1", "start", "end"],
        start_point_id="start",
        end_point_id="end",
    )

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

    ob = OrderBook.from_orders(
        [
            make_order("O1", "I1", "2024-01-01T10:00:00"),
            make_order("O2", "I2", "2024-01-01T10:01:00"),
        ]
    )

    il = ItemLocations.from_records(
        [{"item_id": "I1", "location_id": "L0"}, {"item_id": "I2", "location_id": "L1"}]
    )

    data = build_graph_3d_dense(ob, il, w)

    # Verify Item→Item edges exist in dense mode (even if zero)
    edges = data.edge_index.numpy()
    attrs = data.edge_attr.numpy()
    masks = data.edge_type_mask.numpy()
    I = data.n_items

    src_is_item = edges[0] < I
    dst_is_item = edges[1] < I
    item_to_item_mask = src_is_item & dst_is_item

    # Dense mode should have Item→Item edges
    assert item_to_item_mask.any(), "Dense mode should include Item→Item edges"

    # All Item→Item edge values should be 0.0 (dimension 1)
    for idx in np.where(item_to_item_mask)[0]:
        assert masks[idx][1], "Item→Item edge should have mask[1]=True"
        assert attrs[idx][1] == 0.0, (
            "Item→Item edge should have attr[1]=0.0 (no sequences)"
        )


# =============================================================================
# Group 6: NaN Validation (2 tests)
# =============================================================================


def test_3d_sparse_nan_validation_triggers():
    """Verify NaN from missing warehouse distance raises ValueError."""
    from slotting_optimization.gnn_builder import build_graph_3d_sparse

    # Create warehouse with MISSING distance (will produce NaN)
    w = Warehouse(
        locations=["L0", "start", "end"], start_point_id="start", end_point_id="end"
    )

    w.set_distance("start", "L0", 5.0)
    w.set_distance("start", "end", 10.0)
    w.set_distance("L0", "end", 3.0)
    w.set_distance("L0", "start", 4.0)
    # MISSING: w.set_distance("end", "start", 2.0) - will produce NaN!
    w.set_distance("end", "L0", 6.0)

    ob = OrderBook.from_orders([make_order("O1", "I1", "2024-01-01T10:00:00")])

    il = ItemLocations.from_records([{"item_id": "I1", "location_id": "L0"}])

    # Should raise ValueError during internal matrix building
    with pytest.raises(ValueError, match="NaN detected"):
        build_graph_3d_sparse(ob, il, w)


def test_3d_dense_nan_validation_can_be_disabled():
    """Verify validate_nan=False skips NaN check in 3D builder."""
    from slotting_optimization.gnn_builder import build_graph_3d_dense

    # Create warehouse with MISSING distance (will produce NaN)
    w = Warehouse(
        locations=["L0", "start", "end"], start_point_id="start", end_point_id="end"
    )

    w.set_distance("start", "L0", 5.0)
    w.set_distance("start", "end", 10.0)
    w.set_distance("L0", "end", 3.0)
    w.set_distance("L0", "start", 4.0)
    # MISSING: w.set_distance("end", "start", 2.0)
    w.set_distance("end", "L0", 6.0)

    ob = OrderBook.from_orders([make_order("O1", "I1", "2024-01-01T10:00:00")])

    il = ItemLocations.from_records([{"item_id": "I1", "location_id": "L0"}])

    # Should not raise with validate_nan=False
    data = build_graph_3d_dense(ob, il, w, validate_nan=False)
    assert data is not None


# =============================================================================
# Group 7: Simulation Target (2 tests)
# =============================================================================


def test_3d_sparse_with_simulator_adds_y(full_setup):
    """Verify 3D sparse graph includes y attribute when simulator provided."""
    from slotting_optimization.gnn_builder import build_graph_3d_sparse

    ob, il, w = full_setup
    sim = Simulator()

    data = build_graph_3d_sparse(ob, il, w, simulator=sim.simulate)

    # Verify y attribute exists
    assert hasattr(data, "y"), "Data should have y attribute when simulator provided"
    assert data.y.shape == (1,), "y should have shape [1]"
    assert data.y.dtype == torch.float, "y should be torch.float"
    assert data.y.item() > 0, "y (total distance) should be positive"


def test_3d_dense_y_value_matches_simulation(full_setup):
    """Verify 3D dense y value matches direct simulation result."""
    from slotting_optimization.gnn_builder import build_graph_3d_dense

    ob, il, w = full_setup
    sim = Simulator()

    # Build graph with simulator
    data = build_graph_3d_dense(ob, il, w, simulator=sim.simulate)

    # Run simulation separately
    total_distance, _ = sim.simulate(ob, w, il)

    # Verify y matches simulation
    assert hasattr(data, "y"), "Data should have y attribute"
    assert np.isclose(data.y.item(), total_distance), (
        f"y value {data.y.item()} should match simulation result {total_distance}"
    )


# =============================================================================
# Group 8: Integration Tests (2 tests)
# =============================================================================


def test_3d_graph_with_real_sample_data():
    """Test 3D graph builder with real sample data files."""
    from slotting_optimization.gnn_builder import build_graph_3d_sparse
    import os

    # Check if sample data exists
    project_root = os.path.dirname(os.path.dirname(__file__))
    sample_dir = os.path.join(project_root, "slotting_optimization", "data")

    orders_file = os.path.join(sample_dir, "sample_orders.csv")
    locations_file = os.path.join(sample_dir, "sample_item_locations.csv")

    if not (os.path.exists(orders_file) and os.path.exists(locations_file)):
        pytest.skip("Sample data files not found")

    # Load sample data
    ob = OrderBook.load_csv(orders_file)
    il = ItemLocations.load_csv(locations_file)

    # Create simple warehouse for sample locations
    # (Assuming sample has locations L0-L9)
    all_locs = [f"L{i}" for i in range(10)] + ["start", "end"]
    w = Warehouse(locations=all_locs, start_point_id="start", end_point_id="end")

    # Set basic distances (simple grid)
    for i in range(10):
        w.set_distance("start", f"L{i}", 5.0 + i * 0.5)
        w.set_distance(f"L{i}", "end", 3.0 + i * 0.3)
        w.set_distance(f"L{i}", "start", 4.0 + i * 0.4)
        w.set_distance("end", f"L{i}", 6.0 + i * 0.6)
        for j in range(10):
            if i != j:
                w.set_distance(f"L{i}", f"L{j}", abs(i - j) * 1.0)

    w.set_distance("start", "end", 10.0)
    w.set_distance("end", "start", 2.0)

    # Build 3D graph
    data = build_graph_3d_sparse(ob, il, w)

    # Basic validation
    assert data.num_nodes > 0
    assert data.edge_index.shape[1] > 0
    assert data.edge_attr.shape == (data.edge_index.shape[1], 3)
    assert data.edge_type_mask.shape == (data.edge_index.shape[1], 3)

    # Verify all edges have exactly one type
    masks = data.edge_type_mask.numpy()
    assert all(masks.sum(axis=1) == 1), "Each edge should have exactly one type"


def test_3d_graph_large_scale():
    """Test 3D builder with larger graph (performance/scalability check)."""
    from slotting_optimization.gnn_builder import build_graph_3d_sparse
    from slotting_optimization.generator import DataGenerator

    # Generate larger dataset (30 items, 30 storage locations)
    gen = DataGenerator()
    samples = gen.generate_samples(
        n_locations=30,  # 30 storage locations (start + end added automatically)
        nb_items=30,
        n_orders=100,
        min_items_per_order=1,
        max_items_per_order=5,
        n_samples=1,
        seed=42,
    )

    ob, il, w = samples[0]

    # Build 3D graph
    data = build_graph_3d_sparse(ob, il, w)

    # Basic validation
    expected_nodes = 30 + 32  # items + (30 storage + start + end)
    assert data.num_nodes == expected_nodes
    assert data.n_items == 30
    assert data.n_locs == 32  # 30 storage + start + end

    # Verify 3D structure
    assert data.edge_attr.shape[1] == 3
    assert data.edge_type_mask.shape[1] == 3

    # Verify edge type distribution makes sense
    masks = data.edge_type_mask.numpy()
    loc_to_loc_count = masks[:, 0].sum()
    item_to_loc_count = masks[:, 2].sum()

    # Sanity checks
    assert loc_to_loc_count > 0, "Should have Location→Location edges"
    assert item_to_loc_count > 0, "Should have Item→Location edges"
    # Item→Item count can be 0 or more depending on order sequences

    # Verify mutually exclusive
    assert all(masks.sum(axis=1) == 1), "Each edge should have exactly one type"
