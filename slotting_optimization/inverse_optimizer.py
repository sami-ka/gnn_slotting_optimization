"""Inverse optimization for generating optimized assignments.

This module provides gradient-based inverse optimization to find optimal item-location
assignments by backpropagating through a trained GNN model. The approach is analogous
to CNN activation maximization - a single gradient descent run on a continuous
relaxation, discretized at the end via the Hungarian algorithm.

Usage:
    from slotting_optimization.inverse_optimizer import (
        optimize_assignment,
        assignment_to_graph_data,
    )

    result = optimize_assignment(model, data, mean_y, std_y)
    graph_data = assignment_to_graph_data(
        result["optimized_assignment"], raw_sample, simulator, edge_mean, edge_std, mean_y, std_y
    )
"""

from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from scipy.optimize import linear_sum_assignment

from .simulator import Simulator
from .item_locations import ItemLocations
from .gnn_builder import build_graph_3d_sparse


def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20, temperature: float = 0.1):
    """Convert logits to doubly-stochastic matrix via Sinkhorn iterations.

    Args:
        log_alpha: [n_items, n_locations] raw logits
        n_iters: Number of normalization iterations
        temperature: Lower = sharper assignments

    Returns:
        Doubly-stochastic matrix (rows and columns sum to 1)
    """
    M = log_alpha / temperature
    for _ in range(n_iters):
        M = M - M.logsumexp(dim=1, keepdim=True)  # Row normalize
        M = M - M.logsumexp(dim=0, keepdim=True)  # Col normalize
    return M.exp()


def extract_current_assignment(data: Data) -> np.ndarray:
    """Extract current item->location assignment from sparse graph.

    Returns:
        assignment: [n_items] array where assignment[i] = location index for item i
    """
    edge_type_mask = data.edge_type_mask
    item_loc_mask = edge_type_mask[:, 2]
    item_loc_edges = data.edge_index[:, item_loc_mask]

    n_items = data.n_items
    assignment = np.zeros(n_items, dtype=np.int64)

    for i in range(item_loc_edges.shape[1]):
        item_idx = item_loc_edges[0, i].item()
        loc_node_idx = item_loc_edges[1, i].item()
        loc_idx = loc_node_idx - n_items  # Convert node index to storage index
        assignment[item_idx] = loc_idx

    return assignment


def create_dense_assignment_graph(data: Data, n_items: int, n_storage: int):
    """Create a new graph with ALL possible Item->Location edges (dense).

    Original graph has sparse Item->Loc edges (only current assignments).
    We need dense edges for gradient optimization over all possible assignments.

    Returns:
        new_data: Data with dense Item->Loc edges
        edge_info: Dict with indices for assignment edge manipulation
    """
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    edge_type_mask = data.edge_type_mask

    loc_loc_mask = edge_type_mask[:, 0]
    item_item_mask = edge_type_mask[:, 1]
    item_loc_mask = edge_type_mask[:, 2]

    # Get normalized values for Item->Loc edges from existing data
    existing_item_loc_attr = edge_attr[item_loc_mask]
    if len(existing_item_loc_attr) > 0:
        dim0_norm = existing_item_loc_attr[0, 0].item()  # Normalized 0 for distance
        dim1_norm = existing_item_loc_attr[0, 1].item()  # Normalized 0 for sequence
        dim2_assigned = existing_item_loc_attr[0, 2].item()  # Normalized 1 (assigned)
    else:
        dim0_norm = -0.89
        dim1_norm = -0.73
        dim2_assigned = 6.44

    # Compute normalized value for "not assigned" (raw=0)
    all_dim2 = edge_attr[:, 2]
    dim2_not_assigned = (
        all_dim2[~item_loc_mask].min().item() if (~item_loc_mask).any() else -0.16
    )

    # Keep Loc->Loc and Item->Item edges unchanged
    keep_mask = loc_loc_mask | item_item_mask
    kept_edge_index = edge_index[:, keep_mask]
    kept_edge_attr = edge_attr[keep_mask]
    kept_edge_type_mask = edge_type_mask[keep_mask]

    # Create ALL Item->Location edges (dense: n_items * n_storage)
    item_indices = []
    loc_indices = []
    for item_idx in range(n_items):
        for loc_idx in range(n_storage):
            item_indices.append(item_idx)
            loc_indices.append(n_items + loc_idx)

    new_item_loc_edges = torch.tensor([item_indices, loc_indices], dtype=torch.long)

    # Create edge attributes for new Item->Loc edges with proper normalization
    n_new_edges = n_items * n_storage
    new_edge_attr = torch.zeros(n_new_edges, 3)
    new_edge_attr[:, 0] = dim0_norm
    new_edge_attr[:, 1] = dim1_norm
    new_edge_attr[:, 2] = dim2_not_assigned

    # Create edge type mask for new edges
    new_edge_type_mask = torch.zeros(n_new_edges, 3, dtype=torch.bool)
    new_edge_type_mask[:, 2] = True  # All are Item->Loc type

    # Concatenate
    final_edge_index = torch.cat([kept_edge_index, new_item_loc_edges], dim=1)
    final_edge_attr = torch.cat([kept_edge_attr, new_edge_attr], dim=0)
    final_edge_type_mask = torch.cat([kept_edge_type_mask, new_edge_type_mask], dim=0)

    # Build edge info for quick assignment injection
    n_kept = kept_edge_index.shape[1]

    # Map (item_idx, loc_idx) to edge index
    item_loc_to_edge = {}
    for i, (item_idx, loc_idx) in enumerate(zip(item_indices, loc_indices)):
        item_loc_to_edge[(item_idx, loc_idx - n_items)] = n_kept + i

    new_data = Data(
        edge_index=final_edge_index,
        edge_attr=final_edge_attr,
        edge_type_mask=final_edge_type_mask,
        num_nodes=data.num_nodes,
        n_items=data.n_items,
        n_locs=data.n_locs,
        n_storage=data.n_storage,
    )

    edge_info = {
        "assignment_start_idx": n_kept,
        "n_assignment_edges": n_new_edges,
        "item_loc_to_edge": item_loc_to_edge,
        "n_items": n_items,
        "n_storage": n_storage,
        "dim2_assigned": dim2_assigned,
        "dim2_not_assigned": dim2_not_assigned,
    }

    return new_data, edge_info


def inject_soft_assignment(
    edge_attr: torch.Tensor,
    soft_assignment: torch.Tensor,
    edge_info: dict,
) -> torch.Tensor:
    """Inject soft assignment values into edge_attr dimension 2.

    Maps soft assignment [0, 1] to normalized values matching training data.

    Args:
        edge_attr: [n_edges, 3] edge attributes
        soft_assignment: [n_items, n_storage] doubly-stochastic matrix (values in [0, 1])
        edge_info: Dict from create_dense_assignment_graph

    Returns:
        Modified edge_attr with normalized assignment values in dim 2
    """
    new_edge_attr = edge_attr.clone()
    start_idx = edge_info["assignment_start_idx"]
    n_edges = edge_info["n_assignment_edges"]

    dim2_assigned = edge_info["dim2_assigned"]
    dim2_not_assigned = edge_info["dim2_not_assigned"]

    # Linear interpolation: normalized = not_assigned + soft * (assigned - not_assigned)
    flat_assignment = soft_assignment.reshape(-1)
    normalized_assignment = dim2_not_assigned + flat_assignment * (
        dim2_assigned - dim2_not_assigned
    )

    new_edge_attr[start_idx : start_idx + n_edges, 2] = normalized_assignment

    return new_edge_attr


def optimize_assignment(
    model: nn.Module,
    data: Data,
    mean_y: float,
    std_y: float,
    n_steps: int = None,
    lr: float = None,
    initial_temp: float = 1.0,
    final_temp: float = 0.01,
    verbose: bool = False,
    perturbation_scale: float = 0.0,
) -> Dict[str, Any]:
    """Single-run gradient-based assignment optimization.

    Analogous to CNN activation maximization - gradient descent on a continuous
    relaxation (Sinkhorn), discretized at the end via the Hungarian algorithm.
    Uses temperature annealing to balance exploration (high temp) and exploitation
    (low temp).

    Args:
        model: Trained GNN model
        data: Original graph data (sparse assignment edges)
        mean_y, std_y: Normalization params for denormalizing predictions
        n_steps: Number of optimization steps (auto-scaled if None)
        lr: Learning rate for gradient descent (auto-scaled if None)
        initial_temp: Starting temperature (higher = softer, better gradients)
        final_temp: Ending temperature (lower = sharper, closer to discrete)
        verbose: Print progress

    Returns:
        Dict with optimization results:
            - original_assignment: Starting assignment
            - optimized_assignment: Final discrete assignment
            - original_distance: Predicted distance for original
            - optimized_distance: Predicted distance for optimized
            - improvement_pct: Percentage improvement
    """
    model.eval()
    n_items = data.n_items
    n_storage = data.n_storage

    # Auto-scale based on problem size if not specified
    if n_steps is None:
        n_steps = max(50, n_items * 10)  # Scale with problem size
    if lr is None:
        lr = 0.5 / (n_items / 5.0)  # Scale inversely with size

    # Get current assignment for comparison
    current_assignment = extract_current_assignment(data)

    # Create dense graph for optimization
    dense_data, edge_info = create_dense_assignment_graph(data, n_items, n_storage)

    # Initialize assignment logits from current assignment (smaller bias for exploration)
    log_alpha = torch.zeros(n_items, n_storage, requires_grad=True)
    for item_idx, loc_idx in enumerate(current_assignment):
        log_alpha.data[item_idx, loc_idx] = 1.0
    if perturbation_scale > 0:
        log_alpha.data += torch.randn(n_items, n_storage) * perturbation_scale

    # Compute node features from node embedding (same as training)
    nodes_per_graph = n_items + data.n_locs
    node_ids = torch.arange(dense_data.num_nodes) % nodes_per_graph
    x = model.node_embedding(node_ids).detach()

    # Score original assignment first
    original_soft = torch.zeros(n_items, n_storage)
    for item_idx, loc_idx in enumerate(current_assignment):
        original_soft[item_idx, loc_idx] = 1.0
    original_edge_attr = inject_soft_assignment(
        dense_data.edge_attr, original_soft, edge_info
    )
    with torch.no_grad():
        original_pred_norm = model(x, dense_data.edge_index, original_edge_attr)
        original_pred = original_pred_norm.item() * std_y + mean_y

    # Temperature annealing schedule
    temp_decay = (final_temp / initial_temp) ** (1.0 / n_steps)
    temp = initial_temp

    if verbose:
        print(f"Original assignment predicted distance: {original_pred:.1f}")
        print(f"Optimizing ({n_steps} steps, temp: {initial_temp:.3f} -> {final_temp:.3f})...")

    # Gradient descent loop (manual steps, simpler than Adam for this use case)
    for step in range(n_steps):
        if log_alpha.grad is not None:
            log_alpha.grad.zero_()

        # Sinkhorn to get soft permutation (higher temp = softer = better gradients)
        sinkhorn_iters = max(20, n_items * 3)  # More iterations for larger matrices
        soft_assign = sinkhorn(log_alpha, n_iters=sinkhorn_iters, temperature=temp)

        # Inject into edge attributes
        edge_attr = inject_soft_assignment(dense_data.edge_attr, soft_assign, edge_info)

        # Forward pass
        pred_norm = model(x, dense_data.edge_index, edge_attr)

        # Backward pass
        pred_norm.backward()

        # Manual gradient step (minimize distance)
        log_alpha.data -= lr * log_alpha.grad

        if verbose and (step % 10 == 0 or step == n_steps - 1):
            pred_dist = pred_norm.item() * std_y + mean_y
            print(f"  Step {step:3d}: distance={pred_dist:.1f}, temp={temp:.4f}")

        # Anneal temperature
        temp *= temp_decay

    # Final discretization via Hungarian algorithm
    with torch.no_grad():
        final_iters = max(50, n_items * 8)
        final_soft = sinkhorn(log_alpha, n_iters=final_iters, temperature=0.01)
        cost_matrix = -final_soft.numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        final_assignment = col_ind

    # Score final discrete assignment
    final_hard = torch.zeros(n_items, n_storage)
    for item_idx, loc_idx in enumerate(final_assignment):
        final_hard[item_idx, loc_idx] = 1.0
    final_edge_attr = inject_soft_assignment(
        dense_data.edge_attr, final_hard, edge_info
    )

    with torch.no_grad():
        final_pred_norm = model(x, dense_data.edge_index, final_edge_attr)
        final_pred = final_pred_norm.item() * std_y + mean_y

    improvement = (original_pred - final_pred) / original_pred * 100

    if verbose:
        print(f"\nFinal (discrete) predicted distance: {final_pred:.1f}")
        print(f"Improvement: {improvement:.2f}%")

    return {
        "original_assignment": current_assignment,
        "optimized_assignment": final_assignment,
        "original_distance": original_pred,
        "optimized_distance": final_pred,
        "improvement_pct": improvement,
    }


def assignment_to_graph_data(
    optimized_assignment: np.ndarray,
    raw_sample: Tuple,  # (OrderBook, ItemLocations, Warehouse)
    simulator: Simulator,
    edge_mean: torch.Tensor,
    edge_std: torch.Tensor,
    mean_y: float,
    std_y: float,
) -> Data:
    """Convert optimized assignment to normalized graph Data.

    Creates a new graph with the optimized item-location assignment,
    runs the simulator to get the true distance, and applies normalization.

    Args:
        optimized_assignment: [n_items] array of location indices
        raw_sample: Tuple of (OrderBook, ItemLocations, Warehouse)
        simulator: Simulator instance for computing true distance
        edge_mean: Edge attribute normalization mean
        edge_std: Edge attribute normalization std
        mean_y: Target normalization mean
        std_y: Target normalization std

    Returns:
        Normalized Data object ready for training
    """
    order_book, original_il, warehouse = raw_sample

    # Get item IDs from original ItemLocations (sorted to match graph node order)
    items = sorted(original_il.to_dict().keys())

    # Get storage locations (excluding start/end)
    storage_locs = sorted(
        [
            loc
            for loc in warehouse.locations()
            if loc not in (warehouse.start_point, warehouse.end_point)
        ]
    )

    # Build new ItemLocations with optimized assignment
    new_records = []
    for i, item in enumerate(items):
        new_loc_idx = optimized_assignment[i]
        new_records.append({"item_id": item, "location_id": storage_locs[new_loc_idx]})

    new_il = ItemLocations.from_records(new_records)

    # Build graph with new assignment
    g_data = build_graph_3d_sparse(
        order_book=order_book,
        item_locations=new_il,
        warehouse=warehouse,
        simulator=simulator.simulate,
    )

    # Apply normalization (same as training)
    g_data.edge_attr = (g_data.edge_attr - edge_mean) / (edge_std + 1e-8)
    g_data.y = (g_data.y - mean_y) / std_y

    return g_data


def verify_with_simulator(
    original_assignment: np.ndarray,
    optimized_assignment: np.ndarray,
    raw_sample: Tuple,  # (OrderBook, ItemLocations, Warehouse)
) -> Dict[str, float]:
    """Run both assignments through real simulator and compare.

    Args:
        original_assignment: Original item-location assignment
        optimized_assignment: Optimized item-location assignment
        raw_sample: Tuple of (OrderBook, ItemLocations, Warehouse)

    Returns:
        Dict with simulator distances and improvement percentage
    """
    order_book, original_il, warehouse = raw_sample
    simulator = Simulator()

    # Get item IDs (sorted to match graph node order)
    items = sorted(original_il.to_dict().keys())

    # Original distance
    orig_dist, _ = simulator.simulate(order_book, warehouse, original_il)

    # Build new ItemLocations with optimized assignment
    storage_locs = sorted(
        [
            loc
            for loc in warehouse.locations()
            if loc not in (warehouse.start_point, warehouse.end_point)
        ]
    )

    new_records = []
    for i, item in enumerate(items):
        new_loc_idx = optimized_assignment[i]
        new_records.append({"item_id": item, "location_id": storage_locs[new_loc_idx]})

    new_il = ItemLocations.from_records(new_records)
    opt_dist, _ = simulator.simulate(order_book, warehouse, new_il)

    improvement = (orig_dist - opt_dist) / orig_dist * 100

    return {
        "original_sim_distance": orig_dist,
        "optimized_sim_distance": opt_dist,
        "sim_improvement_pct": improvement,
    }
