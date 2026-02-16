"""Inverse optimization for generating diverse optimized assignments.

This module provides gradient-based inverse optimization to find optimal item-location
assignments by backpropagating through a trained GNN model. It supports generating
diverse solutions via combinatorial parameter grids (seeds, initialization noise,
temperature schedules).

Usage:
    from slotting_optimization.inverse_optimizer import (
        DiversityConfig,
        optimize_assignment_with_noise,
        generate_diverse_optimizations,
        assignment_to_graph_data,
    )

    config = DiversityConfig()  # Default ~30 candidates per sample
    optimizations = generate_diverse_optimizations(
        model, data, raw_sample, mean_y, std_y, edge_mean, edge_std, config
    )
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from scipy.optimize import linear_sum_assignment

from .simulator import Simulator
from .item_locations import ItemLocations
from .gnn_builder import build_graph_3d_sparse


@dataclass
class DiversityConfig:
    """Configuration for diverse optimization runs.

    Default configuration generates ~30 candidates per sample:
    5 seeds x 2 noise_scales x 3 temp_schedules = 30

    Attributes:
        seeds: Random seeds for initialization noise
        init_noise_scales: Noise magnitude added to log_alpha initialization
        temperature_schedules: List of (initial_temp, final_temp) pairs
        n_steps: Number of optimization steps per run
        lr: Learning rate for Adam optimizer
    """

    seeds: List[int] = field(default_factory=lambda: list(range(5)))
    init_noise_scales: List[float] = field(default_factory=lambda: [0.0, 0.5])
    temperature_schedules: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.5, 0.001), (1.0, 0.01), (2.0, 0.001)]
    )
    n_steps: int = 100
    lr: float = 0.5


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


def optimize_assignment_with_noise(
    model: nn.Module,
    data: Data,
    mean_y: float,
    std_y: float,
    n_steps: int = 100,
    lr: float = 0.5,
    initial_temp: float = 1.0,
    final_temp: float = 0.01,
    seed: int = 42,
    init_noise_scale: float = 0.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Optimize item-location assignment via gradient descent through GNN.

    Similar to optimize_assignment but with init_noise_scale parameter for
    creating diverse starting points in the optimization landscape.

    Args:
        model: Trained GNN model
        data: Original graph data (sparse assignment edges)
        mean_y, std_y: Normalization params for denormalizing predictions
        n_steps: Number of optimization steps
        lr: Learning rate
        initial_temp, final_temp: Temperature annealing range
        seed: Random seed for deterministic noise initialization
        init_noise_scale: Scale of random noise added to initial log_alpha
        verbose: Print progress

    Returns:
        Dict with optimization results including:
            - original_assignment: Starting assignment
            - optimized_assignment: Final discrete assignment
            - original_distance: Predicted distance for original
            - optimized_distance: Predicted distance for optimized
            - improvement_pct: Percentage improvement
            - history: List of predicted distances during optimization
    """
    model.eval()
    n_items = data.n_items
    n_storage = data.n_storage

    # Get current assignment for comparison
    current_assignment = extract_current_assignment(data)

    # Create dense graph for optimization
    dense_data, edge_info = create_dense_assignment_graph(data, n_items, n_storage)

    # Initialize assignment logits with noise for diversity
    torch.manual_seed(seed)
    log_alpha = torch.randn(n_items, n_storage) * init_noise_scale
    log_alpha.requires_grad = True

    # Bias toward current assignment
    for item_idx, loc_idx in enumerate(current_assignment):
        log_alpha.data[item_idx, loc_idx] += 2.0

    optimizer = torch.optim.Adam([log_alpha], lr=lr)

    # Temperature annealing schedule
    temp_decay = (final_temp / initial_temp) ** (1.0 / n_steps)
    temp = initial_temp

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

    if verbose:
        print(f"Original assignment predicted distance: {original_pred:.1f}")
        print(f"Optimizing (temp: {initial_temp:.3f} -> {final_temp:.3f})...")

    history = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Sinkhorn to get soft permutation
        soft_assign = sinkhorn(log_alpha, n_iters=20, temperature=temp)

        # Inject into edge attributes
        edge_attr = inject_soft_assignment(dense_data.edge_attr, soft_assign, edge_info)

        # Forward pass
        pred_norm = model(x, dense_data.edge_index, edge_attr)

        # Backward pass
        pred_norm.backward()
        optimizer.step()

        # Denormalize for logging
        pred_dist = pred_norm.item() * std_y + mean_y
        history.append(pred_dist)

        if verbose and (step % 20 == 0 or step == n_steps - 1):
            print(f"  Step {step:3d}: distance={pred_dist:.1f}, temp={temp:.4f}")

        # Anneal temperature
        temp *= temp_decay

    # Final discretization via Hungarian algorithm
    with torch.no_grad():
        final_soft = sinkhorn(log_alpha, n_iters=50, temperature=0.001)
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
        "history": history,
        "config": {
            "seed": seed,
            "init_noise_scale": init_noise_scale,
            "initial_temp": initial_temp,
            "final_temp": final_temp,
        },
    }


def compute_assignment_similarity(a1: np.ndarray, a2: np.ndarray) -> float:
    """Compute similarity between two assignments (fraction of matching locations)."""
    return (a1 == a2).sum() / len(a1)


def deduplicate_assignments(
    results: List[Dict[str, Any]], threshold: float = 0.8
) -> List[Dict[str, Any]]:
    """Remove near-duplicate assignments based on similarity threshold.

    Args:
        results: List of optimization result dictionaries
        threshold: Similarity threshold above which assignments are considered duplicates

    Returns:
        Filtered list with near-duplicates removed
    """
    if not results:
        return []

    unique_results = [results[0]]

    for result in results[1:]:
        assignment = result["optimized_assignment"]
        is_duplicate = False

        for unique_result in unique_results:
            similarity = compute_assignment_similarity(
                assignment, unique_result["optimized_assignment"]
            )
            if similarity > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_results.append(result)

    return unique_results


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


def generate_diverse_optimizations(
    model: nn.Module,
    data: Data,
    raw_sample: Tuple,  # (OrderBook, ItemLocations, Warehouse)
    mean_y: float,
    std_y: float,
    edge_mean: torch.Tensor,
    edge_std: torch.Tensor,
    config: DiversityConfig,
    simulator: Optional[Simulator] = None,
    dedup_threshold: float = 0.8,
    verbose: bool = False,
) -> List[Tuple[Data, Dict[str, Any]]]:
    """Generate multiple diverse optimized assignments.

    Uses combinatorial parameter grid (seeds, init_noise_scales, temperature_schedules)
    to find diverse local optima in the assignment space.

    Args:
        model: Trained GNN model
        data: Original graph data
        raw_sample: Tuple of (OrderBook, ItemLocations, Warehouse)
        mean_y, std_y: Target normalization params
        edge_mean, edge_std: Edge attribute normalization params
        config: DiversityConfig specifying parameter grid
        simulator: Optional Simulator instance (created if not provided)
        dedup_threshold: Similarity threshold for deduplication (0.8 = 80% same locations)
        verbose: Print progress

    Returns:
        List of (Data, metadata) tuples where:
            - Data: Normalized graph ready for training
            - metadata: Dict with optimization info (config, distances, improvement)
    """
    if simulator is None:
        simulator = Simulator()

    all_results = []

    # Generate all parameter combinations
    for seed in config.seeds:
        for noise_scale in config.init_noise_scales:
            for initial_temp, final_temp in config.temperature_schedules:
                result = optimize_assignment_with_noise(
                    model=model,
                    data=data,
                    mean_y=mean_y,
                    std_y=std_y,
                    n_steps=config.n_steps,
                    lr=config.lr,
                    initial_temp=initial_temp,
                    final_temp=final_temp,
                    seed=seed,
                    init_noise_scale=noise_scale,
                    verbose=False,
                )
                all_results.append(result)

    if verbose:
        print(f"  Generated {len(all_results)} candidates")

    # Deduplicate
    unique_results = deduplicate_assignments(all_results, threshold=dedup_threshold)

    if verbose:
        print(f"  After dedup: {len(unique_results)} unique")

    # Convert to graph Data objects
    output = []
    for result in unique_results:
        graph_data = assignment_to_graph_data(
            optimized_assignment=result["optimized_assignment"],
            raw_sample=raw_sample,
            simulator=simulator,
            edge_mean=edge_mean,
            edge_std=edge_std,
            mean_y=mean_y,
            std_y=std_y,
        )

        metadata = {
            "config": result["config"],
            "original_distance": result["original_distance"],
            "optimized_distance": result["optimized_distance"],
            "improvement_pct": result["improvement_pct"],
        }

        output.append((graph_data, metadata))

    return output


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
