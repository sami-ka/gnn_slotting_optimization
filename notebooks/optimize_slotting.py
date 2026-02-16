"""
Gradient-based optimization of warehouse item placement using trained GNN.

Uses backpropagation through the GNN to find optimal item-to-location assignments
that minimize predicted travel distance - similar to adversarial attacks.

Usage:
    cd notebooks && uv run optimize_slotting.py --sample_idx 0
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.data import Data
from scipy.optimize import linear_sum_assignment
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slotting_optimization.simulator import Simulator
from slotting_optimization.item_locations import ItemLocations


# ============================================================================
# Model Architecture (must match training)
# ============================================================================


class EdgeThenNodeLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr="add")
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        edge_attr = self.edge_mlp(torch.cat([x[row], x[col], edge_attr], dim=1))
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x, edge_attr

    def message(self, x_j, edge_attr):
        return self.node_mlp(torch.cat([x_j, edge_attr], dim=1))


class NodeThenEdgeLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr="add")
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        row, col = edge_index
        edge_attr = self.edge_mlp(torch.cat([x[row], x[col], edge_attr], dim=1))
        return x, edge_attr

    def message(self, x_j, edge_attr):
        return self.node_mlp(torch.cat([x_j, edge_attr], dim=1))


class GCNBlock(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.edge_then_node = EdgeThenNodeLayer(node_dim, edge_dim)
        self.node_then_edge = NodeThenEdgeLayer(node_dim, edge_dim)

    def forward(self, x, edge_index, edge_attr):
        x, edge_attr = self.edge_then_node(x, edge_index, edge_attr)
        x, edge_attr = self.node_then_edge(x, edge_index, edge_attr)
        return x, edge_attr


class GraphRegressionModel(nn.Module):
    def __init__(self, hidden_dim, edge_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_embedding = nn.Embedding(256, hidden_dim)  # per-node-index embedding
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GCNBlock(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        """Forward pass with explicit inputs (for gradient optimization).

        Args:
            x: Node features [num_nodes, hidden_dim] - can be from node_embedding
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, 3]
            batch: Batch assignment [num_nodes] (optional, defaults to single graph)
        """
        edge_attr_enc = self.edge_encoder(edge_attr)
        for layer in self.layers:
            x, edge_attr_enc = layer(x, edge_index, edge_attr_enc)
        graph_emb = global_add_pool(x, batch)
        out = self.regressor(graph_emb)
        return out.squeeze(-1)


# ============================================================================
# Sinkhorn Algorithm (Soft Permutation)
# ============================================================================


def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20, temperature: float = 0.1):
    """
    Convert logits to doubly-stochastic matrix via Sinkhorn iterations.

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


# ============================================================================
# Graph Manipulation for Dense Item-Location Edges
# ============================================================================


def create_dense_assignment_graph(data: Data, n_items: int, n_storage: int):
    """
    Create a new graph with ALL possible Item->Location edges (dense).

    Original graph has sparse Item->Loc edges (only current assignments).
    We need dense edges for gradient optimization over all possible assignments.

    Returns:
        new_data: Data with dense Item->Loc edges
        edge_info: Dict with indices for assignment edge manipulation
    """
    # Extract existing edges by type
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    edge_type_mask = data.edge_type_mask

    loc_loc_mask = edge_type_mask[:, 0]
    item_item_mask = edge_type_mask[:, 1]
    item_loc_mask = edge_type_mask[:, 2]

    # Get normalized values for Item->Loc edges from existing data
    # These are the normalized constants for dims 0, 1 (should be same for all Item->Loc edges)
    existing_item_loc_attr = edge_attr[item_loc_mask]
    if len(existing_item_loc_attr) > 0:
        dim0_norm = existing_item_loc_attr[0, 0].item()  # Normalized 0 for distance
        dim1_norm = existing_item_loc_attr[0, 1].item()  # Normalized 0 for sequence
        dim2_assigned = existing_item_loc_attr[0, 2].item()  # Normalized 1 (assigned)
    else:
        # Fallback values (shouldn't happen)
        dim0_norm = -0.89
        dim1_norm = -0.73
        dim2_assigned = 6.44

    # Compute normalized value for "not assigned" (raw=0)
    # From the data: assigned (raw=1) -> 6.44, not_assigned (raw=0) -> -0.16
    # This is (raw - mean) / std normalization
    # We can compute: dim2_not_assigned = dim2_assigned - 1/std
    # But simpler: just compute from data statistics
    all_dim2 = edge_attr[:, 2]
    # The min value (excluding assigned) is the "not assigned" normalized value
    dim2_not_assigned = all_dim2[~item_loc_mask].min().item() if (~item_loc_mask).any() else -0.16

    # Keep Loc->Loc and Item->Item edges unchanged
    keep_mask = loc_loc_mask | item_item_mask
    kept_edge_index = edge_index[:, keep_mask]
    kept_edge_attr = edge_attr[keep_mask]
    kept_edge_type_mask = edge_type_mask[keep_mask]

    # Create ALL Item->Location edges (dense: n_items * n_storage)
    # Items are nodes 0 to n_items-1
    # Locations are nodes n_items to n_items+n_storage-1
    item_indices = []
    loc_indices = []
    for item_idx in range(n_items):
        for loc_idx in range(n_storage):
            item_indices.append(item_idx)
            loc_indices.append(n_items + loc_idx)

    new_item_loc_edges = torch.tensor(
        [item_indices, loc_indices], dtype=torch.long
    )

    # Create edge attributes for new Item->Loc edges with proper normalization
    n_new_edges = n_items * n_storage
    new_edge_attr = torch.zeros(n_new_edges, 3)
    new_edge_attr[:, 0] = dim0_norm  # Normalized 0 for distance
    new_edge_attr[:, 1] = dim1_norm  # Normalized 0 for sequence
    new_edge_attr[:, 2] = dim2_not_assigned  # Will be updated by soft assignment

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
    """
    Inject soft assignment values into edge_attr dimension 2.

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

    # Map soft assignment [0, 1] to normalized space
    # 0 -> dim2_not_assigned, 1 -> dim2_assigned
    dim2_assigned = edge_info["dim2_assigned"]
    dim2_not_assigned = edge_info["dim2_not_assigned"]

    # Linear interpolation: normalized = not_assigned + soft * (assigned - not_assigned)
    flat_assignment = soft_assignment.reshape(-1)
    normalized_assignment = (
        dim2_not_assigned + flat_assignment * (dim2_assigned - dim2_not_assigned)
    )

    new_edge_attr[start_idx : start_idx + n_edges, 2] = normalized_assignment

    return new_edge_attr


def extract_current_assignment(data: Data) -> np.ndarray:
    """
    Extract current item->location assignment from sparse graph.

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


# ============================================================================
# Optimization
# ============================================================================


def optimize_assignment(
    model: nn.Module,
    data: Data,
    mean_y: float,
    std_y: float,
    n_steps: int = 100,
    lr: float = 0.5,
    initial_temp: float = 1.0,
    final_temp: float = 0.01,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Optimize item-location assignment via gradient descent through GNN.

    Args:
        model: Trained GNN model
        data: Original graph data (sparse assignment edges)
        mean_y, std_y: Normalization params for denormalizing predictions
        n_steps: Number of optimization steps
        lr: Learning rate
        initial_temp, final_temp: Temperature annealing range
        seed: Random seed for deterministic node initialization
        verbose: Print progress

    Returns:
        result: Dict with optimization results
    """
    model.eval()
    n_items = data.n_items
    n_storage = data.n_storage

    # Get current assignment for comparison
    current_assignment = extract_current_assignment(data)

    # Create dense graph for optimization
    dense_data, edge_info = create_dense_assignment_graph(data, n_items, n_storage)

    # Initialize assignment logits from current assignment
    log_alpha = torch.zeros(n_items, n_storage, requires_grad=True)
    # Bias toward current assignment
    for item_idx, loc_idx in enumerate(current_assignment):
        log_alpha.data[item_idx, loc_idx] = 2.0

    optimizer = torch.optim.Adam([log_alpha], lr=lr)

    # Temperature annealing schedule
    temp_decay = (final_temp / initial_temp) ** (1.0 / n_steps)
    temp = initial_temp

    # Compute node features from node embedding (same as training)
    # Detach since we're optimizing log_alpha, not model weights
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
        print(f"\nOptimizing (temp: {initial_temp:.3f} -> {final_temp:.3f})...")

    history = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Sinkhorn to get soft permutation
        soft_assign = sinkhorn(log_alpha, n_iters=20, temperature=temp)

        # Inject into edge attributes
        edge_attr = inject_soft_assignment(dense_data.edge_attr, soft_assign, edge_info)

        # Forward pass using node embedding features
        # Note: x is computed once before the loop and reused
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
    final_edge_attr = inject_soft_assignment(dense_data.edge_attr, final_hard, edge_info)

    # Use same node embedding features for final scoring
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
    }


# ============================================================================
# Main
# ============================================================================


def swap_assignment(assignment: np.ndarray, i: int, j: int) -> np.ndarray:
    """Swap locations of items i and j."""
    new_assignment = assignment.copy()
    new_assignment[i], new_assignment[j] = new_assignment[j], new_assignment[i]
    return new_assignment


def apply_assignment_to_data(data: Data, assignment: np.ndarray) -> Data:
    """
    Create a new Data object with modified Item->Loc edges for the given assignment.

    Maintains sparse structure (only assigned edges have value 1).
    """
    n_items = data.n_items

    # Clone data
    new_edge_index = data.edge_index.clone()
    new_edge_attr = data.edge_attr.clone()

    # Find Item->Loc edges and update them
    edge_type_mask = data.edge_type_mask
    item_loc_mask = edge_type_mask[:, 2]
    item_loc_indices = torch.where(item_loc_mask)[0]

    # Update edge targets to reflect new assignment
    for edge_idx in item_loc_indices:
        item_idx = new_edge_index[0, edge_idx].item()
        new_loc_idx = assignment[item_idx]
        new_edge_index[1, edge_idx] = n_items + new_loc_idx

    return Data(
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        edge_type_mask=data.edge_type_mask,
        num_nodes=data.num_nodes,
        n_items=data.n_items,
        n_locs=data.n_locs,
        n_storage=data.n_storage,
    )


def score_assignment_sparse(
    model: nn.Module,
    data: Data,
    assignment: np.ndarray,
    mean_y: float,
    std_y: float,
    seed: int = 42,
) -> float:
    """Score an assignment using sparse graph structure."""
    modified_data = apply_assignment_to_data(data, assignment)

    # Compute node features from node embedding (same as training)
    n_items = data.n_items
    n_locs = data.n_locs
    nodes_per_graph = n_items + n_locs
    node_ids = torch.arange(modified_data.num_nodes) % nodes_per_graph
    x = model.node_embedding(node_ids)

    with torch.no_grad():
        pred_norm = model(x, modified_data.edge_index, modified_data.edge_attr)
        pred = pred_norm.item() * std_y + mean_y

    return pred


def optimize_swaps(
    model: nn.Module,
    data: Data,
    mean_y: float,
    std_y: float,
    max_iters: int = 50,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Optimize assignment via greedy swap search.

    At each iteration, try all pairwise swaps and apply the best improvement.
    """
    model.eval()
    n_items = data.n_items

    # Get current assignment
    current_assignment = extract_current_assignment(data)

    # Score original
    original_score = score_assignment_sparse(model, data, current_assignment, mean_y, std_y, seed)

    if verbose:
        print(f"Original predicted distance: {original_score:.1f}")
        print(f"\nSearching for improving swaps...")

    best_assignment = current_assignment.copy()
    best_score = original_score
    history = [original_score]

    for iteration in range(max_iters):
        improved = False
        best_swap = None
        best_swap_score = best_score

        # Try all swaps
        for i in range(n_items):
            for j in range(i + 1, n_items):
                swapped = swap_assignment(best_assignment, i, j)
                score = score_assignment_sparse(model, data, swapped, mean_y, std_y, seed)

                if score < best_swap_score:
                    best_swap_score = score
                    best_swap = (i, j)

        if best_swap is not None and best_swap_score < best_score - 0.1:
            i, j = best_swap
            best_assignment = swap_assignment(best_assignment, i, j)
            improvement = best_score - best_swap_score
            best_score = best_swap_score
            improved = True
            history.append(best_score)

            if verbose:
                print(f"  Iter {iteration}: Swap items {i}<->{j}, "
                      f"distance: {best_score:.1f} (improved by {improvement:.1f})")
        else:
            if verbose:
                print(f"  Iter {iteration}: No improving swap found. Stopping.")
            break

    final_improvement = (original_score - best_score) / original_score * 100

    if verbose:
        print(f"\nFinal predicted distance: {best_score:.1f}")
        print(f"Improvement: {final_improvement:.2f}%")

    return {
        "original_assignment": current_assignment,
        "optimized_assignment": best_assignment,
        "original_distance": original_score,
        "optimized_distance": best_score,
        "improvement_pct": final_improvement,
        "history": history,
    }


# ============================================================================
# Simulator Verification
# ============================================================================


def verify_with_simulator(
    original_assignment: np.ndarray,
    optimized_assignment: np.ndarray,
    raw_sample: tuple,  # (OrderBook, ItemLocations, Warehouse)
) -> dict:
    """Run both assignments through real simulator and compare."""
    order_book, original_il, warehouse = raw_sample
    simulator = Simulator()

    # Get item IDs from original ItemLocations (sorted to match graph node order)
    items = sorted(original_il.to_dict().keys())

    # Original distance (should match ground truth)
    orig_dist, _ = simulator.simulate(order_book, warehouse, original_il)

    # Build new ItemLocations with optimized assignment
    # Map assignment indices back to location IDs
    storage_locs = sorted([loc for loc in warehouse.locations()
                           if loc not in (warehouse.start_point, warehouse.end_point)])

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


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Optimize warehouse slotting via GNN")
    parser.add_argument("--sample_idx", type=int, default=0, help="Test sample index")
    parser.add_argument("--n_steps", type=int, default=100, help="Optimization steps")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate")
    parser.add_argument("--method", type=str, default="swap", choices=["swap", "gradient"],
                        help="Optimization method")
    parser.add_argument("--model", type=str, default="model_cpu.pt", help="Model checkpoint path")
    parser.add_argument("--data", type=str, default="test_dataset_cpu.pt", help="Test data path")
    args = parser.parse_args()

    print("=" * 60)
    print("GNN-Based Slotting Optimization")
    print("=" * 60)

    # Load model
    device = torch.device("cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    # Get model config (support both old and new checkpoint formats)
    if "config" in checkpoint:
        config = checkpoint["config"]
        hidden_dim = config["hidden_dim"]
        num_layers = config["num_layers"]
    else:
        hidden_dim = 64
        num_layers = 5

    model = GraphRegressionModel(hidden_dim=hidden_dim, edge_dim=3, num_layers=num_layers)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean_y = checkpoint["mean_y"]
    std_y = checkpoint["std_y"]

    print(f"Model loaded. Normalization: mean={mean_y:.1f}, std={std_y:.1f}")

    # Load test data
    test_data = torch.load(args.data, map_location=device, weights_only=False)
    print(f"Test dataset: {len(test_data)} samples")

    # Select sample
    sample = test_data[args.sample_idx]
    print(f"\nSample {args.sample_idx}:")
    print(f"  Items: {sample.n_items}, Storage locations: {sample.n_storage}")
    ground_truth = sample.y.item() * std_y + mean_y
    print(f"  Ground truth distance (simulator): {ground_truth:.1f}")

    # Optimize
    if args.method == "swap":
        result = optimize_swaps(
            model=model,
            data=sample,
            mean_y=mean_y,
            std_y=std_y,
            max_iters=args.n_steps,
        )
    else:
        result = optimize_assignment(
            model=model,
            data=sample,
            mean_y=mean_y,
            std_y=std_y,
            n_steps=args.n_steps,
            lr=args.lr,
        )

    # Print assignment changes
    print("\n" + "=" * 60)
    print("Assignment Changes:")
    print("=" * 60)
    orig = result["original_assignment"]
    opt = result["optimized_assignment"]
    changes = 0
    for i in range(len(orig)):
        if orig[i] != opt[i]:
            print(f"  Item {i}: Loc {orig[i]} -> Loc {opt[i]}")
            changes += 1
    print(f"\nTotal items moved: {changes}/{len(orig)}")

    # Verify with real simulator
    try:
        raw_samples = torch.load("test_samples_cpu.pt", map_location=device, weights_only=False)
        raw_sample = raw_samples[args.sample_idx]

        sim_result = verify_with_simulator(
            result["original_assignment"],
            result["optimized_assignment"],
            raw_sample,
        )

        print("\n" + "=" * 60)
        print("Simulator Verification:")
        print("=" * 60)
        print(f"  Original (simulator):  {sim_result['original_sim_distance']:.1f}")
        print(f"  Optimized (simulator): {sim_result['optimized_sim_distance']:.1f}")
        print(f"  Real improvement:      {sim_result['sim_improvement_pct']:.2f}%")
    except FileNotFoundError:
        print("\nNote: test_samples_cpu.pt not found. Re-run train_cpu.py to enable simulator verification.")


if __name__ == "__main__":
    main()
