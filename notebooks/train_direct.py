"""
Train a model that directly computes the distance relationship.

The key insight is that total distance depends on:
- Which items are ordered together (seq_mat)
- Where those items are located (item_loc_mat)
- Distances between locations (loc_mat)

Distance ≈ sum over (item_i, item_j) pairs of: seq_count(i,j) * distance(loc[i], loc[j])

This script trains a model that learns this relationship directly.
"""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slotting_optimization.generator import DataGenerator
from slotting_optimization.gnn_builder import build_graph_3d_sparse
from slotting_optimization.simulator import Simulator


class DirectDistanceModel(nn.Module):
    """
    Model that directly computes distance from the matrix relationship.

    The model learns to predict distance as a weighted combination of:
    - Item-to-item distances weighted by sequence counts
    - Start/end distances for each item
    """

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learn to weight different components of the distance
        self.seq_weight = nn.Parameter(torch.ones(1))
        self.start_weight = nn.Parameter(torch.ones(1))
        self.end_weight = nn.Parameter(torch.ones(1))

        # Small MLP to refine the prediction
        self.refine = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        """
        Compute distance from graph data.

        We extract:
        - seq_mat: Item-Item sequence counts (from edge_attr dim 1)
        - loc_mat: Location-Location distances (from edge_attr dim 0)
        - item_loc: Item-Location assignments (from edge structure)
        """
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        edge_type_mask = data.edge_type_mask
        n_items = data.n_items
        n_storage = data.n_storage

        # Get Loc-Loc distances
        loc_loc_mask = edge_type_mask[:, 0]
        loc_edges = edge_index[:, loc_loc_mask]
        loc_dists = edge_attr[loc_loc_mask, 0]

        # Build loc_mat as sparse
        # Location nodes are n_items to n_items + n_locs
        n_locs = data.n_locs
        loc_mat = torch.zeros(n_locs, n_locs)
        for i in range(loc_edges.shape[1]):
            src = loc_edges[0, i].item() - n_items
            dst = loc_edges[1, i].item() - n_items
            if 0 <= src < n_locs and 0 <= dst < n_locs:
                loc_mat[src, dst] = loc_dists[i]

        # Get Item-Item sequences
        item_item_mask = edge_type_mask[:, 1]
        item_edges = edge_index[:, item_item_mask]
        seq_counts = edge_attr[item_item_mask, 1]

        seq_mat = torch.zeros(n_items, n_items)
        for i in range(item_edges.shape[1]):
            src = item_edges[0, i].item()
            dst = item_edges[1, i].item()
            if 0 <= src < n_items and 0 <= dst < n_items:
                seq_mat[src, dst] = seq_counts[i]

        # Get Item-Loc assignments
        item_loc_mask = edge_type_mask[:, 2]
        item_loc_edges = edge_index[:, item_loc_mask]

        # item_to_loc[item_idx] = storage_loc_idx
        item_to_loc = torch.zeros(n_items, dtype=torch.long)
        for i in range(item_loc_edges.shape[1]):
            item_idx = item_loc_edges[0, i].item()
            loc_idx = item_loc_edges[1, i].item() - n_items
            if 0 <= item_idx < n_items and 0 <= loc_idx < n_storage:
                item_to_loc[item_idx] = loc_idx

        # Compute distance components

        # 1. Item-to-item travel: for each (i,j) pair with seq_count > 0,
        #    add seq_count * distance(loc[i], loc[j])
        item_item_dist = 0.0
        for i in range(n_items):
            for j in range(n_items):
                if seq_mat[i, j] > 0:
                    loc_i = item_to_loc[i].item()
                    loc_j = item_to_loc[j].item()
                    item_item_dist += seq_mat[i, j] * loc_mat[loc_i, loc_j]

        # 2. Start distances: sum of (start -> item_loc) for each pick
        # Start is the second-to-last location (n_storage)
        start_idx = n_storage  # index in loc_mat
        start_dist = 0.0
        for i in range(n_items):
            loc_i = item_to_loc[i].item()
            n_picks = seq_mat[i, :].sum() + seq_mat[:, i].sum()  # rough proxy
            start_dist += loc_mat[start_idx, loc_i] if start_idx < n_locs and loc_i < n_locs else 0

        # 3. End distances: sum of (item_loc -> end) for each pick
        end_idx = n_storage + 1  # index in loc_mat
        end_dist = 0.0
        for i in range(n_items):
            loc_i = item_to_loc[i].item()
            end_dist += loc_mat[loc_i, end_idx] if loc_i < n_locs and end_idx < n_locs else 0

        # Combine components
        components = torch.tensor([
            item_item_dist,
            start_dist,
            end_dist,
        ]).unsqueeze(0)

        # Learnable refinement
        pred = self.refine(components)

        return pred.squeeze()


def train_simple():
    """Train a simpler linear model first to verify the approach."""
    print("=" * 60)
    print("Direct Distance Model Training")
    print("=" * 60)

    # Generate data
    print("\n[1/3] Generating data...")
    gen = DataGenerator()
    samples = gen.generate_samples(5, 5, 50, 1, 3, n_samples=500, distances_fixed=True, seed=42)

    simulator = Simulator()
    list_data = []
    for ob, il, w in tqdm(samples, desc="Building graphs"):
        g_data = build_graph_3d_sparse(
            order_book=ob, item_locations=il, warehouse=w, simulator=simulator.simulate
        )
        list_data.append(g_data)

    # Verify assignments vary
    a0 = list_data[0].edge_index[:, list_data[0].edge_type_mask[:, 2]][1].tolist()
    a1 = list_data[1].edge_index[:, list_data[1].edge_type_mask[:, 2]][1].tolist()
    print(f"  Assignments vary: {a0 != a1}")

    # Split
    train_data = list_data[:400]
    test_data = list_data[400:]

    # Normalize targets
    all_y = torch.cat([d.y for d in train_data])
    mean_y = all_y.mean().item()
    std_y = all_y.std().item()
    print(f"  Target: mean={mean_y:.1f}, std={std_y:.1f}")

    for d in list_data:
        d.y = (d.y - mean_y) / std_y

    # Don't normalize edge_attr - keep raw values for interpretability
    # The model will learn to handle the raw values

    # Simple test: compute distance directly and compare to target
    print("\n[2/3] Testing direct computation...")

    sample = test_data[0]
    n_items = sample.n_items
    n_storage = sample.n_storage
    n_locs = sample.n_locs

    # Extract matrices
    edge_index = sample.edge_index
    edge_attr = sample.edge_attr
    edge_type_mask = sample.edge_type_mask

    # Loc-Loc
    loc_mask = edge_type_mask[:, 0]
    loc_edges = edge_index[:, loc_mask]
    loc_vals = edge_attr[loc_mask, 0]

    print(f"  n_items={n_items}, n_storage={n_storage}, n_locs={n_locs}")
    print(f"  Loc-Loc edges: {loc_mask.sum().item()}")
    print(f"  Item-Item edges: {edge_type_mask[:, 1].sum().item()}")
    print(f"  Item-Loc edges: {edge_type_mask[:, 2].sum().item()}")

    # Target
    true_dist = sample.y.item() * std_y + mean_y
    print(f"  True distance: {true_dist:.1f}")

    print("\n[3/3] Training...")
    # For now, just verify data is correct
    print("  (Skipping training - need to fix model for batch processing)")

    # Save data for later use
    torch.save(test_data, "test_direct.pt")
    print("\n  Saved: test_direct.pt")


if __name__ == "__main__":
    train_simple()
