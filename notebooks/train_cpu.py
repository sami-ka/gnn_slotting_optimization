"""
CPU-only GNN training script for warehouse slotting optimization.

Smaller problem size for fast iteration on CPU.

Usage:
    cd notebooks && uv run train_cpu.py
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slotting_optimization.generator import DataGenerator
from slotting_optimization.gnn_builder import build_graph_3d_sparse
from slotting_optimization.simulator import Simulator


# ============================================================================
# Model Architecture
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

    def forward(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        edge_type_mask = data.edge_type_mask
        num_nodes = data.num_nodes

        # Handle batched data - use first sample's metadata (all same for this problem)
        if hasattr(data, 'batch') and data.batch is not None:
            # Batched - get from first sample's values
            n_items = data.n_items[0].item() if isinstance(data.n_items, torch.Tensor) else data.n_items
            n_storage = data.n_storage[0].item() if isinstance(data.n_storage, torch.Tensor) else data.n_storage
            n_locs = data.n_locs[0].item() if isinstance(data.n_locs, torch.Tensor) else data.n_locs
        else:
            n_items = data.n_items
            n_storage = data.n_storage
            n_locs = data.n_locs

        # Initialize node features by local node index within each graph
        nodes_per_graph = n_items + n_locs
        node_ids = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device) % nodes_per_graph
        x = self.node_embedding(node_ids)

        # Encode edges
        edge_attr_enc = self.edge_encoder(edge_attr)

        # Message passing
        for layer in self.layers:
            x, edge_attr_enc = layer(x, edge_index, edge_attr_enc)

        graph_emb = global_add_pool(x, data.batch)
        out = self.regressor(graph_emb)
        return out.squeeze(-1)


# ============================================================================
# Training Configuration (smaller for CPU)
# ============================================================================

CONFIG = {
    # Problem size (tiny for fast iteration)
    "n_locations": 5,
    "n_items": 5,
    "n_orders": 50,
    "min_items_per_order": 1,
    "max_items_per_order": 3,

    # Dataset
    "n_samples": 1000,
    "train_split": 0.8,
    "seed": 42,

    # Model
    "hidden_dim": 32,
    "num_layers": 3,

    # Training
    "epochs": 30,
    "batch_size": 32,
    "learning_rate": 0.001,
    "grad_clip": 1.0,
}


def main():
    print("=" * 60)
    print("GNN Training for Warehouse Slotting (CPU)")
    print("=" * 60)
    print(f"Config: {CONFIG['n_items']} items, {CONFIG['n_locations']} locations")
    print(f"Samples: {CONFIG['n_samples']}, Epochs: {CONFIG['epochs']}")
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")

    # ========================================================================
    # Generate Data
    # ========================================================================
    print("\n[1/4] Generating training data...")

    gen = DataGenerator()
    samples = gen.generate_samples(
        n_locations=CONFIG["n_locations"],
        nb_items=CONFIG["n_items"],
        n_orders=CONFIG["n_orders"],
        min_items_per_order=CONFIG["min_items_per_order"],
        max_items_per_order=CONFIG["max_items_per_order"],
        n_samples=CONFIG["n_samples"],
        distances_fixed=True,
        seed=CONFIG["seed"],
    )

    print(f"  Generated {len(samples)} samples")

    # Build graphs
    print("  Building graphs...")
    simulator = Simulator()
    list_data = []
    for ob, il, w in tqdm(samples, desc="  Building graphs"):
        g_data = build_graph_3d_sparse(
            order_book=ob,
            item_locations=il,
            warehouse=w,
            simulator=simulator.simulate,
        )
        list_data.append(g_data)

    # Verify setup: assignments vary, but orders are same
    assigns_0 = list_data[0].edge_index[:, list_data[0].edge_type_mask[:, 2]][1].tolist()
    assigns_1 = list_data[1].edge_index[:, list_data[1].edge_type_mask[:, 2]][1].tolist()

    # Check Item-Item edges (sequence patterns) - should be SAME
    seq_edges_0 = list_data[0].edge_attr[list_data[0].edge_type_mask[:, 1], 1].sum().item()
    seq_edges_1 = list_data[1].edge_attr[list_data[1].edge_type_mask[:, 1], 1].sum().item()

    print(f"  Assignments vary: {assigns_0 != assigns_1}")
    print(f"  Orders same (seq_sum match): {abs(seq_edges_0 - seq_edges_1) < 0.01}")

    if assigns_0 == assigns_1:
        print("  WARNING: Assignments are identical!")
    if abs(seq_edges_0 - seq_edges_1) > 0.01:
        print("  WARNING: Orders differ between samples!")

    # Split
    split_idx = int(len(list_data) * CONFIG["train_split"])
    train_dataset = list_data[:split_idx]
    test_dataset = list_data[split_idx:]
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # ========================================================================
    # Normalize
    # ========================================================================
    print("\n[2/4] Normalizing...")

    # Target normalization
    all_y = torch.cat([data.y for data in train_dataset])
    mean_y = all_y.mean().item()
    std_y = all_y.std().item()
    print(f"  Target: mean={mean_y:.1f}, std={std_y:.1f}")

    for data in train_dataset + test_dataset:
        data.y = (data.y - mean_y) / std_y

    # Edge attribute normalization
    all_edge_attrs = torch.cat([data.edge_attr for data in train_dataset], dim=0)
    edge_mean = all_edge_attrs.mean(dim=0)
    edge_std = all_edge_attrs.std(dim=0)
    print(f"  Edge attr mean: {edge_mean.tolist()}")
    print(f"  Edge attr std: {edge_std.tolist()}")

    for data in train_dataset + test_dataset:
        data.edge_attr = (data.edge_attr - edge_mean) / (edge_std + 1e-8)

    # ========================================================================
    # Create Model
    # ========================================================================
    print("\n[3/4] Creating model...")

    model = GraphRegressionModel(
        hidden_dim=CONFIG["hidden_dim"],
        edge_dim=3,
        num_layers=CONFIG["num_layers"],
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # ========================================================================
    # Train
    # ========================================================================
    print("\n[4/4] Training...")

    best_val_loss = float("inf")

    for epoch in range(CONFIG["epochs"]):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs

        train_loss /= len(train_dataset)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch)
                loss = criterion(pred, batch.y.squeeze())
                val_loss += loss.item() * batch.num_graphs

        val_loss /= len(test_dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            marker = " *"
        else:
            marker = ""

        if epoch % 10 == 0 or epoch == CONFIG["epochs"] - 1:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}{marker}")

    # ========================================================================
    # Save
    # ========================================================================
    print("\n" + "=" * 60)
    print("Saving model and test data...")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "mean_y": mean_y,
        "std_y": std_y,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "config": CONFIG,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, "model_cpu.pt")
    print("  Saved: model_cpu.pt")

    torch.save(test_dataset, "test_dataset_cpu.pt")
    print("  Saved: test_dataset_cpu.pt")

    # Save raw samples for simulator verification
    test_samples = samples[split_idx:]  # Same indices as test_dataset
    torch.save(test_samples, "test_samples_cpu.pt")
    print("  Saved: test_samples_cpu.pt")

    # ========================================================================
    # Test Assignment Sensitivity
    # ========================================================================
    print("\n" + "=" * 60)
    print("Testing assignment sensitivity...")

    model.eval()
    sample = test_dataset[0]

    # Original prediction
    torch.manual_seed(42)
    with torch.no_grad():
        orig_pred = model(sample).item()

    # Swap two items' locations
    swapped_edge_index = sample.edge_index.clone()
    item_loc_mask = sample.edge_type_mask[:, 2]
    item_loc_indices = torch.where(item_loc_mask)[0]

    if len(item_loc_indices) >= 2:
        idx0, idx1 = item_loc_indices[0], item_loc_indices[1]
        # Swap targets
        swapped_edge_index[1, idx0], swapped_edge_index[1, idx1] = (
            sample.edge_index[1, idx1].clone(),
            sample.edge_index[1, idx0].clone(),
        )

        from torch_geometric.data import Data
        swapped_sample = Data(
            edge_index=swapped_edge_index,
            edge_attr=sample.edge_attr,
            edge_type_mask=sample.edge_type_mask,
            num_nodes=sample.num_nodes,
            n_items=sample.n_items,
            n_locs=sample.n_locs,
            n_storage=sample.n_storage,
            y=sample.y,
        )

        with torch.no_grad():
            swap_pred = model(swapped_sample).item()

        diff = abs(orig_pred - swap_pred)
        print(f"  Original prediction: {orig_pred:.4f}")
        print(f"  After swap:          {swap_pred:.4f}")
        print(f"  Difference:          {diff:.4f}")

        if diff > 0.01:
            print("  SUCCESS: Model is sensitive to assignment changes!")
        else:
            print("  WARNING: Model shows little sensitivity to assignments")

    print("\nDone!")


if __name__ == "__main__":
    main()
