"""
Training with inverse optimization augmentation.

Trains GNN model, then uses single-run gradient-based inverse optimization
(analogous to CNN activation maximization) to generate one optimized sample
per training sample. This is simpler and faster than the previous approach
that ran 30 optimization candidates per sample.

Usage:
    cd notebooks && uv run train_augmented.py
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slotting_optimization.generator import DataGenerator
from slotting_optimization.gnn_builder import build_graph_3d_sparse
from slotting_optimization.simulator import Simulator
from slotting_optimization.inverse_optimizer import (
    optimize_assignment,
    assignment_to_graph_data,
    verify_with_simulator,
)


# ============================================================================
# Model Architecture (same as train_cpu.py)
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
        self.node_embedding = nn.Embedding(256, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [GCNBlock(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data_or_x, edge_index=None, edge_attr=None, batch=None):
        """Forward pass supporting both Data objects and explicit tensor inputs."""
        if isinstance(data_or_x, Data):
            # Called with Data object (batched training)
            data = data_or_x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            num_nodes = data.num_nodes
            batch = data.batch

            if hasattr(data, "batch") and data.batch is not None:
                n_items = (
                    data.n_items[0].item()
                    if isinstance(data.n_items, torch.Tensor)
                    else data.n_items
                )
                n_locs = (
                    data.n_locs[0].item()
                    if isinstance(data.n_locs, torch.Tensor)
                    else data.n_locs
                )
            else:
                n_items = data.n_items
                n_locs = data.n_locs

            nodes_per_graph = n_items + n_locs
            node_ids = (
                torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
                % nodes_per_graph
            )
            x = self.node_embedding(node_ids)
        else:
            # Called with explicit tensors (optimization)
            x = data_or_x

        edge_attr_enc = self.edge_encoder(edge_attr)

        for layer in self.layers:
            x, edge_attr_enc = layer(x, edge_index, edge_attr_enc)

        graph_emb = global_add_pool(x, batch)
        out = self.regressor(graph_emb)
        return out.squeeze(-1)


# ============================================================================
# Training Configuration
# ============================================================================

CONFIG = {
    # Problem size
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
    # Training - Phase 1 (initial)
    "epochs_phase1": 30,
    "batch_size": 32,
    "learning_rate": 0.001,
    "grad_clip": 1.0,
    # Training - Phase 2 (with augmentation)
    "epochs_phase2": 20,
    # Augmentation config
    "augmentation_n_steps": 50,  # Steps per optimization (like CNN activation maximization)
    "max_augmented_samples": 500,  # Stop generating after this many samples (None = no limit)
}


def train_epoch(model, train_loader, optimizer, criterion, device, grad_clip):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_samples = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_samples += batch.num_graphs

    return total_loss / n_samples


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    n_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = criterion(pred, batch.y.squeeze())
            total_loss += loss.item() * batch.num_graphs
            n_samples += batch.num_graphs

    return total_loss / n_samples


def augment_dataset(
    model,
    train_dataset,
    raw_train_samples,
    mean_y,
    std_y,
    edge_mean,
    edge_std,
    n_steps: int = 50,
    max_samples: int = None,
) -> list:
    """Generate one optimized sample per training sample.

    Uses single-run gradient-based optimization (like CNN activation maximization)
    to find an improved assignment for each training sample.

    Args:
        model: Trained GNN model
        train_dataset: List of training Data objects
        raw_train_samples: List of (OrderBook, ItemLocations, Warehouse) tuples
        mean_y, std_y: Target normalization params
        edge_mean, edge_std: Edge attribute normalization params
        n_steps: Number of optimization steps per sample
        max_samples: Stop generating after this many samples (None = no limit)

    Returns:
        List of augmented Data objects
    """
    model.eval()
    simulator = Simulator()
    augmented = []

    n_to_process = len(train_dataset)
    if max_samples is not None:
        n_to_process = min(n_to_process, max_samples)

    for idx, (data, raw_sample) in enumerate(
        tqdm(
            zip(train_dataset, raw_train_samples),
            desc="  Augmenting",
            total=n_to_process,
        )
    ):
        # Check if we've reached the limit
        if max_samples is not None and len(augmented) >= max_samples:
            break

        # Single optimization run (like CNN activation maximization)
        result = optimize_assignment(
            model=model,
            data=data,
            mean_y=mean_y,
            std_y=std_y,
            n_steps=n_steps,
        )

        # Convert to graph Data
        graph_data = assignment_to_graph_data(
            optimized_assignment=result["optimized_assignment"],
            raw_sample=raw_sample,
            simulator=simulator,
            edge_mean=edge_mean,
            edge_std=edge_std,
            mean_y=mean_y,
            std_y=std_y,
        )
        augmented.append(graph_data)

    print(f"  Generated {len(augmented)} augmented samples (1 per training sample)")
    return augmented


def main():
    print("=" * 60)
    print("GNN Training with Inverse Optimization Augmentation")
    print("=" * 60)
    print(f"Config: {CONFIG['n_items']} items, {CONFIG['n_locations']} locations")
    print(f"Samples: {CONFIG['n_samples']}")
    print(
        f"Training: Phase 1 ({CONFIG['epochs_phase1']} epochs) + Phase 2 ({CONFIG['epochs_phase2']} epochs)"
    )
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")

    # ========================================================================
    # Generate Data
    # ========================================================================
    print("\n[1/6] Generating training data...")

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

    # Split
    split_idx = int(len(list_data) * CONFIG["train_split"])
    train_dataset = list_data[:split_idx]
    test_dataset = list_data[split_idx:]
    raw_train_samples = samples[:split_idx]
    raw_test_samples = samples[split_idx:]
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # ========================================================================
    # Normalize
    # ========================================================================
    print("\n[2/6] Normalizing...")

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
    print("\n[3/6] Creating model...")

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

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"], shuffle=False
    )

    # ========================================================================
    # Phase 1: Initial Training
    # ========================================================================
    print("\n[4/6] Phase 1: Initial training...")

    best_val_loss = float("inf")

    for epoch in range(CONFIG["epochs_phase1"]):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, CONFIG["grad_clip"]
        )
        val_loss = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            marker = " *"
        else:
            marker = ""

        if epoch % 10 == 0 or epoch == CONFIG["epochs_phase1"] - 1:
            print(
                f"  Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}{marker}"
            )

    print(f"\n  Phase 1 best val_loss: {best_val_loss:.4f}")

    # ========================================================================
    # Phase 2: Augmentation
    # ========================================================================
    print("\n[5/6] Generating augmented samples...")

    augmented = augment_dataset(
        model=model,
        train_dataset=train_dataset,
        raw_train_samples=raw_train_samples,
        mean_y=mean_y,
        std_y=std_y,
        edge_mean=edge_mean,
        edge_std=edge_std,
        n_steps=CONFIG["augmentation_n_steps"],
        max_samples=CONFIG["max_augmented_samples"],
    )

    print(f"  Generated {len(augmented)} augmented samples")

    # ========================================================================
    # Phase 3: Continue Training with Augmented Data
    # ========================================================================
    print("\n[6/6] Phase 2: Training with augmented data...")

    # Combine original and augmented
    train_dataset_aug = train_dataset + augmented
    print(f"  Augmented training set: {len(train_dataset_aug)} samples")

    train_loader_aug = DataLoader(
        train_dataset_aug, batch_size=CONFIG["batch_size"], shuffle=True
    )

    # Reset optimizer for phase 2
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss_phase2 = float("inf")

    for epoch in range(CONFIG["epochs_phase2"]):
        train_loss = train_epoch(
            model, train_loader_aug, optimizer, criterion, device, CONFIG["grad_clip"]
        )
        val_loss = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_loss)

        actual_epoch = CONFIG["epochs_phase1"] + epoch

        if val_loss < best_val_loss_phase2:
            best_val_loss_phase2 = val_loss
            marker = " *"
        else:
            marker = ""

        if epoch % 10 == 0 or epoch == CONFIG["epochs_phase2"] - 1:
            print(
                f"  Epoch {actual_epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}{marker}"
            )

    print(f"\n  Phase 2 best val_loss: {best_val_loss_phase2:.4f}")

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
        "best_val_loss_phase1": best_val_loss,
        "best_val_loss_phase2": best_val_loss_phase2,
        "n_augmented_samples": len(augmented),
    }
    torch.save(checkpoint, "model_augmented.pt")
    print("  Saved: model_augmented.pt")

    torch.save(test_dataset, "test_dataset_augmented.pt")
    print("  Saved: test_dataset_augmented.pt")

    torch.save(raw_test_samples, "test_samples_augmented.pt")
    print("  Saved: test_samples_augmented.pt")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"  Original training samples: {len(train_dataset)}")
    print(f"  Augmented samples added: {len(augmented)}")
    print(f"  Total training samples: {len(train_dataset_aug)}")
    print(f"  Phase 1 best val_loss: {best_val_loss:.4f}")
    print(f"  Phase 2 best val_loss: {best_val_loss_phase2:.4f}")

    improvement = (best_val_loss - best_val_loss_phase2) / best_val_loss * 100
    if improvement > 0:
        print(f"  Validation loss improvement: {improvement:.1f}%")
    else:
        print(f"  Validation loss change: {improvement:.1f}%")

    # ========================================================================
    # Final Evaluation: Verify real improvement with simulator
    # ========================================================================
    print("\n" + "=" * 60)
    print("Final Evaluation: Inverse Optimization with Simulator Verification")
    print("=" * 60)

    # Build fresh unnormalized test graphs for evaluation
    # (The test_dataset was normalized in-place earlier, so we rebuild)
    print("  Rebuilding unnormalized test graphs for evaluation...")
    eval_test_graphs = []
    for ob, il, w in raw_test_samples:
        g_data = build_graph_3d_sparse(
            order_book=ob,
            item_locations=il,
            warehouse=w,
            simulator=simulator.simulate,
        )
        # Apply edge normalization (but NOT target normalization, that's handled by optimize_assignment)
        g_data.edge_attr = (g_data.edge_attr - edge_mean) / (edge_std + 1e-8)
        eval_test_graphs.append(g_data)

    n_eval_samples = min(50, len(eval_test_graphs))  # Evaluate on subset of test set
    print(f"  Evaluating {n_eval_samples} test samples...")

    improvements_gnn = []
    improvements_sim = []

    for idx in tqdm(range(n_eval_samples), desc="  Evaluating"):
        data = eval_test_graphs[idx]
        raw_sample = raw_test_samples[idx]

        # Run inverse optimization
        result = optimize_assignment(
            model=model,
            data=data,
            mean_y=mean_y,
            std_y=std_y,
            n_steps=CONFIG["augmentation_n_steps"],
        )

        # Verify with real simulator
        sim_result = verify_with_simulator(
            original_assignment=result["original_assignment"],
            optimized_assignment=result["optimized_assignment"],
            raw_sample=raw_sample,
        )

        improvements_gnn.append(result["improvement_pct"])
        improvements_sim.append(sim_result["sim_improvement_pct"])

    # Statistics
    improvements_gnn = np.array(improvements_gnn)
    improvements_sim = np.array(improvements_sim)

    print("\n  GNN Predicted Improvements:")
    print(f"    Mean:   {improvements_gnn.mean():.2f}%")
    print(f"    Std:    {improvements_gnn.std():.2f}%")
    print(f"    Min:    {improvements_gnn.min():.2f}%")
    print(f"    Max:    {improvements_gnn.max():.2f}%")

    print("\n  Real Simulator Improvements:")
    print(f"    Mean:   {improvements_sim.mean():.2f}%")
    print(f"    Std:    {improvements_sim.std():.2f}%")
    print(f"    Min:    {improvements_sim.min():.2f}%")
    print(f"    Max:    {improvements_sim.max():.2f}%")

    # How many actually improved?
    n_improved = (improvements_sim > 0).sum()
    print(f"\n  Samples with real improvement: {n_improved}/{n_eval_samples} ({100*n_improved/n_eval_samples:.1f}%)")

    # Correlation between GNN prediction and real improvement
    correlation = np.corrcoef(improvements_gnn, improvements_sim)[0, 1]
    print(f"  Correlation (GNN vs Simulator): {correlation:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
