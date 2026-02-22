"""
Training with validation framework AND augmentation to prove metaheuristic convergence.

This script combines:
- Validation framework from train_validation.py (known optimal via brute force)
- Augmentation from train_augmented.py (inverse optimization to generate better samples)

Usage:
    cd notebooks && uv run train_validation_with_augmentation.py
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
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slotting_optimization.validation import (
    generate_frequency_optimal_scenario,
    compute_brute_force_optimal,
    ValidationScenario,
    ConvergenceTracker,
)
from slotting_optimization.gnn_builder import build_graph_3d_sparse
from slotting_optimization.simulator import Simulator
from slotting_optimization.inverse_optimizer import (
    optimize_assignment,
    assignment_to_graph_data,
)
from slotting_optimization.item_locations import ItemLocations


# ============================================================================
# Model Architecture (same as train_augmented.py)
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
    def __init__(self, hidden_dim, edge_dim, num_layers, max_nodes=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_embedding = nn.Embedding(max_nodes, hidden_dim)
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
# Training Configuration - Scales with problem size
# ============================================================================


def get_config_for_problem_size(n_items: int) -> dict:
    """Generate configuration scaled for problem size.

    Scaling rationale:
    - Search space grows as n! (factorial), so larger problems need more exploration
    - Learning rate scales inversely to prevent overshooting in larger spaces
    - Epochs and steps scale with complexity ratio
    """
    base_config = {
        "n_locations": n_items,
        "n_items": n_items,
        "n_orders": max(100, n_items * 20),  # More orders for richer signal
        "n_samples": 500,
        "train_split": 0.8,
        "seed": 42,
        "hidden_dim": 32,
        "num_layers": 3,
        "batch_size": 32,
        "grad_clip": 1.0,
        "track_every_n_epochs": 5,
        "max_augmented_samples": 250,
        "n_augmentation_phases": 5,  # Number of augmentation cycles to run
    }

    # Complexity scales roughly with search space size (n!)
    # For relative scaling, use (n/5)^2 as proxy
    complexity_ratio = (n_items / 5.0) ** 2

    # Epochs scale with complexity
    base_config["epochs_phase1"] = max(30, int(30 * complexity_ratio))
    base_config["epochs_per_aug_phase"] = max(20, int(20 * complexity_ratio))

    # Learning rate scales inversely with size (larger problems need finer steps)
    base_config["learning_rate"] = 0.001 / (n_items / 5.0)

    # Optimization steps for augmentation scale with problem size
    base_config["augmentation_n_steps"] = max(50, n_items * 10)

    return base_config


# Default to 5x5 for debugging (fast brute force verification)
# To run longer training with multiple augmentation cycles, modify:
#   CONFIG["n_augmentation_phases"] = 3  # Run 3 augmentation cycles instead of 1
#   CONFIG["epochs_per_aug_phase"] = 30  # More epochs per phase
CONFIG = get_config_for_problem_size(5)


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


def track_convergence(
    model, scenario, test_data, mean_y, std_y, epoch, phase, tracker, n_steps=None
):
    """Track convergence at current epoch."""
    model.eval()

    # Run inverse optimization on a test sample
    result = optimize_assignment(
        model=model,
        data=test_data,
        mean_y=mean_y,
        std_y=std_y,
        n_steps=n_steps,
        verbose=False,
    )

    # Record convergence point
    point = tracker.record(
        epoch=epoch,
        phase=phase,
        optimized_assignment=result["optimized_assignment"],
    )

    return point


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
        train_dataset: List of training Data objects (unnormalized targets)
        raw_train_samples: List of (OrderBook, ItemLocations, Warehouse) tuples
        mean_y, std_y: Target normalization params
        edge_mean, edge_std: Edge attribute normalization params
        n_steps: Number of optimization steps per sample
        max_samples: Stop generating after this many samples (None = no limit)

    Returns:
        List of augmented Data objects (normalized)
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

    print(f"  Generated {len(augmented)} augmented samples")
    return augmented


def main():
    print("=" * 80)
    print("VALIDATION WITH AUGMENTATION: Training to Prove Metaheuristic Convergence")
    print("=" * 80)
    print(f"Problem: {CONFIG['n_items']} items, {CONFIG['n_locations']} locations")
    print(f"Training samples: {CONFIG['n_samples']} (different random starting assignments)")
    print(f"Augmentation: {CONFIG['n_augmentation_phases']} phase(s), up to {CONFIG['max_augmented_samples']} samples per phase")
    print(f"Tracking every {CONFIG['track_every_n_epochs']} epochs")
    total_epochs = CONFIG['epochs_phase1'] + CONFIG['n_augmentation_phases'] * CONFIG['epochs_per_aug_phase']
    print(f"Total training epochs: {total_epochs}")
    print()

    device = torch.device("cpu")

    # ========================================================================
    # Generate Validation Scenario
    # ========================================================================
    print("[1/8] Generating validation scenario with known optimal...")

    order_book, warehouse, random_initial, items, storage_locations, optimal_assignment, optimal_distance = (
        generate_frequency_optimal_scenario(
            n_items=CONFIG["n_items"],
            n_locations=CONFIG["n_locations"],
            n_orders=CONFIG["n_orders"],
            seed=CONFIG["seed"],
        )
    )

    print(f"  Optimal distance (analytical): {optimal_distance:.2f}")

    # Verify with brute force
    import math
    n_perms = math.factorial(CONFIG["n_items"])
    print(f"  Verifying with brute force enumeration ({CONFIG['n_items']}! = {n_perms:,} permutations)...")
    bf_optimal, bf_dist = compute_brute_force_optimal(
        order_book, warehouse, items, storage_locations
    )

    print(f"  Optimal distance (brute force): {bf_dist:.2f}")

    # Use brute force as ground truth (it's always optimal)
    if abs(bf_dist - optimal_distance) > 1e-6:
        print(f"  Note: Brute force found better solution than analytical heuristic")
        print(f"        Using brute force optimal as ground truth: {bf_dist:.2f}")
        optimal_assignment = bf_optimal
        optimal_distance = bf_dist
    else:
        print("  [OK] Analytical heuristic matches brute force optimal")

    # Create ValidationScenario
    scenario = ValidationScenario(
        name=f"{CONFIG['n_items']}x{CONFIG['n_locations']}_frequency_optimal",
        order_book=order_book,
        warehouse=warehouse,
        initial_assignment=random_initial,
        optimal_assignment=optimal_assignment,
        optimal_distance=optimal_distance,
        items=items,
        storage_locations=storage_locations,
    )

    # ========================================================================
    # Generate Training Samples (Random Starting Assignments)
    # ========================================================================
    print(f"\n[2/8] Generating {CONFIG['n_samples']} training samples...")
    print("  (Same orders/warehouse, different random starting assignments)")

    rng = random.Random(CONFIG["seed"])
    samples = []

    for i in range(CONFIG["n_samples"]):
        # Shuffle locations for random assignment
        shuffled_locs = storage_locations[:]
        rng.shuffle(shuffled_locs)

        random_assignment = ItemLocations.from_records([
            {"item_id": item, "location_id": loc}
            for item, loc in zip(items, shuffled_locs)
        ])

        samples.append((order_book, random_assignment, warehouse))

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
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # ========================================================================
    # Normalize
    # ========================================================================
    print("\n[3/8] Normalizing...")

    all_y = torch.cat([data.y for data in train_dataset])
    mean_y = all_y.mean().item()
    std_y = all_y.std().item()
    print(f"  Target: mean={mean_y:.1f}, std={std_y:.1f}")

    for data in train_dataset + test_dataset:
        data.y = (data.y - mean_y) / std_y

    all_edge_attrs = torch.cat([data.edge_attr for data in train_dataset], dim=0)
    edge_mean = all_edge_attrs.mean(dim=0)
    edge_std = all_edge_attrs.std(dim=0)

    for data in train_dataset + test_dataset:
        data.edge_attr = (data.edge_attr - edge_mean) / (edge_std + 1e-8)

    # ========================================================================
    # Create Model
    # ========================================================================
    print("\n[4/8] Creating model...")

    model = GraphRegressionModel(
        hidden_dim=CONFIG["hidden_dim"],
        edge_dim=3,
        num_layers=CONFIG["num_layers"],
        max_nodes=max(256, (CONFIG["n_items"] + CONFIG["n_locations"]) * 2),
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

    # Initialize convergence tracker
    tracker = ConvergenceTracker(scenario)

    # ========================================================================
    # Phase 1: Initial Training with Convergence Tracking
    # ========================================================================
    print("\n[5/8] Phase 1: Initial training...")

    best_val_loss = float("inf")

    # Track initial convergence (epoch 0)
    print("  Tracking convergence at epoch 0...")
    point = track_convergence(
        model, scenario, test_dataset[0], mean_y, std_y, 0, "phase1", tracker,
        n_steps=CONFIG["augmentation_n_steps"]
    )
    print(f"    Gap: {point.optimality_gap_pct:.2f}%, Accuracy: {point.assignment_accuracy_pct:.1f}%")
    print(f"    Using {CONFIG['augmentation_n_steps']} optimization steps, LR scaled for {CONFIG['n_items']}x{CONFIG['n_items']}")

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

        # Track convergence
        if (epoch + 1) % CONFIG["track_every_n_epochs"] == 0 or epoch == CONFIG["epochs_phase1"] - 1:
            point = track_convergence(
                model, scenario, test_dataset[0], mean_y, std_y, epoch + 1, "phase1", tracker,
                n_steps=CONFIG["augmentation_n_steps"]
            )
            print(
                f"  Epoch {epoch + 1:3d}: loss={val_loss:.4f}{marker}, "
                f"gap={point.optimality_gap_pct:6.2f}%, acc={point.assignment_accuracy_pct:5.1f}%"
            )

    print(f"\n  Phase 1 complete. Best val_loss: {best_val_loss:.4f}")

    # ========================================================================
    # Phases 2+: Iterative Augmentation Cycles
    # ========================================================================
    # Each cycle: generate augmented samples → train on combined dataset
    # This allows the model to iteratively improve via self-generated data

    current_train_dataset = train_dataset
    cumulative_epoch = CONFIG["epochs_phase1"]
    phase_best_losses = []

    for aug_phase_num in range(1, CONFIG["n_augmentation_phases"] + 1):
        phase_name = f"aug{aug_phase_num}"
        print(f"\n[{5 + aug_phase_num*2}/...] Augmentation Phase {aug_phase_num}: Generating samples...")

        # Need unnormalized data for augmentation (assignment_to_graph_data handles normalization)
        # The train_dataset was normalized in-place, so we rebuild for augmentation
        print("  Rebuilding unnormalized training graphs for augmentation...")
        aug_train_graphs = []
        for ob, il, w in tqdm(raw_train_samples[:CONFIG["max_augmented_samples"]], desc="  Rebuilding"):
            g_data = build_graph_3d_sparse(
                order_book=ob,
                item_locations=il,
                warehouse=w,
                simulator=simulator.simulate,
            )
            # Apply edge normalization (but NOT target normalization for optimization)
            g_data.edge_attr = (g_data.edge_attr - edge_mean) / (edge_std + 1e-8)
            aug_train_graphs.append(g_data)

        augmented = augment_dataset(
            model=model,
            train_dataset=aug_train_graphs,
            raw_train_samples=raw_train_samples[:CONFIG["max_augmented_samples"]],
            mean_y=mean_y,
            std_y=std_y,
            edge_mean=edge_mean,
            edge_std=edge_std,
            n_steps=CONFIG["augmentation_n_steps"],
            max_samples=CONFIG["max_augmented_samples"],
        )

        print(f"  Generated {len(augmented)} augmented samples")

        # ========================================================================
        # Training with Augmented Data
        # ========================================================================
        print(f"\n[{6 + aug_phase_num*2}/...] Augmentation Phase {aug_phase_num}: Training...")

        # Combine original and augmented
        current_train_dataset = current_train_dataset + augmented
        print(f"  Training set size: {len(current_train_dataset)} samples (original + phase {aug_phase_num} augmentation)")

        train_loader_aug = DataLoader(
            current_train_dataset, batch_size=CONFIG["batch_size"], shuffle=True
        )

        # Reset optimizer for this augmentation phase
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_val_loss_aug = float("inf")

        for epoch in range(CONFIG["epochs_per_aug_phase"]):
            train_loss = train_epoch(
                model, train_loader_aug, optimizer, criterion, device, CONFIG["grad_clip"]
            )
            val_loss = evaluate(model, test_loader, criterion, device)
            scheduler.step(val_loss)

            cumulative_epoch += 1

            if val_loss < best_val_loss_aug:
                best_val_loss_aug = val_loss
                marker = " *"
            else:
                marker = ""

            # Track convergence
            if (epoch + 1) % CONFIG["track_every_n_epochs"] == 0 or epoch == CONFIG["epochs_per_aug_phase"] - 1:
                point = track_convergence(
                    model, scenario, test_dataset[0], mean_y, std_y, cumulative_epoch, phase_name, tracker,
                    n_steps=CONFIG["augmentation_n_steps"]
                )
                print(
                    f"  Epoch {cumulative_epoch:3d}: loss={val_loss:.4f}{marker}, "
                    f"gap={point.optimality_gap_pct:6.2f}%, acc={point.assignment_accuracy_pct:5.1f}%"
                )

        phase_best_losses.append(best_val_loss_aug)
        print(f"\n  Phase {aug_phase_num} complete. Best val_loss: {best_val_loss_aug:.4f}")

    # ========================================================================
    # Final Report
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONVERGENCE REPORT")
    print("=" * 80)

    history = tracker.get_history()

    print(f"Optimal distance: {scenario.optimal_distance:.2f}")
    print()
    print(f"Initial state (epoch 0):")
    print(f"  Gap: {history[0].optimality_gap_pct:.2f}%")
    print(f"  Accuracy: {history[0].assignment_accuracy_pct:.1f}%")
    print()
    print(f"After Phase 1 (epoch {CONFIG['epochs_phase1']}):")
    # Find phase 1 end point
    phase1_end = [p for p in history if p.phase == "phase1"][-1]
    print(f"  Gap: {phase1_end.optimality_gap_pct:.2f}%")
    print(f"  Accuracy: {phase1_end.assignment_accuracy_pct:.1f}%")
    print()

    # Show progress for each augmentation phase
    if CONFIG["n_augmentation_phases"] > 0:
        print(f"After each augmentation phase:")
        for aug_num in range(1, CONFIG["n_augmentation_phases"] + 1):
            phase_name = f"aug{aug_num}"
            phase_points = [p for p in history if p.phase == phase_name]
            if phase_points:
                phase_end = phase_points[-1]
                print(f"  Phase {aug_num} (epoch {phase_end.epoch}): Gap: {phase_end.optimality_gap_pct:.2f}%, Accuracy: {phase_end.assignment_accuracy_pct:.1f}%")
        print()

    print(f"Final state (epoch {history[-1].epoch}):")
    print(f"  Gap: {history[-1].optimality_gap_pct:.2f}%")
    print(f"  Accuracy: {history[-1].assignment_accuracy_pct:.1f}%")
    print()
    print(f"Overall Improvement:")
    print(f"  Gap reduction: {history[0].optimality_gap_pct - history[-1].optimality_gap_pct:.2f} percentage points")
    print(f"  Accuracy gain: {history[-1].assignment_accuracy_pct - history[0].assignment_accuracy_pct:.1f} percentage points")
    print()
    print(f"Augmentation Impact:")
    print(f"  Phase 1 final gap: {phase1_end.optimality_gap_pct:.2f}%")
    print(f"  Final gap after {CONFIG['n_augmentation_phases']} augmentation phase(s): {history[-1].optimality_gap_pct:.2f}%")
    augmentation_improvement = phase1_end.optimality_gap_pct - history[-1].optimality_gap_pct
    print(f"  Total gap reduction from augmentation: {augmentation_improvement:.2f} percentage points")
    print()

    # Show convergence trajectory
    print("Convergence trajectory:")
    print(f"{'Epoch':>6} {'Phase':>8} {'Gap (%)':>10} {'Accuracy (%)':>14}")
    print("-" * 42)
    for point in history:
        print(
            f"{point.epoch:6d} {point.phase:>8} "
            f"{point.optimality_gap_pct:10.2f} {point.assignment_accuracy_pct:14.1f}"
        )

    print("\n" + "=" * 80)
    print(f"[OK] Validation with {CONFIG['n_augmentation_phases']} augmentation phase(s) complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
