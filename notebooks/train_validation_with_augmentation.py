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
)
from slotting_optimization.gnn_builder import build_graph_3d_sparse
from slotting_optimization.simulator import Simulator
from slotting_optimization.inverse_optimizer import (
    optimize_assignment,
    assignment_to_graph_data,
    extract_current_assignment,
)
from slotting_optimization.item_locations import ItemLocations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


# ============================================================================
# Candidate Pool Evaluation
# ============================================================================


@dataclass
class CandidateResult:
    """Result of evaluating a single candidate solution."""
    source: str  # "training" or "inverse_opt"
    assignment: np.ndarray
    simulated_distance: float
    gap_to_optimal_pct: float


@dataclass
class PhaseEvaluation:
    """Results of evaluating the full candidate pool at a phase boundary."""
    phase: str
    epoch: int

    # Best overall
    best_distance: float
    best_gap_pct: float
    best_source: str
    best_assignment: np.ndarray

    # Breakdown by source
    best_training_distance: float
    best_training_gap_pct: float
    best_inverse_opt_distance: float
    best_inverse_opt_gap_pct: float

    # Stats
    n_training_candidates: int
    n_inverse_opt_candidates: int


def get_gnn_predictions(model, dataset: list, mean_y: float, std_y: float) -> List[float]:
    """Get GNN predictions for all samples in dataset.

    Returns list of predicted distances (denormalized).
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for data in dataset:
            pred_norm = model(data)
            pred = pred_norm.item() * std_y + mean_y
            predictions.append(pred)

    return predictions


def simulate_assignment(
    assignment: np.ndarray,
    raw_sample: tuple,  # (OrderBook, ItemLocations, Warehouse)
    simulator: Simulator,
) -> float:
    """Simulate a single assignment and return the distance."""
    order_book, original_il, warehouse = raw_sample

    # Get item IDs (sorted to match graph node order)
    items = sorted(original_il.to_dict().keys())

    # Get storage locations
    storage_locs = sorted([
        loc for loc in warehouse.locations()
        if loc not in (warehouse.start_point, warehouse.end_point)
    ])

    # Build ItemLocations with the assignment
    records = []
    for i, item in enumerate(items):
        loc_idx = assignment[i]
        records.append({"item_id": item, "location_id": storage_locs[loc_idx]})

    il = ItemLocations.from_records(records)
    distance, _ = simulator.simulate(order_book, warehouse, il)
    return distance


def evaluate_candidate_pool(
    model: nn.Module,
    accumulated_dataset: list,  # All training samples (original + augmented)
    raw_samples: list,  # Raw samples for simulation (first N correspond to original training)
    scenario: "ValidationScenario",
    phase: str = "phase1",
    epoch: int = 0,
) -> PhaseEvaluation:
    """Evaluate accumulated training samples at a phase boundary.

    Simulates all assignments in the accumulated dataset to find the best one.

    Args:
        model: Trained GNN model (unused, kept for API compatibility)
        accumulated_dataset: All training Data objects accumulated so far
        raw_samples: Raw (OrderBook, ItemLocations, Warehouse) tuples for first N samples
        scenario: ValidationScenario with optimal solution
        phase: Phase name for reporting
        epoch: Current epoch for reporting

    Returns:
        PhaseEvaluation with results
    """
    simulator = Simulator()
    optimal_distance = scenario.optimal_distance

    results = []

    print(f"    Evaluating {len(accumulated_dataset)} candidates...")

    for idx, data in enumerate(accumulated_dataset):
        # Extract assignment from graph
        assignment = extract_current_assignment(data)

        # Simulate - use first raw_sample since all share the same order_book/warehouse
        raw_idx = min(idx, len(raw_samples) - 1)
        distance = simulate_assignment(assignment, raw_samples[raw_idx], simulator)

        gap_pct = (distance - optimal_distance) / optimal_distance * 100

        results.append(CandidateResult(
            source="training",
            assignment=assignment,
            simulated_distance=distance,
            gap_to_optimal_pct=gap_pct,
        ))

    best = min(results, key=lambda r: r.simulated_distance)

    return PhaseEvaluation(
        phase=phase,
        epoch=epoch,
        best_distance=best.simulated_distance,
        best_gap_pct=best.gap_to_optimal_pct,
        best_source=best.source,
        best_assignment=best.assignment,
        best_training_distance=best.simulated_distance,
        best_training_gap_pct=best.gap_to_optimal_pct,
        best_inverse_opt_distance=float('inf'),
        best_inverse_opt_gap_pct=float('inf'),
        n_training_candidates=len(results),
        n_inverse_opt_candidates=0,
    )


def print_phase_evaluation(eval_result: PhaseEvaluation, optimal_distance: float):
    """Print phase evaluation results."""
    print(f"\n    {'='*60}")
    print(f"    PHASE EVALUATION: {eval_result.phase} (epoch {eval_result.epoch})")
    print(f"    {'='*60}")
    print(f"    Optimal distance: {optimal_distance:.2f}")
    print(f"    Candidates evaluated: {eval_result.n_training_candidates}")
    print(f"    Best distance: {eval_result.best_distance:.2f} (gap: {eval_result.best_gap_pct:+.2f}%)")
    print(f"    {'='*60}")


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
        "n_samples": 200,
        "train_split": 0.8,
        "seed": 42,
        "hidden_dim": 32,
        "num_layers": 3,
        "batch_size": 32,
        "grad_clip": 1.0,
        "track_every_n_epochs": 5,
        "n_augmentation_phases": 20,  # Number of augmentation cycles to run
        "n_top_for_inverse_opt": 15,  # Top N by GNN prediction to inverse optimize
        "perturbation_scale": 1.0,  # Random noise added to logit init for diversity
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
CONFIG = get_config_for_problem_size(10)


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
    top_graphs,
    top_raw_samples,
    mean_y,
    std_y,
    edge_mean,
    edge_std,
    n_steps: int = 50,
    existing_assignments: set = None,
    perturbation_scale: float = 0.0,
) -> tuple:
    """Generate augmented samples by inverse optimizing a pre-selected set of graphs.

    Uses single-run gradient-based optimization (like CNN activation maximization)
    to find an improved assignment for each selected sample.

    Args:
        model: Trained GNN model
        top_graphs: Pre-selected list of top-N Data objects (already normalized)
        top_raw_samples: Corresponding list of (OrderBook, ItemLocations, Warehouse) tuples
        mean_y, std_y: Target normalization params
        edge_mean, edge_std: Edge attribute normalization params
        n_steps: Number of optimization steps per sample
        existing_assignments: Set of assignment tuples already in the dataset (for deduplication)
        perturbation_scale: Random noise scale for logit initialization diversity

    Returns:
        Tuple of (augmented Data objects, updated existing_assignments set,
                  list of (raw_sample, graph_data) pairs for accepted augmented samples)
    """
    model.eval()
    simulator = Simulator()
    augmented = []
    augmented_raw_and_graph = []  # (raw_sample, graph_data) for accepted samples

    if existing_assignments is None:
        existing_assignments = set()

    n_duplicates = 0
    for data, raw_sample in tqdm(zip(top_graphs, top_raw_samples), total=len(top_graphs), desc="  Augmenting"):
        # Single optimization run with optional perturbation for diversity
        result = optimize_assignment(
            model=model,
            data=data,
            mean_y=mean_y,
            std_y=std_y,
            n_steps=n_steps,
            perturbation_scale=perturbation_scale,
        )

        # Check for duplicate assignment
        assignment_tuple = tuple(result["optimized_assignment"])
        if assignment_tuple in existing_assignments:
            n_duplicates += 1
            continue

        existing_assignments.add(assignment_tuple)

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
        augmented_raw_and_graph.append((raw_sample, graph_data))

    print(f"  Generated {len(augmented)} augmented samples ({n_duplicates} duplicates skipped)")
    return augmented, existing_assignments, augmented_raw_and_graph


def main():
    print("=" * 80)
    print("VALIDATION WITH AUGMENTATION: Training to Prove Metaheuristic Convergence")
    print("=" * 80)
    print(f"Problem: {CONFIG['n_items']} items, {CONFIG['n_locations']} locations")
    print(f"Training samples: {CONFIG['n_samples']} (different random starting assignments)")
    print(f"Augmentation: {CONFIG['n_augmentation_phases']} phase(s), top {CONFIG['n_top_for_inverse_opt']} by GNN prediction per phase")
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

    # ========================================================================
    # Track phase evaluations (candidate pool evaluation at each phase boundary)
    # ========================================================================
    phase_evaluations: List[PhaseEvaluation] = []
    best_solution_ever: Optional[PhaseEvaluation] = None

    # ========================================================================
    # Phase 1: Initial Training
    # ========================================================================
    print("\n[5/8] Phase 1: Initial training...")

    best_val_loss = float("inf")

    # Initial evaluation before training (epoch 0)
    print("  Evaluating candidate pool at epoch 0 (before training)...")
    eval_result = evaluate_candidate_pool(
        model=model,
        accumulated_dataset=train_dataset,
        raw_samples=raw_train_samples,
        scenario=scenario,
        phase="phase1_start",
        epoch=0,
    )
    phase_evaluations.append(eval_result)
    best_solution_ever = eval_result
    print_phase_evaluation(eval_result, scenario.optimal_distance)

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

        # Print progress every N epochs
        if (epoch + 1) % CONFIG["track_every_n_epochs"] == 0 or epoch == CONFIG["epochs_phase1"] - 1:
            print(f"  Epoch {epoch + 1:3d}: loss={val_loss:.4f}{marker}")

    # Evaluate candidate pool at end of Phase 1
    print(f"\n  Phase 1 complete. Best val_loss: {best_val_loss:.4f}")
    print("  Evaluating candidate pool at end of Phase 1...")
    eval_result = evaluate_candidate_pool(
        model=model,
        accumulated_dataset=train_dataset,
        raw_samples=raw_train_samples,
        scenario=scenario,
        phase="phase1_end",
        epoch=CONFIG["epochs_phase1"],
    )
    phase_evaluations.append(eval_result)
    if eval_result.best_gap_pct < best_solution_ever.best_gap_pct:
        best_solution_ever = eval_result
    print_phase_evaluation(eval_result, scenario.optimal_distance)

    # ========================================================================
    # Phases 2+: Iterative Augmentation Cycles
    # ========================================================================
    # Each cycle: generate augmented samples → train on combined dataset
    # This allows the model to iteratively improve via self-generated data

    current_train_dataset = list(train_dataset)  # Copy to avoid modifying original
    accumulated_raw_samples = list(raw_train_samples)  # Grows with augmented samples
    cumulative_epoch = CONFIG["epochs_phase1"]
    phase_best_losses = []

    # Initialize existing_assignments set with assignments from original training dataset
    existing_assignments = set()
    for data in train_dataset:
        assignment = extract_current_assignment(data)
        existing_assignments.add(tuple(assignment))
    print(f"  Initialized with {len(existing_assignments)} unique assignments from training set")

    for aug_phase_num in range(1, CONFIG["n_augmentation_phases"] + 1):
        phase_name = f"aug{aug_phase_num}"
        print(f"\n[{5 + aug_phase_num*2}/...] Augmentation Phase {aug_phase_num}: Generating samples...")

        # Select top-N from the full accumulated dataset (includes previously augmented samples)
        predictions = get_gnn_predictions(model, current_train_dataset, mean_y, std_y)
        sorted_indices = np.argsort(predictions)
        top_indices = sorted_indices[:CONFIG["n_top_for_inverse_opt"]]

        top_graphs = [current_train_dataset[i] for i in top_indices]
        top_raw = [accumulated_raw_samples[i] for i in top_indices]

        print(f"  Selected top {len(top_indices)} samples from {len(current_train_dataset)} accumulated samples")

        augmented, existing_assignments, augmented_raw_and_graph = augment_dataset(
            model=model,
            top_graphs=top_graphs,
            top_raw_samples=top_raw,
            mean_y=mean_y,
            std_y=std_y,
            edge_mean=edge_mean,
            edge_std=edge_std,
            n_steps=CONFIG["augmentation_n_steps"],
            existing_assignments=existing_assignments,
            perturbation_scale=CONFIG["perturbation_scale"],
        )

        # ========================================================================
        # Training with Augmented Data
        # ========================================================================
        print(f"\n[{6 + aug_phase_num*2}/...] Augmentation Phase {aug_phase_num}: Training...")

        # Combine original and augmented; track raw samples for future phases
        current_train_dataset = current_train_dataset + augmented
        for raw_s, _ in augmented_raw_and_graph:
            accumulated_raw_samples.append(raw_s)
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

            # Print progress every N epochs
            if (epoch + 1) % CONFIG["track_every_n_epochs"] == 0 or epoch == CONFIG["epochs_per_aug_phase"] - 1:
                print(f"  Epoch {cumulative_epoch:3d}: loss={val_loss:.4f}{marker}")

        phase_best_losses.append(best_val_loss_aug)
        print(f"\n  Phase {aug_phase_num} complete. Best val_loss: {best_val_loss_aug:.4f}")

        # Evaluate candidate pool at end of this augmentation phase
        print(f"  Evaluating candidate pool at end of augmentation phase {aug_phase_num}...")
        eval_result = evaluate_candidate_pool(
            model=model,
            accumulated_dataset=current_train_dataset,
            raw_samples=accumulated_raw_samples,
            scenario=scenario,
            phase=f"aug{aug_phase_num}_end",
            epoch=cumulative_epoch,
        )
        phase_evaluations.append(eval_result)
        if eval_result.best_gap_pct < best_solution_ever.best_gap_pct:
            best_solution_ever = eval_result
        print_phase_evaluation(eval_result, scenario.optimal_distance)

    # ========================================================================
    # Final Report
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONVERGENCE REPORT - CANDIDATE POOL EVALUATION")
    print("=" * 80)

    print(f"\nOptimal distance (brute force verified): {scenario.optimal_distance:.2f}")
    print()

    # Summary of best solution found
    print("=" * 60)
    print("BEST SOLUTION FOUND")
    print("=" * 60)
    print(f"  Distance: {best_solution_ever.best_distance:.2f}")
    print(f"  Gap to optimal: {best_solution_ever.best_gap_pct:+.2f}%")
    print(f"  Found at: {best_solution_ever.phase} (epoch {best_solution_ever.epoch})")
    print()

    # Trajectory table
    print("=" * 60)
    print("CONVERGENCE TRAJECTORY")
    print("=" * 60)
    print(f"{'Phase':<15} {'Epoch':>6} {'Candidates':>12} {'Best Gap':>12}")
    print("-" * 50)
    for ev in phase_evaluations:
        print(
            f"{ev.phase:<15} {ev.epoch:>6} {ev.n_training_candidates:>12} "
            f"{ev.best_gap_pct:>+11.2f}%"
        )
    print()

    # Improvement analysis
    initial_eval = phase_evaluations[0]
    final_eval = phase_evaluations[-1]

    print("=" * 60)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 60)
    print(f"Initial best gap (before training): {initial_eval.best_gap_pct:+.2f}%")
    print(f"Final best gap (after all phases): {final_eval.best_gap_pct:+.2f}%")
    print(f"Total gap reduction: {initial_eval.best_gap_pct - final_eval.best_gap_pct:.2f} percentage points")
    print()

    # Phase 1 vs augmentation impact
    phase1_end_eval = [ev for ev in phase_evaluations if ev.phase == "phase1_end"][0]
    print(f"After Phase 1: {phase1_end_eval.best_gap_pct:+.2f}%")

    if CONFIG["n_augmentation_phases"] > 0:
        aug_reduction = phase1_end_eval.best_gap_pct - final_eval.best_gap_pct
        print(f"Augmentation impact: {aug_reduction:.2f} percentage points additional reduction")
    print()

    # Dataset growth
    print("=" * 60)
    print("DATASET GROWTH")
    print("=" * 60)
    for ev in phase_evaluations:
        print(f"  {ev.phase}: {ev.n_training_candidates} candidates")
    print()

    print("=" * 80)
    print(f"[OK] Validation with {CONFIG['n_augmentation_phases']} augmentation phase(s) complete!")
    if best_solution_ever.best_gap_pct <= 0:
        print("[SUCCESS] Found optimal solution!")
    else:
        print(f"[INFO] Best gap to optimal: {best_solution_ever.best_gap_pct:+.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
