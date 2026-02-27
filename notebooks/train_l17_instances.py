"""
Training GNN on L17_533 benchmark instances.

Loads real warehouse instances instead of synthetic ones. Uses known-optimal
solutions from _sol.json as ground truth.

Key differences from train_validation_with_augmentation.py:
- Instances loaded from L17_533 benchmark files
- No brute force needed (solutions provided)
- Partial assignment: only SKUS_TO_SLOT need placement, rest are fixed
- 220 storage locations (much larger than synthetic 10x10)

Note: The benchmark's "Best known objective" uses TSP-based routing within
each order, while our simulator uses start→pick→end per pick. Distances
are not directly comparable, but relative rankings are preserved.

Usage:
    cd notebooks && uv run train_l17_instances.py
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

from slotting_optimization.instance_loader import load_l17_instance, load_single_vehicle_instances
from slotting_optimization.gnn_builder import build_graph_3d_sparse
from slotting_optimization.simulator import Simulator
from slotting_optimization.inverse_optimizer import (
    optimize_assignment,
    assignment_to_graph_data,
    extract_current_assignment,
)
from slotting_optimization.item_locations import ItemLocations
from slotting_optimization.validation.scenario import ValidationScenario
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


# ============================================================================
# Candidate Pool Evaluation (same structure as train_validation_with_augmentation)
# ============================================================================


@dataclass
class CandidateResult:
    source: str
    assignment: np.ndarray
    simulated_distance: float
    gap_to_optimal_pct: float


@dataclass
class PhaseEvaluation:
    phase: str
    epoch: int
    best_distance: float
    best_gap_pct: float
    best_source: str
    best_assignment: np.ndarray
    best_training_distance: float
    best_training_gap_pct: float
    best_inverse_opt_distance: float
    best_inverse_opt_gap_pct: float
    n_training_candidates: int
    n_inverse_opt_candidates: int


def get_gnn_predictions(model, dataset: list, mean_y: float, std_y: float) -> List[float]:
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
    raw_sample: tuple,
    simulator: Simulator,
) -> float:
    order_book, original_il, warehouse = raw_sample
    items = sorted(original_il.to_dict().keys())
    storage_locs = sorted([
        loc for loc in warehouse.locations()
        if loc not in (warehouse.start_point, warehouse.end_point)
    ])
    records = []
    for i, item in enumerate(items):
        loc_idx = assignment[i]
        records.append({"item_id": item, "location_id": storage_locs[loc_idx]})
    il = ItemLocations.from_records(records)
    distance, _ = simulator.simulate(order_book, warehouse, il)
    return distance


def evaluate_candidate_pool(
    model: nn.Module,
    accumulated_dataset: list,
    raw_samples: list,
    scenario: ValidationScenario,
    phase: str = "phase1",
    epoch: int = 0,
) -> PhaseEvaluation:
    simulator = Simulator()
    optimal_distance = scenario.optimal_distance
    results = []
    print(f"    Evaluating {len(accumulated_dataset)} candidates...")
    for idx, data in enumerate(accumulated_dataset):
        assignment = extract_current_assignment(data)
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
        phase=phase, epoch=epoch,
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
    print(f"\n    {'='*60}")
    print(f"    PHASE EVALUATION: {eval_result.phase} (epoch {eval_result.epoch})")
    print(f"    {'='*60}")
    print(f"    Optimal distance: {optimal_distance:.2f}")
    print(f"    Candidates evaluated: {eval_result.n_training_candidates}")
    print(f"    Best distance: {eval_result.best_distance:.2f} (gap: {eval_result.best_gap_pct:+.2f}%)")
    print(f"    {'='*60}")


# ============================================================================
# Model Architecture (identical to train_validation_with_augmentation.py)
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
        if isinstance(data_or_x, Data):
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
            x = data_or_x
        edge_attr_enc = self.edge_encoder(edge_attr)
        for layer in self.layers:
            x, edge_attr_enc = layer(x, edge_index, edge_attr_enc)
        graph_emb = global_add_pool(x, batch)
        out = self.regressor(graph_emb)
        return out.squeeze(-1)


# ============================================================================
# Configuration
# ============================================================================

def get_config_for_instance(n_items: int, n_locations: int) -> dict:
    """Config scaled for L17_533 instance dimensions."""
    complexity_ratio = (n_items / 5.0) ** 2
    return {
        "n_items": n_items,
        "n_locations": n_locations,
        "n_samples": 200,
        "train_split": 0.8,
        "seed": 42,
        "hidden_dim": 32,
        "num_layers": 3,
        "batch_size": 32,
        "grad_clip": 1.0,
        "track_every_n_epochs": 5,
        "epochs_phase1": max(30, int(30 * complexity_ratio)),
        "epochs_per_aug_phase": max(20, int(20 * complexity_ratio)),
        "learning_rate": 0.001 / (n_items / 5.0),
        "n_augmentation_phases": 5,
        "n_top_for_inverse_opt": 15,
        "perturbation_scale": 1.0,
        "augmentation_n_steps": max(50, n_items * 10),
    }


def train_epoch(model, train_loader, optimizer, criterion, device, grad_clip):
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
    model, top_graphs, top_raw_samples, mean_y, std_y, edge_mean, edge_std,
    n_steps=50, existing_assignments=None, perturbation_scale=0.0,
) -> tuple:
    model.eval()
    simulator = Simulator()
    augmented = []
    augmented_raw_and_graph = []
    if existing_assignments is None:
        existing_assignments = set()
    n_duplicates = 0
    for data, raw_sample in tqdm(zip(top_graphs, top_raw_samples), total=len(top_graphs), desc="  Augmenting"):
        result = optimize_assignment(
            model=model, data=data, mean_y=mean_y, std_y=std_y,
            n_steps=n_steps, perturbation_scale=perturbation_scale,
        )
        assignment_tuple = tuple(result["optimized_assignment"])
        if assignment_tuple in existing_assignments:
            n_duplicates += 1
            continue
        existing_assignments.add(assignment_tuple)
        graph_data = assignment_to_graph_data(
            optimized_assignment=result["optimized_assignment"],
            raw_sample=raw_sample, simulator=simulator,
            edge_mean=edge_mean, edge_std=edge_std,
            mean_y=mean_y, std_y=std_y,
        )
        augmented.append(graph_data)
        augmented_raw_and_graph.append((raw_sample, graph_data))
    print(f"  Generated {len(augmented)} augmented samples ({n_duplicates} duplicates skipped)")
    return augmented, existing_assignments, augmented_raw_and_graph


def generate_random_assignment(
    items: List[str],
    storage_locations: List[str],
    fixed_assignments: Dict[str, str],
    skus_to_slot: List[str],
    rng: random.Random,
) -> ItemLocations:
    """Generate a random assignment, keeping fixed SKUs in place.

    Only SKUS_TO_SLOT get random locations from unused storage locations.
    """
    # Start with fixed assignments
    assignment = dict(fixed_assignments)

    # Find unused storage locations
    used_locs = set(fixed_assignments.values())
    available_locs = [loc for loc in storage_locations if loc not in used_locs]

    # Randomly assign SKUS_TO_SLOT
    chosen_locs = rng.sample(available_locs, len(skus_to_slot))
    for sku, loc in zip(skus_to_slot, chosen_locs):
        assignment[sku] = loc

    records = [{"item_id": k, "location_id": v} for k, v in assignment.items()]
    return ItemLocations.from_records(records)


def main():
    # ========================================================================
    # Configuration
    # ========================================================================
    LAYOUT_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "L17_533", "Conventional",
    )

    print("=" * 80)
    print("L17_533 INSTANCE TRAINING")
    print("=" * 80)

    # ========================================================================
    # Load Instance
    # ========================================================================
    print("\n[1/8] Loading single-vehicle instances...")

    all_instances = load_single_vehicle_instances(LAYOUT_DIR)
    print(f"  Found {len(all_instances)} single-vehicle instances")

    # Use first instance for training (smallest/simplest)
    data = all_instances[0]
    INSTANCE_NAME = data.get("instance_name", "unknown")

    warehouse = data["warehouse"]
    order_book = data["order_book"]
    item_locations = data["item_locations"]
    solution_il = data["solution_item_locations"]
    skus_to_slot = data["skus_to_slot"]
    items = data["items"]
    storage_locations = data["storage_locations"]
    best_objective = data["best_objective"]

    print(f"  Items: {len(items)}, Storage locations: {len(storage_locations)}")
    print(f"  Orders (picks): {len(order_book)}")
    print(f"  SKUs to slot: {len(skus_to_slot)} ({skus_to_slot})")
    print(f"  Fixed assignments: {len(item_locations)}")
    print(f"  Vehicle capacity: {data['vehicle_capacity']}")
    print(f"  Best known objective (benchmark routing): {best_objective}")

    # Compute our simulator's distance for the solution
    simulator = Simulator()
    sol_distance, _ = simulator.simulate(order_book, warehouse, solution_il)
    print(f"  Solution distance (our routing): {sol_distance:.2f}")

    # Fixed assignments (non-null from VISIT_LOCATION_SECTION)
    fixed_assignments = item_locations.to_dict()

    CONFIG = get_config_for_instance(len(items), len(storage_locations))
    print(f"\n  Config: {CONFIG['n_samples']} samples, "
          f"{CONFIG['epochs_phase1']} phase1 epochs, "
          f"{CONFIG['n_augmentation_phases']} aug phases")

    # Create ValidationScenario using solution as optimal
    scenario = ValidationScenario(
        name="l17_single_vehicle",
        order_book=order_book,
        warehouse=warehouse,
        initial_assignment=item_locations,
        optimal_assignment=solution_il.to_dict(),
        optimal_distance=sol_distance,
        items=items,
        storage_locations=storage_locations,
    )

    device = torch.device("cpu")

    # ========================================================================
    # Generate Training Samples
    # ========================================================================
    print(f"\n[2/8] Generating {CONFIG['n_samples']} training samples...")
    print("  (Same orders/warehouse, random placement of SKUS_TO_SLOT)")

    rng = random.Random(CONFIG["seed"])
    samples = []
    for _ in range(CONFIG["n_samples"]):
        random_il = generate_random_assignment(
            items, storage_locations, fixed_assignments, skus_to_slot, rng,
        )
        samples.append((order_book, random_il, warehouse))

    print(f"  Generated {len(samples)} samples")

    # Build graphs
    print("  Building graphs...")
    list_data = []
    for ob, il, w in tqdm(samples, desc="  Building graphs"):
        g_data = build_graph_3d_sparse(
            order_book=ob, item_locations=il, warehouse=w,
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
    all_y = torch.cat([d.y for d in train_dataset])
    mean_y = all_y.mean().item()
    std_y = all_y.std().item()
    print(f"  Target: mean={mean_y:.1f}, std={std_y:.1f}")

    for d in train_dataset + test_dataset:
        d.y = (d.y - mean_y) / std_y

    all_edge_attrs = torch.cat([d.edge_attr for d in train_dataset], dim=0)
    edge_mean = all_edge_attrs.mean(dim=0)
    edge_std = all_edge_attrs.std(dim=0)
    for d in train_dataset + test_dataset:
        d.edge_attr = (d.edge_attr - edge_mean) / (edge_std + 1e-8)

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

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # ========================================================================
    # Phase evaluations tracking
    # ========================================================================
    phase_evaluations: List[PhaseEvaluation] = []
    best_solution_ever: Optional[PhaseEvaluation] = None

    # ========================================================================
    # Phase 1: Initial Training
    # ========================================================================
    print(f"\n[5/8] Phase 1: Initial training ({CONFIG['epochs_phase1']} epochs)...")

    best_val_loss = float("inf")

    print("  Evaluating candidate pool at epoch 0 (before training)...")
    eval_result = evaluate_candidate_pool(
        model=model, accumulated_dataset=train_dataset,
        raw_samples=raw_train_samples, scenario=scenario,
        phase="phase1_start", epoch=0,
    )
    phase_evaluations.append(eval_result)
    best_solution_ever = eval_result
    print_phase_evaluation(eval_result, scenario.optimal_distance)

    for epoch in range(CONFIG["epochs_phase1"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, CONFIG["grad_clip"])
        val_loss = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            marker = " *"
        else:
            marker = ""
        if (epoch + 1) % CONFIG["track_every_n_epochs"] == 0 or epoch == CONFIG["epochs_phase1"] - 1:
            print(f"  Epoch {epoch + 1:3d}: loss={val_loss:.4f}{marker}")

    print(f"\n  Phase 1 complete. Best val_loss: {best_val_loss:.4f}")
    print("  Evaluating candidate pool at end of Phase 1...")
    eval_result = evaluate_candidate_pool(
        model=model, accumulated_dataset=train_dataset,
        raw_samples=raw_train_samples, scenario=scenario,
        phase="phase1_end", epoch=CONFIG["epochs_phase1"],
    )
    phase_evaluations.append(eval_result)
    if eval_result.best_gap_pct < best_solution_ever.best_gap_pct:
        best_solution_ever = eval_result
    print_phase_evaluation(eval_result, scenario.optimal_distance)

    # ========================================================================
    # Phases 2+: Iterative Augmentation
    # ========================================================================
    current_train_dataset = list(train_dataset)
    accumulated_raw_samples = list(raw_train_samples)
    cumulative_epoch = CONFIG["epochs_phase1"]

    existing_assignments = set()
    for d in train_dataset:
        assignment = extract_current_assignment(d)
        existing_assignments.add(tuple(assignment))

    for aug_phase_num in range(1, CONFIG["n_augmentation_phases"] + 1):
        print(f"\n[...] Augmentation Phase {aug_phase_num}: Generating samples...")

        predictions = get_gnn_predictions(model, current_train_dataset, mean_y, std_y)
        sorted_indices = np.argsort(predictions)
        top_indices = sorted_indices[:CONFIG["n_top_for_inverse_opt"]]
        top_graphs = [current_train_dataset[i] for i in top_indices]
        top_raw = [accumulated_raw_samples[i] for i in top_indices]

        print(f"  Selected top {len(top_indices)} from {len(current_train_dataset)} accumulated samples")

        augmented, existing_assignments, augmented_raw_and_graph = augment_dataset(
            model=model, top_graphs=top_graphs, top_raw_samples=top_raw,
            mean_y=mean_y, std_y=std_y,
            edge_mean=edge_mean, edge_std=edge_std,
            n_steps=CONFIG["augmentation_n_steps"],
            existing_assignments=existing_assignments,
            perturbation_scale=CONFIG["perturbation_scale"],
        )

        print(f"\n[...] Augmentation Phase {aug_phase_num}: Training...")
        current_train_dataset = current_train_dataset + augmented
        for raw_s, _ in augmented_raw_and_graph:
            accumulated_raw_samples.append(raw_s)
        print(f"  Training set size: {len(current_train_dataset)}")

        train_loader_aug = DataLoader(current_train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        best_val_loss_aug = float("inf")
        for epoch in range(CONFIG["epochs_per_aug_phase"]):
            train_loss = train_epoch(model, train_loader_aug, optimizer, criterion, device, CONFIG["grad_clip"])
            val_loss = evaluate(model, test_loader, criterion, device)
            scheduler.step(val_loss)
            cumulative_epoch += 1
            if val_loss < best_val_loss_aug:
                best_val_loss_aug = val_loss
                marker = " *"
            else:
                marker = ""
            if (epoch + 1) % CONFIG["track_every_n_epochs"] == 0 or epoch == CONFIG["epochs_per_aug_phase"] - 1:
                print(f"  Epoch {cumulative_epoch:3d}: loss={val_loss:.4f}{marker}")

        print(f"\n  Phase {aug_phase_num} complete. Best val_loss: {best_val_loss_aug:.4f}")
        eval_result = evaluate_candidate_pool(
            model=model, accumulated_dataset=current_train_dataset,
            raw_samples=accumulated_raw_samples, scenario=scenario,
            phase=f"aug{aug_phase_num}_end", epoch=cumulative_epoch,
        )
        phase_evaluations.append(eval_result)
        if eval_result.best_gap_pct < best_solution_ever.best_gap_pct:
            best_solution_ever = eval_result
        print_phase_evaluation(eval_result, scenario.optimal_distance)

    # ========================================================================
    # Final Report
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONVERGENCE REPORT")
    print("=" * 80)
    print(f"\nInstances: {len(all_instances)} single-vehicle")
    print(f"Benchmark best objective (TSP routing): {best_objective}")
    print(f"Solution distance (our routing): {scenario.optimal_distance:.2f}")
    print()

    print("BEST SOLUTION FOUND")
    print("=" * 60)
    print(f"  Distance: {best_solution_ever.best_distance:.2f}")
    print(f"  Gap to solution: {best_solution_ever.best_gap_pct:+.2f}%")
    print(f"  Found at: {best_solution_ever.phase} (epoch {best_solution_ever.epoch})")
    print()

    print("CONVERGENCE TRAJECTORY")
    print("=" * 60)
    print(f"{'Phase':<15} {'Epoch':>6} {'Candidates':>12} {'Best Gap':>12}")
    print("-" * 50)
    for ev in phase_evaluations:
        print(f"{ev.phase:<15} {ev.epoch:>6} {ev.n_training_candidates:>12} {ev.best_gap_pct:>+11.2f}%")
    print()

    initial_eval = phase_evaluations[0]
    final_eval = phase_evaluations[-1]
    print(f"Initial best gap: {initial_eval.best_gap_pct:+.2f}%")
    print(f"Final best gap: {final_eval.best_gap_pct:+.2f}%")
    print(f"Total reduction: {initial_eval.best_gap_pct - final_eval.best_gap_pct:.2f} pp")
    print()

    if best_solution_ever.best_gap_pct <= 0:
        print("[SUCCESS] Found optimal solution!")
    else:
        print(f"[INFO] Best gap to optimal: {best_solution_ever.best_gap_pct:+.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
