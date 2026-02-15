"""
Direct differentiable optimization of warehouse slotting.

Instead of using the GNN, we directly compute a differentiable approximation
of the travel distance based on the matrix representation.

The key formula is:
  distance ≈ sum_{i,j} seq_mat[i,j] * loc_mat[item_loc[i], item_loc[j]]
             + sum_i (start_dist[item_loc[i]] + end_dist[item_loc[i]])

With a soft assignment matrix, this becomes differentiable.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slotting_optimization.generator import DataGenerator
from slotting_optimization.simulator import Simulator, build_matrices_fast
from slotting_optimization.item_locations import ItemLocations


def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20, temperature: float = 0.1):
    """Convert logits to doubly-stochastic matrix."""
    M = log_alpha / temperature
    for _ in range(n_iters):
        M = M - M.logsumexp(dim=1, keepdim=True)
        M = M - M.logsumexp(dim=0, keepdim=True)
    return M.exp()


def compute_differentiable_distance(
    soft_assignment: torch.Tensor,
    seq_mat: torch.Tensor,
    loc_mat: torch.Tensor,
    n_items: int,
    n_storage: int,
) -> torch.Tensor:
    """
    Compute travel distance using soft assignment.

    Args:
        soft_assignment: [n_items, n_storage] soft permutation matrix
        seq_mat: [n_items, n_items] sequence counts
        loc_mat: [n_locs, n_locs] location distances (includes start, end)
        n_items: number of items
        n_storage: number of storage locations

    Returns:
        Predicted travel distance (scalar)
    """
    # loc_mat layout: [storage_0, ..., storage_{n-1}, start, end]
    start_idx = n_storage
    end_idx = n_storage + 1

    # Compute expected item-to-item distances
    # For each item pair (i, j), the expected distance is:
    #   E[dist(loc[i], loc[j])] = sum_{l1, l2} P(item_i at l1) * P(item_j at l2) * dist(l1, l2)
    # This is: soft_assignment[i] @ loc_mat[:n_storage, :n_storage] @ soft_assignment[j].T

    # Storage-to-storage distances
    storage_dists = loc_mat[:n_storage, :n_storage]

    # Item-to-item expected distances: [n_items, n_items]
    # = soft_assignment @ storage_dists @ soft_assignment.T
    item_item_dists = soft_assignment @ storage_dists @ soft_assignment.T

    # Total travel from item sequences
    seq_travel = (seq_mat * item_item_dists).sum()

    # Start-to-item distances
    start_dists = loc_mat[start_idx, :n_storage]  # [n_storage]
    # Expected start distance for each item = soft_assignment @ start_dists
    item_start_dists = soft_assignment @ start_dists  # [n_items]

    # Item-to-end distances
    end_dists = loc_mat[:n_storage, end_idx]  # [n_storage]
    item_end_dists = soft_assignment @ end_dists  # [n_items]

    # Count picks per item (simplified: sum of sequence counts)
    picks_per_item = seq_mat.sum(dim=1) + seq_mat.sum(dim=0)
    picks_per_item = picks_per_item / 2  # Avoid double counting

    # Total start/end travel
    start_travel = (picks_per_item * item_start_dists).sum()
    end_travel = (picks_per_item * item_end_dists).sum()

    total = seq_travel + start_travel + end_travel

    return total


def optimize_slotting(
    seq_mat: np.ndarray,
    loc_mat: np.ndarray,
    n_items: int,
    n_storage: int,
    initial_assignment: np.ndarray,
    n_steps: int = 100,
    lr: float = 1.0,
    verbose: bool = True,
):
    """
    Optimize item-location assignment using gradient descent.

    Args:
        seq_mat: [n_items, n_items] sequence count matrix
        loc_mat: [n_locs, n_locs] location distance matrix
        n_items: number of items
        n_storage: number of storage locations
        initial_assignment: [n_items] initial assignment (item i -> location initial_assignment[i])
        n_steps: optimization steps
        lr: learning rate
        verbose: print progress

    Returns:
        result dict with original and optimized assignments
    """
    # Convert to tensors
    seq_mat_t = torch.tensor(seq_mat, dtype=torch.float32)
    loc_mat_t = torch.tensor(loc_mat, dtype=torch.float32)

    # Initialize logits from current assignment
    log_alpha = torch.zeros(n_items, n_storage, requires_grad=True)
    for i, loc in enumerate(initial_assignment):
        log_alpha.data[i, loc] = 3.0  # Bias toward initial

    optimizer = torch.optim.Adam([log_alpha], lr=lr)

    # Compute original distance
    orig_soft = torch.zeros(n_items, n_storage)
    for i, loc in enumerate(initial_assignment):
        orig_soft[i, loc] = 1.0
    orig_dist = compute_differentiable_distance(
        orig_soft, seq_mat_t, loc_mat_t, n_items, n_storage
    ).item()

    if verbose:
        print(f"Original (differentiable) distance: {orig_dist:.1f}")

    # Temperature annealing
    temp = 1.0
    temp_decay = 0.95

    history = [orig_dist]

    for step in range(n_steps):
        optimizer.zero_grad()

        soft_assign = sinkhorn(log_alpha, n_iters=20, temperature=temp)
        dist = compute_differentiable_distance(
            soft_assign, seq_mat_t, loc_mat_t, n_items, n_storage
        )

        dist.backward()
        optimizer.step()

        history.append(dist.item())
        temp *= temp_decay

        if verbose and step % 20 == 0:
            print(f"  Step {step:3d}: distance={dist.item():.1f}, temp={temp:.4f}")

    # Final discretization
    with torch.no_grad():
        final_soft = sinkhorn(log_alpha, n_iters=50, temperature=0.001)
        cost_matrix = -final_soft.numpy()
        _, col_ind = linear_sum_assignment(cost_matrix)
        final_assignment = col_ind

    # Compute final discrete distance
    final_soft_discrete = torch.zeros(n_items, n_storage)
    for i, loc in enumerate(final_assignment):
        final_soft_discrete[i, loc] = 1.0
    final_dist = compute_differentiable_distance(
        final_soft_discrete, seq_mat_t, loc_mat_t, n_items, n_storage
    ).item()

    if verbose:
        print(f"\nFinal (discrete) distance: {final_dist:.1f}")
        print(f"Differentiable improvement: {(orig_dist - final_dist) / orig_dist * 100:.2f}%")

    return {
        "original_assignment": initial_assignment,
        "optimized_assignment": final_assignment,
        "original_distance": orig_dist,
        "optimized_distance": final_dist,
        "history": history,
        "log_alpha": log_alpha.detach(),  # For generating alternative candidates
    }


def generate_candidates(log_alpha: torch.Tensor, n_candidates: int = 10):
    """Generate multiple discrete candidates from soft assignment."""
    candidates = []

    # Best Hungarian assignment
    with torch.no_grad():
        soft = sinkhorn(log_alpha, n_iters=50, temperature=0.001)
        cost = -soft.numpy()
        _, cols = linear_sum_assignment(cost)
        candidates.append(cols.copy())

        # Sample with different temperatures
        for temp in [0.01, 0.05, 0.1, 0.2]:
            soft = sinkhorn(log_alpha, n_iters=20, temperature=temp)
            cost = -soft.numpy()
            _, cols = linear_sum_assignment(cost)
            if not any(np.array_equal(cols, c) for c in candidates):
                candidates.append(cols.copy())

    return candidates


def optimize_with_simulator_validation(
    ob, il, w, items, locs,
    seq_mat: np.ndarray,
    loc_mat: np.ndarray,
    n_items: int,
    n_storage: int,
    initial_assignment: np.ndarray,
    max_iters: int = 50,
    verbose: bool = True,
):
    """
    Hybrid optimization: use gradients to identify swaps, validate with simulator.
    """
    sim = Simulator()

    current_assignment = initial_assignment.copy()

    # Get initial simulator distance
    current_il = ItemLocations.from_records([
        {"item_id": items[i], "location_id": locs[current_assignment[i]]}
        for i in range(n_items)
    ])
    current_dist, _ = sim.simulate(ob, w, current_il)

    if verbose:
        print(f"Initial simulator distance: {current_dist:.1f}")

    history = [current_dist]

    for iteration in range(max_iters):
        # Find best improving swap by checking with simulator
        best_swap = None
        best_improvement = 0

        for i in range(n_items):
            for j in range(i + 1, n_items):
                # Try swap
                swapped = current_assignment.copy()
                swapped[i], swapped[j] = swapped[j], swapped[i]

                swapped_il = ItemLocations.from_records([
                    {"item_id": items[k], "location_id": locs[swapped[k]]}
                    for k in range(n_items)
                ])
                swapped_dist, _ = sim.simulate(ob, w, swapped_il)

                improvement = current_dist - swapped_dist
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_swap = (i, j)
                    best_dist = swapped_dist

        if best_swap is None or best_improvement < 0.1:
            if verbose:
                print(f"  Iter {iteration}: No improving swap found. Stopping.")
            break

        # Apply best swap
        i, j = best_swap
        current_assignment[i], current_assignment[j] = current_assignment[j], current_assignment[i]
        current_dist = best_dist
        history.append(current_dist)

        if verbose:
            print(f"  Iter {iteration}: Swap {items[i]}<->{items[j]}, "
                  f"distance: {current_dist:.1f} (improved by {best_improvement:.1f})")

    final_improvement = (history[0] - current_dist) / history[0] * 100

    if verbose:
        print(f"\nFinal simulator distance: {current_dist:.1f}")
        print(f"Total improvement: {final_improvement:.2f}%")

    return {
        "original_assignment": initial_assignment,
        "optimized_assignment": current_assignment,
        "original_distance": history[0],
        "optimized_distance": current_dist,
        "history": history,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_items", type=int, default=10)
    parser.add_argument("--n_orders", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--method", type=str, default="gradient", choices=["gradient", "simulator"])
    args = parser.parse_args()

    print("=" * 60)
    print("Direct Differentiable Slotting Optimization")
    print("=" * 60)

    # Generate a sample
    gen = DataGenerator()
    samples = gen.generate_samples(
        args.n_items, args.n_items, args.n_orders, 1, 5,
        n_samples=1, distances_fixed=True, seed=args.seed
    )
    ob, il, w = samples[0]

    # Get matrices
    loc_mat, seq_mat, item_loc_mat, locs, items = build_matrices_fast(ob, il, w)

    n_items = len(items)
    n_storage = len(locs) - 2  # Exclude start, end
    n_locs = len(locs)

    print(f"\nProblem size: {n_items} items, {n_storage} storage locations")

    # Get initial assignment
    initial_assignment = np.argmax(item_loc_mat, axis=1)
    print(f"Initial assignment: {initial_assignment}")

    # Verify with simulator
    sim = Simulator()
    sim_dist, _ = sim.simulate(ob, w, il)
    print(f"Simulator distance: {sim_dist:.1f}")

    # Run optimization
    print(f"\nOptimizing (method={args.method})...")

    if args.method == "simulator":
        result = optimize_with_simulator_validation(
            ob=ob, il=il, w=w, items=items, locs=locs,
            seq_mat=seq_mat,
            loc_mat=loc_mat,
            n_items=n_items,
            n_storage=n_storage,
            initial_assignment=initial_assignment,
            max_iters=args.n_steps,
        )
        opt_assign = result["optimized_assignment"]
        sim_dist_opt = result["optimized_distance"]
    else:
        result = optimize_slotting(
            seq_mat=seq_mat,
            loc_mat=loc_mat,
            n_items=n_items,
            n_storage=n_storage,
            initial_assignment=initial_assignment,
            n_steps=args.n_steps,
            lr=1.0,
        )

        # Generate multiple candidates and validate with simulator
        print("\nValidating candidates with simulator...")
        candidates = generate_candidates(result["log_alpha"], n_candidates=10)
        candidates.insert(0, initial_assignment)  # Include original

        best_sim_dist = sim_dist
        best_assignment = initial_assignment

        for i, cand in enumerate(candidates):
            new_records = []
            for j, item in enumerate(items):
                loc_idx = cand[j]
                new_records.append({"item_id": item, "location_id": locs[loc_idx]})
            il_cand = ItemLocations.from_records(new_records)
            cand_dist, _ = sim.simulate(ob, w, il_cand)

            if cand_dist < best_sim_dist:
                best_sim_dist = cand_dist
                best_assignment = cand

        opt_assign = best_assignment
        sim_dist_opt = best_sim_dist

    print(f"\nOptimized assignment: {opt_assign}")
    print(f"Final simulator distance: {sim_dist_opt:.1f}")
    print(f"Improvement: {(sim_dist - sim_dist_opt) / sim_dist * 100:.2f}%")

    # Show changes
    print("\nAssignment changes:")
    changes = 0
    for i in range(n_items):
        if initial_assignment[i] != opt_assign[i]:
            print(f"  {items[i]}: {locs[initial_assignment[i]]} -> {locs[opt_assign[i]]}")
            changes += 1
    print(f"Total items moved: {changes}/{n_items}")


if __name__ == "__main__":
    main()
