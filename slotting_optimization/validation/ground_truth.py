from __future__ import annotations

from typing import Dict, List, Tuple
from itertools import permutations
import random
from datetime import datetime

from ..order_book import OrderBook
from ..warehouse import Warehouse
from ..item_locations import ItemLocations
from ..simulator import Simulator


def compute_brute_force_optimal(
    order_book: OrderBook,
    warehouse: Warehouse,
    items: List[str],
    storage_locations: List[str],
) -> Tuple[Dict[str, str], float]:
    """Enumerate all n! assignments and return the optimal one.

    Use for n_items <= 8 (40320 permutations max).

    Args:
        order_book: The order book to evaluate
        warehouse: The warehouse with distances
        items: List of item IDs to assign
        storage_locations: List of storage location IDs to assign to

    Returns:
        Tuple of (optimal_assignment_dict, optimal_distance)
    """
    if len(items) != len(storage_locations):
        raise ValueError(f"Number of items ({len(items)}) must equal number of locations ({len(storage_locations)})")

    if len(items) > 8:
        raise ValueError(f"Brute force only supports up to 8 items (got {len(items)}). Use heuristic methods for larger problems.")

    simulator = Simulator()
    best_distance = float('inf')
    best_assignment = None

    # Enumerate all permutations
    for perm in permutations(storage_locations):
        assignment_dict = {item: loc for item, loc in zip(items, perm)}
        assignment = ItemLocations.from_records([
            {"item_id": item, "location_id": loc}
            for item, loc in assignment_dict.items()
        ])

        distance, _ = simulator.simulate(order_book, warehouse, assignment)

        if distance < best_distance:
            best_distance = distance
            best_assignment = assignment_dict

    return best_assignment, best_distance


def generate_frequency_optimal_scenario(
    n_items: int,
    n_locations: int,
    n_orders: int,
    seed: int = 42,
) -> Tuple[OrderBook, Warehouse, ItemLocations, List[str], List[str], Dict[str, str], float]:
    """Generate a scenario where the optimal assignment is analytically known.

    Strategy:
    1. Create warehouse where loc_i has round-trip cost = i * base_cost (increasing)
    2. Generate orders where item_i is picked with frequency ~ (n_items - i) (decreasing)
    3. Optimal: item_0 (most frequent) -> loc_0 (lowest cost), etc. (diagonal assignment)

    Args:
        n_items: Number of items (must be <= n_locations)
        n_locations: Number of storage locations
        n_orders: Number of orders to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (order_book, warehouse, random_initial_assignment, items, storage_locations,
                  optimal_assignment_dict, optimal_distance)
    """
    if n_items > n_locations:
        raise ValueError(f"n_items ({n_items}) cannot exceed n_locations ({n_locations})")

    rng = random.Random(seed)

    # Create items and locations
    items = [f"sku{i}" for i in range(n_items)]
    all_locations = [f"L{i}" for i in range(n_locations)]
    storage_locations = all_locations[:n_items]  # Use first n_items locations

    # Create warehouse with structured distances
    # loc_i has round-trip cost = (i + 1) * base_cost
    base_cost = 10.0
    warehouse = Warehouse(
        locations=["start", "end"] + all_locations,
        start_point_id="start",
        end_point_id="end"
    )

    distance_map = {}

    # Set distances from start to locations and locations to end
    for i, loc in enumerate(all_locations):
        # Distance increases with location index
        dist_from_start = (i + 1) * base_cost / 2
        dist_to_end = (i + 1) * base_cost / 2

        distance_map[("start", loc)] = dist_from_start
        distance_map[(loc, "end")] = dist_to_end
        distance_map[(loc, "start")] = dist_from_start  # Symmetric for simplicity
        distance_map[("end", loc)] = dist_to_end

    # Return trip
    distance_map[("end", "start")] = 5.0
    distance_map[("start", "end")] = 5.0

    # Location-to-location distances (not used in basic simulation but needed for completeness)
    for i, loc_a in enumerate(all_locations):
        for j, loc_b in enumerate(all_locations):
            if loc_a != loc_b:
                distance_map[(loc_a, loc_b)] = abs(i - j) * base_cost * 0.5

    warehouse.set_distances_bulk(distance_map)

    # Generate orders with frequency pattern: item_i appears with weight (n_items - i)
    # This creates a clear frequency gradient
    weights = [(n_items - i) for i in range(n_items)]
    total_weight = sum(weights)

    order_dicts = []
    base_ts = int(datetime(2025, 1, 1, 8, 0, 0).timestamp())

    for oid in range(n_orders):
        # Pick item based on weights
        # Higher weight = more frequent
        rand_val = rng.random() * total_weight
        cumsum = 0
        selected_item = items[-1]  # Default to last item
        for i, weight in enumerate(weights):
            cumsum += weight
            if rand_val <= cumsum:
                selected_item = items[i]
                break

        order_dicts.append({
            "order_id": f"o{oid}",
            "item_id": selected_item,
            "timestamp": datetime.fromtimestamp(base_ts + oid * 60)
        })

    order_book = OrderBook.from_dicts_direct(order_dicts)

    # Optimal assignment: item_i -> loc_i (diagonal)
    optimal_assignment = {items[i]: storage_locations[i] for i in range(n_items)}

    # Compute optimal distance
    optimal_il = ItemLocations.from_records([
        {"item_id": item, "location_id": loc}
        for item, loc in optimal_assignment.items()
    ])
    simulator = Simulator()
    optimal_distance, _ = simulator.simulate(order_book, warehouse, optimal_il)

    # Create random initial assignment
    shuffled_locs = storage_locations[:]
    rng.shuffle(shuffled_locs)
    random_assignment = ItemLocations.from_records([
        {"item_id": item, "location_id": loc}
        for item, loc in zip(items, shuffled_locs)
    ])

    return order_book, warehouse, random_assignment, items, storage_locations, optimal_assignment, optimal_distance
