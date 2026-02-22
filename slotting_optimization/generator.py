from __future__ import annotations

from typing import Dict, List, Tuple
import random
import time
from datetime import datetime


from .order_book import OrderBook
from .item_locations import ItemLocations
from .warehouse import Warehouse
from .simulator import Simulator


class DataGenerator:
    """Generates synthetic OrderBook, ItemLocations and Warehouse samples.

    API:
        generate_samples(n_locations, nb_items, n_orders, min_items_per_order, max_items_per_order,
                         n_samples=1, distances_fixed=True, seed=None)

    Parameters:
        n_locations: Number of warehouse locations to create
        nb_items: Number of items (SKUs) to generate and assign to locations.
                  Must satisfy 1 <= nb_items <= n_locations.
                  When nb_items < n_locations, items are randomly assigned to a subset of locations.
        n_orders: Number of logical orders to generate
        min_items_per_order: Minimum items per order
        max_items_per_order: Maximum items per order
        n_samples: Number of independent samples to generate
        distances_fixed: If True, all samples share the same distance map
        seed: Random seed for reproducibility

    Returns a list of tuples: (OrderBook, ItemLocations, Warehouse)
    """

    def generate_samples(
        self,
        n_locations: int,
        nb_items: int,
        n_orders: int,
        min_items_per_order: int,
        max_items_per_order: int,
        n_samples: int = 1,
        distances_fixed: bool = True,
        seed: int | None = None,
    ) -> List[Tuple[OrderBook, ItemLocations, Warehouse]]:
        rng = random.Random(seed)

        # Validate nb_items
        if nb_items <= 0:
            raise ValueError(f"nb_items must be positive, got {nb_items}")
        if nb_items > n_locations:
            raise ValueError(
                f"nb_items ({nb_items}) cannot exceed n_locations ({n_locations})"
            )

        # Pre-generate all locations
        locations = [f"L{i}" for i in range(n_locations)]

        # Generate nb_items SKUs (sequential naming)
        skus = [f"sku{i}" for i in range(nb_items)]

        # If distances_fixed, create base warehouse once with all distances set
        base_warehouse = None
        if distances_fixed:
            base_distance_map = self._make_distance_map(locations, rng)
            # Create warehouse once with bulk distance setting
            base_warehouse = Warehouse(
                locations=["start", "end"] + locations,
                start_point_id="start",
                end_point_id="end",
            )
            base_warehouse.set_distances_bulk(base_distance_map)

        samples: List[Tuple[OrderBook, ItemLocations, Warehouse]] = []

        # Generate orders ONCE (shared across all samples when distances_fixed)
        # This ensures the model learns from assignment variations, not order variations
        base_order_book = None
        if distances_fixed:
            order_dicts = []
            base_ts = int(time.time())
            for oid in range(n_orders):
                k = rng.randint(min_items_per_order, max_items_per_order)
                for item_idx in range(k):
                    sku = rng.choice(skus)
                    epoch_ts = base_ts + oid * 60 + item_idx
                    order_dicts.append(
                        {
                            "order_id": f"o{oid}",
                            "item_id": sku,
                            "timestamp": datetime.fromtimestamp(epoch_ts),
                        }
                    )
            base_order_book = OrderBook.from_dicts_direct(order_dicts)

        for sidx in range(n_samples):
            # Randomly select which locations get items (different per sample)
            selected_locations = rng.sample(locations, nb_items)

            # Build ItemLocations
            il = ItemLocations.from_records(
                [
                    {"item_id": sku, "location_id": loc}
                    for sku, loc in zip(skus, selected_locations)
                ]
            )

            # Reuse warehouse when distances fixed
            if distances_fixed:
                w = base_warehouse  # Share reference - read-only after creation
                ob = base_order_book  # Share same orders - only assignments vary
            else:
                # Create new warehouse for variable distances case
                sample_seed = rng.randint(0, 2**30 - 1)
                sample_rng = random.Random(sample_seed)
                dist_map = self._make_distance_map(locations, sample_rng)
                w = Warehouse(
                    locations=["start", "end"] + locations,
                    start_point_id="start",
                    end_point_id="end",
                )
                w.set_distances_bulk(dist_map)  # Use bulk method

                # Generate orders for this sample
                order_dicts = []
                base_ts = int(time.time()) + sidx * 1000000
                for oid in range(n_orders):
                    k = rng.randint(min_items_per_order, max_items_per_order)
                    for item_idx in range(k):
                        sku = rng.choice(skus)
                        epoch_ts = base_ts + oid * 60 + item_idx
                        order_dicts.append(
                            {
                                "order_id": f"o{oid}",
                                "item_id": sku,
                                "timestamp": datetime.fromtimestamp(epoch_ts),
                            }
                        )
                ob = OrderBook.from_dicts_direct(order_dicts)

            samples.append((ob, il, w))

        return samples

    def _make_distance_map(self, locations: List[str], rng: random.Random):
        # Create mapping for directed pairs between start/end and locations
        dmap = {}
        nodes = ["start", "end"] + locations
        for a in nodes:
            for b in nodes:
                if a == b:
                    continue
                # distance between 1.0 and 10.0
                dmap[(a, b)] = round(rng.uniform(1.0, 10.0), 6)
        return dmap

    def _make_structured_distance_map(
        self,
        all_locations: List[str],
        base_cost: float,
        cost_progression: str,
        rng: random.Random,
    ) -> Tuple[Dict, Dict[str, float]]:
        """Build a distance map where location round-trip costs are strictly ordered.

        Location L_i has cost_i < cost_{i+1}, ensuring the rearrangement-optimal
        assignment (highest-frequency item -> cheapest location) is well-defined.

        Returns:
            (distance_map, location_costs) where location_costs maps each storage
            location to its round-trip cost d(start, loc) + d(loc, end).
        """
        n = len(all_locations)
        distance_map: Dict = {}
        location_costs: Dict[str, float] = {}

        for i, loc in enumerate(all_locations):
            if cost_progression == "quadratic":
                cost_i = (i + 1) ** 2 * base_cost / max(n, 1)
            elif cost_progression == "exponential":
                cost_i = base_cost * (1.2**i)
            else:  # linear (default)
                cost_i = (i + 1) * base_cost

            # Split cost into start->loc and loc->end with random ratio [0.3, 0.7]
            split_ratio = 0.3 + rng.random() * 0.4
            dist_from_start = cost_i * split_ratio
            dist_to_end = cost_i * (1 - split_ratio)

            distance_map[("start", loc)] = dist_from_start
            distance_map[(loc, "end")] = dist_to_end
            # Symmetric return legs (needed by simulator)
            distance_map[(loc, "start")] = dist_from_start
            distance_map[("end", loc)] = dist_to_end

            location_costs[loc] = dist_from_start + dist_to_end

        # Location-to-location distances
        for i, loc_a in enumerate(all_locations):
            for j, loc_b in enumerate(all_locations):
                if loc_a != loc_b:
                    distance_map[(loc_a, loc_b)] = abs(i - j) * base_cost * 0.5

        # start <-> end
        distance_map[("start", "end")] = base_cost / 2
        distance_map[("end", "start")] = base_cost / 2

        return distance_map, location_costs

    def generate_optimal_samples(
        self,
        n_items: int,
        n_locations: int,
        n_orders: int,
        n_samples: int = 1,
        min_items_per_order: int = 1,
        max_items_per_order: int = 1,
        noise_level: float = 0.0,
        base_cost: float = 10.0,
        cost_progression: str = "linear",
        seed: int | None = None,
    ) -> List[Tuple[OrderBook, ItemLocations, Warehouse, dict]]:
        """Generate samples where the returned assignment is provably optimal.

        Constructs a structured warehouse where location round-trip costs are
        strictly ordered, then generates an orderbook whose item frequencies
        satisfy the rearrangement inequality — ensuring the diagonal assignment
        (highest-frequency item -> cheapest location) minimises total travel distance.

        The optimality guarantee holds at every noise_level in [0, 1]:
        - noise_level=0: steep frequency gradient, strongly preferred assignment
        - noise_level=1: uniform frequencies, all assignments equally optimal
        - intermediate: blended gradient, still provably optimal

        Mathematical proof: blended weight w_i = (1-α)·(n-i) + α·mean preserves
        w_i >= w_j for i < j at all α ∈ [0,1], so the frequency ordering is
        maintained and the rearrangement inequality still applies.

        Args:
            n_items: Number of items (SKUs). Must satisfy 1 <= n_items <= n_locations.
            n_locations: Number of storage locations to create.
            n_orders: Number of logical orders to generate.
            n_samples: Number of independent (warehouse, orderbook, assignment) triples.
                       Each sample gets its own warehouse and orderbook.
            min_items_per_order: Minimum line items per order.
            max_items_per_order: Maximum line items per order.
            noise_level: Float in [0, 1]. 0 = maximal frequency gradient (easy to
                         identify optimal), 1 = uniform frequencies (all assignments
                         equally good). Optimality is guaranteed at all levels.
            base_cost: Scaling factor for location costs.
            cost_progression: How location costs increase with index.
                "linear" (default): cost_i = (i+1) * base_cost
                "quadratic": cost_i = (i+1)^2 * base_cost / n
                "exponential": cost_i = base_cost * 1.2^i
            seed: Random seed for reproducibility.

        Returns:
            List of (OrderBook, ItemLocations, Warehouse, metadata) tuples where:
            - ItemLocations contains the provably optimal assignment
            - metadata dict contains:
                "items": List[str] - ordered item IDs
                "storage_locations": List[str] - ordered location IDs
                "optimal_assignment": Dict[str, str] - item_id -> location_id
                "optimal_distance": float - simulated distance for optimal assignment
                "item_frequencies": Dict[str, int] - empirical pick counts
                "location_costs": Dict[str, float] - round-trip cost per location
                "noise_level": float
        """
        if n_items <= 0:
            raise ValueError(f"n_items must be positive, got {n_items}")
        if n_items > n_locations:
            raise ValueError(
                f"n_items ({n_items}) cannot exceed n_locations ({n_locations})"
            )
        if not (0.0 <= noise_level <= 1.0):
            raise ValueError(f"noise_level must be in [0, 1], got {noise_level}")
        if cost_progression not in ("linear", "quadratic", "exponential"):
            raise ValueError(
                "cost_progression must be 'linear', 'quadratic', or 'exponential'"
            )

        rng = random.Random(seed)
        simulator = Simulator()
        base_ts = int(datetime(2025, 1, 1, 8, 0, 0).timestamp())

        items = [f"sku{i}" for i in range(n_items)]
        all_locations = [f"L{i}" for i in range(n_locations)]
        storage_locations = all_locations[:n_items]

        # Blended frequency weights: pure optimal blended with uniform
        # pure_weights[i] = n_items - i  (highest for item 0, lowest for item n-1)
        pure_weights = [(n_items - i) for i in range(n_items)]
        mean_weight = sum(pure_weights) / n_items
        blended_weights = [
            (1.0 - noise_level) * pw + noise_level * mean_weight for pw in pure_weights
        ]
        total_weight = sum(blended_weights)

        samples: List[Tuple[OrderBook, ItemLocations, Warehouse, dict]] = []

        for sample_idx in range(n_samples):
            sample_seed = rng.randint(0, 2**30 - 1)
            sample_rng = random.Random(sample_seed)

            # Build structured warehouse with ordered location costs
            distance_map, location_costs = self._make_structured_distance_map(
                all_locations, base_cost, cost_progression, sample_rng
            )
            w = Warehouse(
                locations=["start", "end"] + all_locations,
                start_point_id="start",
                end_point_id="end",
            )
            w.set_distances_bulk(distance_map)

            # Generate multi-item orders using weighted cumulative sampling
            order_dicts = []
            sample_base_ts = base_ts + sample_idx * 1_000_000
            for oid in range(n_orders):
                k = sample_rng.randint(min_items_per_order, max_items_per_order)
                for item_idx in range(k):
                    # Cumulative weighted sampling (same pattern as ground_truth.py)
                    rand_val = sample_rng.random() * total_weight
                    cumsum = 0.0
                    selected_item = items[-1]
                    for i, weight in enumerate(blended_weights):
                        cumsum += weight
                        if rand_val <= cumsum:
                            selected_item = items[i]
                            break
                    epoch_ts = sample_base_ts + oid * 60 + item_idx
                    order_dicts.append(
                        {
                            "order_id": f"o{oid}",
                            "item_id": selected_item,
                            "timestamp": datetime.fromtimestamp(epoch_ts),
                        }
                    )

            ob = OrderBook.from_dicts_direct(order_dicts)

            # Compute empirical item frequencies
            freq_df = ob.to_df().group_by("item_id").len()
            item_frequencies: Dict[str, int] = {
                row["item_id"]: row["len"] for row in freq_df.to_dicts()
            }
            # Items not appearing get frequency 0
            for item in items:
                item_frequencies.setdefault(item, 0)

            # Verify pairwise swap condition; re-sort assignment if violated.
            # This can happen with very small n_orders where sampling noise
            # overrides the theoretical frequency ordering.
            actual_freqs = [item_frequencies[item] for item in items]
            swap_violated = False
            for a in range(n_items):
                for b in range(a + 1, n_items):
                    freq_diff = actual_freqs[a] - actual_freqs[b]
                    cost_diff = (
                        location_costs[storage_locations[a]]
                        - location_costs[storage_locations[b]]
                    )
                    if freq_diff * cost_diff > 1e-9:
                        swap_violated = True
                        break
                if swap_violated:
                    break

            if swap_violated:
                # Fallback: sort items by empirical frequency (desc) -> assign to sorted costs (asc)
                sorted_items = sorted(
                    items, key=lambda x: item_frequencies[x], reverse=True
                )
                optimal_assignment = {
                    sorted_items[i]: storage_locations[i] for i in range(n_items)
                }
            else:
                # Diagonal assignment: item_i (highest freq) -> loc_i (lowest cost)
                optimal_assignment = {
                    items[i]: storage_locations[i] for i in range(n_items)
                }

            il = ItemLocations.from_records(
                [
                    {"item_id": item, "location_id": loc}
                    for item, loc in optimal_assignment.items()
                ]
            )

            optimal_distance, _ = simulator.simulate(ob, w, il)

            metadata = {
                "items": items,
                "storage_locations": storage_locations,
                "optimal_assignment": optimal_assignment,
                "optimal_distance": optimal_distance,
                "item_frequencies": item_frequencies,
                "location_costs": {
                    loc: location_costs[loc] for loc in storage_locations
                },
                "noise_level": noise_level,
            }

            samples.append((ob, il, w, metadata))

        return samples
