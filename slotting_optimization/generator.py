from __future__ import annotations

from typing import List, Tuple
import random
import time
from datetime import datetime

import polars as pl

from .order_book import OrderBook
from .item_locations import ItemLocations
from .warehouse import Warehouse
from .models import Order


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
            raise ValueError(f"nb_items ({nb_items}) cannot exceed n_locations ({n_locations})")

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
                end_point_id="end"
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
                    order_dicts.append({
                        "order_id": f"o{oid}",
                        "item_id": sku,
                        "timestamp": datetime.fromtimestamp(epoch_ts)
                    })
            base_order_book = OrderBook.from_dicts_direct(order_dicts)

        for sidx in range(n_samples):
            # Randomly select which locations get items (different per sample)
            selected_locations = rng.sample(locations, nb_items)

            # Build ItemLocations
            il = ItemLocations.from_records([{"item_id": sku, "location_id": loc} for sku, loc in zip(skus, selected_locations)])

            # Reuse warehouse when distances fixed
            if distances_fixed:
                w = base_warehouse  # Share reference - read-only after creation
                ob = base_order_book  # Share same orders - only assignments vary
            else:
                # Create new warehouse for variable distances case
                sample_seed = rng.randint(0, 2**30 - 1)
                sample_rng = random.Random(sample_seed)
                dist_map = self._make_distance_map(locations, sample_rng)
                w = Warehouse(locations=["start", "end"] + locations,
                             start_point_id="start", end_point_id="end")
                w.set_distances_bulk(dist_map)  # Use bulk method

                # Generate orders for this sample
                order_dicts = []
                base_ts = int(time.time()) + sidx * 1000000
                for oid in range(n_orders):
                    k = rng.randint(min_items_per_order, max_items_per_order)
                    for item_idx in range(k):
                        sku = rng.choice(skus)
                        epoch_ts = base_ts + oid * 60 + item_idx
                        order_dicts.append({
                            "order_id": f"o{oid}",
                            "item_id": sku,
                            "timestamp": datetime.fromtimestamp(epoch_ts)
                        })
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
