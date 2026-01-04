from __future__ import annotations

from typing import List, Tuple

import numpy as np
import polars as pl
from scipy import sparse as sp

from .order_book import OrderBook
from .item_locations import ItemLocations
from .warehouse import Warehouse


class Simulator:
    """Simple simulator that computes travel distance for each order.

    Behavior:
      - Processes orders in increasing timestamp order.
      - For each order: travel start -> item_location -> end.
      - Between orders, add distance end -> start before starting the next order.

    The simulator requires the warehouse to have distances defined for all used
    legs; missing distances raise ValueError.
    """

    def simulate(self, order_book: OrderBook, warehouse: Warehouse, item_locations: ItemLocations) -> Tuple[float, List[float]]:
        df: pl.DataFrame = order_book.to_df().sort("timestamp")
        total_distance = 0.0
        per_order: List[float] = []

        start = warehouse.start_point
        end = warehouse.end_point
        if start is None or end is None:
            raise ValueError("Warehouse must define start_point and end_point")

        n = df.height
        if n == 0:
            return 0.0, []

        get_loc = item_locations.get_location
        get_dist = warehouse.get_distance

        # Build legs: each order contributes start->loc and loc->end with an order_idx
        legs_records = []
        item_ids = df.get_column("item_id").to_list()
        for idx, item_id in enumerate(item_ids):
            loc = get_loc(item_id)
            if loc is None:
                raise ValueError(f"No location for item {item_id}")
            legs_records.append({"order_idx": idx, "from": start, "to": loc})
            legs_records.append({"order_idx": idx, "from": loc, "to": end})

        # Add return legs between orders: end->start repeated n-1 times (no order_idx)
        for _ in range(max(0, n - 1)):
            legs_records.append({"order_idx": None, "from": end, "to": start})

        legs_df = pl.DataFrame(legs_records)

        # Build distance table for unique pairs used in legs_df
        unique_pairs = legs_df.select(["from", "to"]).unique()
        pairs = unique_pairs.to_dicts()
        dist_recs = []
        for p in pairs:
            d = get_dist(p["from"], p["to"])
            if d is None:
                raise ValueError(f"Missing distance from {p['from']} to {p['to']}")
            dist_recs.append({"from": p["from"], "to": p["to"], "distance": float(d)})

        dist_df = pl.DataFrame(dist_recs)

        # Join legs with distances
        joined = legs_df.join(dist_df, on=["from", "to"], how="left")
        if joined.filter(pl.col("distance").is_null()).height > 0:
            raise ValueError("Some legs have no matching distance after join")

        # Per-order distances: group by order_idx and sum
        per_order_df = (
            joined.filter(pl.col("order_idx").is_not_null())
            .group_by("order_idx")
            .agg(pl.col("distance").sum())
            .sort("order_idx")
        )
        per_order = [float(x) for x in per_order_df.get_column("distance").to_list()]

        total_distance = float(joined.get_column("distance").sum())

        return total_distance, per_order



    def simulate_sparse_matrix(self, order_book: OrderBook, warehouse: Warehouse, item_locations: ItemLocations) -> Tuple[float, List[float]]:
        """Sparse-matrix based simulation.

        Build a sparse count matrix C (CSR) where C[i,j] is the number
        of times travel from location i to j. Build a sparse distance
        matrix D that contains distances only for pairs present in C,
        then compute total_distance = sum(C.multiply(D)).
        """
        df: pl.DataFrame = order_book.to_df().sort("timestamp")

        start = warehouse.start_point
        end = warehouse.end_point
        if start is None or end is None:
            raise ValueError("Warehouse must define start_point and end_point")

        n = df.height
        if n == 0:
            return 0.0, []

        item_ids = df.get_column("item_id").to_list()

        # Build legs list
        legs = []
        for item_id in item_ids:
            loc = item_locations.get_location(item_id)
            if loc is None:
                raise ValueError(f"No location for item {item_id}")
            legs.append((start, loc))
            legs.append((loc, end))
        for _ in range(max(0, n - 1)):
            legs.append((end, start))

        # Unique locations involved
        locs = []
        loc_set = set()
        for a, b in legs:
            if a not in loc_set:
                loc_set.add(a)
                locs.append(a)
            if b not in loc_set:
                loc_set.add(b)
                locs.append(b)

        idx = {loc: i for i, loc in enumerate(locs)}
        m = len(locs)

        # Build COO arrays for counts and convert to CSR
        rows = [idx[a] for a, b in legs]
        cols = [idx[b] for a, b in legs]
        data = np.ones(len(rows), dtype=np.int64)
        C = sp.coo_matrix((data, (rows, cols)), shape=(m, m)).tocsr()

        # Get nonzero positions and fetch distances
        r_nonzero, c_nonzero = C.nonzero()
        if len(r_nonzero) == 0:
            return 0.0, []

        d_vals = []
        for r, c in zip(r_nonzero, c_nonzero):
            a = locs[r]
            b = locs[c]
            d = warehouse.get_distance(a, b)
            if d is None:
                raise ValueError(f"Missing distance from {a} to {b}")
            d_vals.append(float(d))

        D = sp.coo_matrix((d_vals, (r_nonzero, c_nonzero)), shape=(m, m)).tocsr()

        total_distance = float(C.multiply(D).sum())

        # Per-order distances (fast lookups)
        per_order = []
        for item_id in item_ids:
            loc = item_locations.get_location(item_id)
            d1 = warehouse.get_distance(start, loc)
            d2 = warehouse.get_distance(loc, end)
            per_order.append(float(d1 + d2))

        return total_distance, per_order
