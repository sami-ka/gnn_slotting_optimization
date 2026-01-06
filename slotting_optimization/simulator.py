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


def build_matrices(order_book: OrderBook, item_locations: ItemLocations, warehouse: Warehouse):
    """Build three matrices from inputs:

    - loc_mat: (L x L) numpy array of distances between warehouse locations (np.nan where missing)
    - seq_mat: (I x I) integer numpy array where seq_mat[i,j] is the count of item i -> item j sequences within orders
    - item_loc_mat: (I x L) binary numpy array where item_loc_mat[i,j]==1 iff item i is assigned to location j

    The function returns a tuple (loc_mat, seq_mat, item_loc_mat, locs, items) where
    `locs` and `items` are the lists giving the row/column ordering used for the matrices.
    """
    df: pl.DataFrame = order_book.to_df()

    # Locations (keep warehouse.locations() ordering)
    locs = warehouse.locations()
    m = len(locs)
    loc_index = {loc: i for i, loc in enumerate(locs)}

    # Location x Location matrix (float) with np.nan for missing distances
    loc_mat = np.full((m, m), np.nan, dtype=float)
    for i, a in enumerate(locs):
        for j, b in enumerate(locs):
            d = warehouse.get_distance(a, b)
            if d is not None:
                loc_mat[i, j] = float(d)

    # Items: include items present in orders or in the item_locations mapping
    items_set = set(df.get_column("item_id").to_list()) | set(item_locations.to_dict().keys())
    items = sorted(items_set)
    n = len(items)
    item_index = {it: idx for idx, it in enumerate(items)}

    # Item x Item sequence matrix
    seq_mat = np.zeros((n, n), dtype=np.int64)

    if df.height > 0:
        # Build per-order item sequences by timestamp without using list aggregation (compatibility across polars versions)
        order_items = {}
        for rec in df.sort("timestamp").to_dicts():
            oid = rec["order_id"]
            order_items.setdefault(oid, []).append(str(rec["item_id"]))
        for item_list in order_items.values():
            for a, b in zip(item_list, item_list[1:]):
                seq_mat[item_index[a], item_index[b]] += 1

    # Item x Location binary matrix
    item_loc_mat = np.zeros((n, m), dtype=np.int64)
    mapping = item_locations.to_dict()
    for it in items:
        loc = mapping.get(it)
        if loc is not None and loc in loc_index:
            item_loc_mat[item_index[it], loc_index[loc]] = 1

    return loc_mat, seq_mat, item_loc_mat, locs, items


def build_matrices_fast(order_book: OrderBook, item_locations: ItemLocations, warehouse: Warehouse):
    """Faster, vectorized implementation of build_matrices.

    Strategies used:
    - Build `loc_mat` by filling from warehouse's distance map (no nested LxL loops)
    - Use Polars windowed `shift` to compute next-item within each order, then groupby to count unique pairs
    - Use NumPy advanced indexing to construct `item_loc_mat` efficiently
    Returns same tuple as `build_matrices`.
    """
    df: pl.DataFrame = order_book.to_df()

    # Locations and index mapping
    locs = warehouse.locations()
    m = len(locs)
    loc_index = {loc: i for i, loc in enumerate(locs)}

    # Location matrix: fill from distance map (only iterate defined distances)
    loc_mat = np.full((m, m), np.nan, dtype=float)
    # Accessing private attribute _distance_map (module-local optimization)
    for (a, b), d in getattr(warehouse, "_distance_map", {}).items():
        if a in loc_index and b in loc_index:
            loc_mat[loc_index[a], loc_index[b]] = float(d)

    # Items set and mapping
    mapping = item_locations.to_dict()
    items_set = set(df.get_column("item_id").to_list()) | set(mapping.keys())
    items = sorted(items_set)
    n = len(items)
    item_index = {it: idx for idx, it in enumerate(items)}

    # Sequence matrix: use Polars windowed shift + groupby to count pairs
    seq_mat = np.zeros((n, n), dtype=np.int64)
    if df.height > 0:
        # Sort per order and compute next item per-order
        df_sorted = df.sort(["order_id", "timestamp"]) 
        df_next = df_sorted.with_columns(
            pl.col("item_id").shift(-1).over("order_id").alias("next_item")
        )
        pairs = (
            df_next.filter(pl.col("next_item").is_not_null())
            .group_by(["item_id", "next_item"])  # counts per pair
            .agg(pl.count().alias("cnt"))
        )
        # Vectorized mapping from item ids to integer indices and assignment
        if pairs.height > 0:
            # Temporary mapping table for items -> idx
            items_df = pl.DataFrame({"item": items, "idx": list(range(n))})
            # Join to get integer indices for item_id and next_item
            pairs = pairs.join(items_df.rename({"item": "item_id", "idx": "ai"}), on="item_id", how="inner")
            pairs = pairs.join(items_df.rename({"item": "next_item", "idx": "bi"}), on="next_item", how="inner")

            ai_arr = pairs.get_column("ai").to_numpy().astype(np.int64)
            bi_arr = pairs.get_column("bi").to_numpy().astype(np.int64)
            cnt_arr = pairs.get_column("cnt").to_numpy().astype(np.int64)

            # Vectorized assignment using numpy indexing
            seq_mat[ai_arr, bi_arr] = cnt_arr

    # Item x Location matrix using vectorized indexing
    item_loc_mat = np.zeros((n, m), dtype=np.int64)
    # Build lists of (item_idx, loc_idx)
    item_idxs = []
    loc_idxs = []
    for it, loc in mapping.items():
        if it in item_index and loc in loc_index:
            item_idxs.append(item_index[it])
            loc_idxs.append(loc_index[loc])
    if item_idxs:
        item_loc_mat[np.array(item_idxs, dtype=np.int64), np.array(loc_idxs, dtype=np.int64)] = 1

    np.fill_diagonal(loc_mat, 0)
    return loc_mat, seq_mat, item_loc_mat, locs, items
