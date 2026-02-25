"""Load L17_533 benchmark instances into repo data structures.

Each L17_533 layout directory contains:
- tsplib_parent.json: warehouse layout with coordinates, depots, obstacles
- instances/<name>/: per-instance files with orders, assignments, solutions

Usage:
    result = load_l17_instance("L17_533/Conventional", "c10_8502")
    ob, w, il = result["order_book"], result["warehouse"], result["item_locations"]
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

from .item_locations import ItemLocations
from .order_book import OrderBook
from .warehouse import Warehouse


def get_storage_location_ids(parent: dict) -> List[str]:
    """Extract pick (storage) location IDs from a parent layout.

    Filters out depot IDs and obstacle corner IDs, returning only the
    ``num_pick_locs_warehouse`` storage locations.
    """
    depots = set(str(d) for d in parent["DEPOTS"])
    obstacle_ids: set[str] = set()
    for corners in parent["OBSTACLES"].values():
        obstacle_ids.update(str(c) for c in corners)

    all_ids = set(parent["LOCATION_COORD_SECTION"].keys())
    pick_ids = sorted(all_ids - depots - obstacle_ids, key=lambda x: int(x))

    expected = parent.get("num_pick_locs_warehouse", len(pick_ids))
    if len(pick_ids) != expected:
        raise ValueError(
            f"Expected {expected} pick locations, got {len(pick_ids)}"
        )
    return pick_ids


def _build_warehouse(parent: dict) -> Tuple[Warehouse, List[str]]:
    """Build Warehouse from parent layout JSON.

    Returns (warehouse, storage_location_ids).
    """
    coords = parent["LOCATION_COORD_SECTION"]
    depots = [str(d) for d in parent["DEPOTS"]]
    storage_ids = get_storage_location_ids(parent)

    # All relevant location IDs: storage + depots
    relevant_ids = storage_ids + depots
    start_id, end_id = depots[0], depots[1]

    # Compute pairwise Manhattan distances
    distance_map: Dict[Tuple[str, str], float] = {}
    for i, a in enumerate(relevant_ids):
        ax, ay = coords[a]
        for b in relevant_ids[i + 1 :]:
            bx, by = coords[b]
            dist = abs(ax - bx) + abs(ay - by)
            distance_map[(a, b)] = float(dist)
            distance_map[(b, a)] = float(dist)

    w = Warehouse()
    w.set_distances_bulk(distance_map)
    w.set_start_point(start_id)
    w.set_end_point(end_id)

    return w, storage_ids


def load_l17_instance(layout_dir: str, instance_name: str) -> dict:
    """Load a single L17_533 instance.

    Args:
        layout_dir: Path to a layout directory (e.g. ``L17_533/Conventional``)
        instance_name: Instance folder name (e.g. ``c10_8502``)

    Returns:
        Dictionary with keys:
        - warehouse, order_book, item_locations, solution_item_locations
        - skus_to_slot, items, storage_locations, best_objective
    """
    # Load parent layout
    parent_path = os.path.join(layout_dir, "tsplib_parent.json")
    with open(parent_path) as f:
        parent = json.load(f)

    warehouse, storage_ids = _build_warehouse(parent)

    # Load instance
    inst_dir = os.path.join(layout_dir, "instances", instance_name)
    inst_path = os.path.join(inst_dir, f"{instance_name}.json")
    with open(inst_path) as f:
        instance = json.load(f)

    # Load solution
    sol_path = os.path.join(inst_dir, f"{instance_name}_sol.json")
    with open(sol_path) as f:
        solution = json.load(f)

    # Build OrderBook: flatten ORDERS into one record per SKU pick
    time_avail = instance.get("TIME_AVAIL_SECTION", {})
    records = []
    for order_id, skus in instance["ORDERS"].items():
        # Use TIME_AVAIL_SECTION as integer timestep -> datetime
        ts_val = int(time_avail.get(order_id, order_id))
        ts = datetime(2024, 1, 1, 0, 0, ts_val)
        for sku in skus:
            records.append({
                "order_id": str(order_id),
                "item_id": str(sku),
                "timestamp": ts,
            })
    order_book = OrderBook.from_dicts_direct(records)

    # Build ItemLocations from VISIT_LOCATION_SECTION (non-null only)
    vls = instance.get("VISIT_LOCATION_SECTION", {})
    il_records = [
        {"item_id": str(sku), "location_id": str(loc)}
        for sku, loc in vls.items()
        if loc is not None
    ]
    item_locations = ItemLocations.from_records(il_records)

    # Build solution ItemLocations
    sol_records = [
        {"item_id": str(sku), "location_id": str(loc)}
        for sku, loc in solution.items()
    ]
    solution_item_locations = ItemLocations.from_records(sol_records)

    # Extract metadata
    skus_to_slot = [str(s) for s in instance.get("SKUS_TO_SLOT", [])]
    all_skus = sorted(set(vls.keys()) | set(skus_to_slot), key=lambda x: int(x))
    best_obj_str = (
        instance.get("HEADER", {})
        .get("COMMENTS", {})
        .get("Best known objective", "0")
    )
    best_objective = float(best_obj_str)

    return {
        "warehouse": warehouse,
        "order_book": order_book,
        "item_locations": item_locations,
        "solution_item_locations": solution_item_locations,
        "skus_to_slot": skus_to_slot,
        "items": all_skus,
        "storage_locations": storage_ids,
        "best_objective": best_objective,
    }


def load_all_instances(layout_dir: str) -> List[dict]:
    """Load all instances from a layout directory.

    Returns list of dicts (same format as ``load_l17_instance``).
    """
    instances_dir = os.path.join(layout_dir, "instances")
    names = sorted(
        d
        for d in os.listdir(instances_dir)
        if os.path.isdir(os.path.join(instances_dir, d))
    )
    return [load_l17_instance(layout_dir, name) for name in names]
