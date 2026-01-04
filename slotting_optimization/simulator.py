from __future__ import annotations

from typing import List, Tuple

import polars as pl

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
        df: pl.DataFrame = order_book.to_df()
        # Ensure sorted by timestamp
        df = df.sort("timestamp")
        total_distance = 0.0
        per_order: List[float] = []

        start = warehouse.start_point
        end = warehouse.end_point
        if start is None or end is None:
            raise ValueError("Warehouse must define start_point and end_point")

        rows = df.to_dicts()
        for idx, r in enumerate(rows):
            item_id = r["item_id"]
            loc = item_locations.get_location(item_id)
            if loc is None:
                raise ValueError(f"No location for item {item_id}")

            # start -> loc
            d1 = warehouse.get_distance(start, loc)
            if d1 is None:
                raise ValueError(f"Missing distance from {start} to {loc}")
            # loc -> end
            d2 = warehouse.get_distance(loc, end)
            if d2 is None:
                raise ValueError(f"Missing distance from {loc} to {end}")

            order_dist = float(d1) + float(d2)
            total_distance += order_dist
            per_order.append(order_dist)

            # If there is another order, add end -> start travel
            if idx != len(rows) - 1:
                ret = warehouse.get_distance(end, start)
                if ret is None:
                    raise ValueError(f"Missing distance from {end} to {start}")
                total_distance += float(ret)

        return total_distance, per_order
