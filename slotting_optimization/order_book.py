from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
from datetime import datetime

import polars as pl

from .models import Order


class OrderBook:
    """Stores orders in a Polars DataFrame and provides basic operations."""

    def __init__(self, df: Optional[pl.DataFrame] = None):
        if df is None:
            self._df = pl.DataFrame(
                {
                    "order_id": pl.Series(dtype=pl.Utf8),
                    "item_id": pl.Series(dtype=pl.Utf8),
                    "timestamp": pl.Series(dtype=pl.Datetime),
                }
            )
        else:
            # Ensure expected dtypes
            self._df = df.with_columns(
                [
                    pl.col("order_id").cast(pl.Utf8),
                    pl.col("item_id").cast(pl.Utf8),
                    pl.col("timestamp").cast(pl.Datetime),
                ]
            )

    @classmethod
    def from_orders(cls, orders: Iterable[Order]) -> "OrderBook":
        records = [o.to_dict() for o in orders]
        if not records:
            return cls()
        df = pl.from_dicts(records)
        # Parse ISO datetimes
        df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S", strict=False))
        return cls(df)

    @classmethod
    def from_dicts_direct(cls, records: List[Dict[str, Any]]) -> "OrderBook":
        """Create OrderBook directly from dict records with datetime objects.

        More efficient than from_orders() when you have datetime objects,
        as it avoids string conversion overhead.

        Args:
            records: List of dicts with keys: order_id (str), item_id (str),
                     timestamp (datetime)

        Returns:
            OrderBook instance
        """
        if not records:
            return cls()

        df = pl.from_dicts(records)
        df = df.with_columns([
            pl.col("order_id").cast(pl.Utf8),
            pl.col("item_id").cast(pl.Utf8),
            pl.col("timestamp").cast(pl.Datetime),
        ])

        return cls(df)

    def add(self, order: Order) -> None:
        rec = order.to_dict()
        new = pl.DataFrame([rec])
        new = new.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S", strict=False))
        self._df = pl.concat([self._df, new], how="vertical")

    @classmethod
    def load_csv(cls, path: str) -> "OrderBook":
        df = pl.read_csv(path, try_parse_dates=True)
        # Ensure timestamp is Datetime
        if df.schema.get("timestamp") != pl.Datetime:
            # Attempt to parse ISO strings
            if df["timestamp"].dtype == pl.Utf8:
                df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S", strict=False))
            else:
                df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))
        return cls(df)

    def save_csv(self, path: str) -> None:
        # Save timestamps as ISO strings
        out = self._df.with_columns(pl.col("timestamp").dt.strftime("%Y-%m-%dT%H:%M:%S"))
        out.write_csv(path)

    def filter_by_time(self, start: datetime, end: datetime) -> pl.DataFrame:
        return self._df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))

    def to_df(self) -> pl.DataFrame:
        return self._df

    def __len__(self) -> int:
        return self._df.shape[0]

    def __repr__(self) -> str:
        return f"OrderBook(n_orders={len(self)})"
