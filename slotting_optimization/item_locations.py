from __future__ import annotations

from typing import Dict, Iterable, Optional

import polars as pl

from .models import ItemLocation


class ItemLocations:
    """Manages mapping from item_id to location_id."""

    def __init__(self, mapping: Optional[Dict[str, str]] = None):
        self._mapping: Dict[str, str] = dict(mapping or {})

    @classmethod
    def from_records(cls, records: Iterable[Dict[str, str]]) -> "ItemLocations":
        mapping: Dict[str, str] = {}
        loc_to_item: Dict[str, str] = {}
        for r in records:
            item = str(r["item_id"])
            loc = str(r["location_id"])
            # Allow repeated records for same item->same-loc, but disallow multiple items at same location
            existing = loc_to_item.get(loc)
            if existing is not None and existing != item:
                raise ValueError(f"Location '{loc}' already assigned to item '{existing}'; cannot assign to '{item}'")
            mapping[item] = loc
            loc_to_item[loc] = item
        return cls(mapping)

    @classmethod
    def load_csv(cls, path: str) -> "ItemLocations":
        df = pl.read_csv(path)
        if "item_id" not in df.columns or "location_id" not in df.columns:
            raise ValueError("CSV must have columns 'item_id' and 'location_id'")
        records = df.select(["item_id", "location_id"]).to_dicts()
        return cls.from_records(records)

    def save_csv(self, path: str) -> None:
        df = pl.DataFrame(
            [{"item_id": k, "location_id": v} for k, v in self._mapping.items()]
        )
        df.write_csv(path)

    def get_location(self, item_id: str) -> Optional[str]:
        return self._mapping.get(str(item_id))

    def to_dict(self) -> Dict[str, str]:
        return dict(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def __repr__(self) -> str:
        return f"ItemLocations(n_items={len(self)})"
