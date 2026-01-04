from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict


@dataclass
class Order:
    """Represents a single order with minimal fields.

    Fields:
        order_id: str
        item_id: str
        timestamp: datetime
    """

    order_id: str
    item_id: str
    timestamp: datetime

    @staticmethod
    def parse_timestamp(value: Any) -> datetime:
        """Parse timestamp from ISO string, epoch (int/float), or datetime."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value))
        if isinstance(value, str):
            # Try ISO 8601 parse
            return datetime.fromisoformat(value)
        raise TypeError("Unsupported timestamp type")

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Order":
        return cls(
            order_id=str(obj["order_id"]),
            item_id=str(obj["item_id"]),
            timestamp=cls.parse_timestamp(obj["timestamp"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": str(self.order_id),
            "item_id": str(self.item_id),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ItemLocation:
    """Simple mapping of an item to a location id."""

    item_id: str
    location_id: str

    def to_dict(self) -> Dict[str, str]:
        return {"item_id": str(self.item_id), "location_id": str(self.location_id)}
