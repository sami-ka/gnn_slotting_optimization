import pytest

from slotting_optimization.item_locations import ItemLocations


def test_duplicate_location_raises():
    records = [
        {"item_id": "sku1", "location_id": "A"},
        {"item_id": "sku2", "location_id": "A"},
    ]
    with pytest.raises(ValueError):
        ItemLocations.from_records(records)
