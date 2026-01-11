from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple


class Warehouse:
    """Stores discrete locations and a start/end point for order preparation.

    This is a light-weight placeholder for now: we keep a list of location ids and
    an optional distance mapping between pairs of locations.
    """

    def __init__(self, locations: Optional[Iterable[str]] = None, start_point_id: Optional[str] = None, end_point_id: Optional[str] = None):
        self._locations: List[str] = list(locations or [])
        self._start_point_id: Optional[str] = start_point_id
        self._end_point_id: Optional[str] = end_point_id
        # Optional mapping (from_id, to_id) -> distance (float)
        self._distance_map: Dict[Tuple[str, str], float] = {}

    def add_location(self, location_id: str) -> None:
        if location_id not in self._locations:
            self._locations.append(location_id)

    def set_start_point(self, start_point_id: str) -> None:
        if start_point_id not in self._locations:
            self.add_location(start_point_id)
        self._start_point_id = start_point_id

    @property
    def start_point(self) -> Optional[str]:
        return self._start_point_id

    def set_end_point(self, end_point_id: str) -> None:
        if end_point_id not in self._locations:
            self.add_location(end_point_id)
        self._end_point_id = end_point_id

    @property
    def end_point(self) -> Optional[str]:
        return self._end_point_id

    def locations(self) -> List[str]:
        return list(self._locations)

    def location_exists(self, location_id: str) -> bool:
        return location_id in self._locations

    def set_distance(self, from_id: str, to_id: str, distance: float) -> None:
        if from_id not in self._locations:
            self.add_location(from_id)
        if to_id not in self._locations:
            self.add_location(to_id)
        self._distance_map[(from_id, to_id)] = float(distance)

    def get_distance(self, from_id: str, to_id: str) -> Optional[float]:
        return self._distance_map.get((from_id, to_id))

    def set_distances_bulk(self, distance_map: Dict[Tuple[str, str], float]) -> None:
        """Efficiently set multiple distances at once.

        Significantly faster than calling set_distance() in a loop for large
        distance maps, as it performs bulk location validation.

        Args:
            distance_map: Dictionary mapping (from_id, to_id) tuples to distances
        """
        # Extract all unique location IDs from distance map
        unique_locs = set()
        for (from_id, to_id) in distance_map.keys():
            unique_locs.add(from_id)
            unique_locs.add(to_id)

        # Add new locations (deterministic ordering with sorted)
        current_locs = set(self._locations)
        new_locs = sorted(unique_locs - current_locs)
        self._locations.extend(new_locs)

        # Bulk update distance map
        for (from_id, to_id), distance in distance_map.items():
            self._distance_map[(from_id, to_id)] = float(distance)

    def __repr__(self) -> str:
        return f"Warehouse(n_locations={len(self._locations)}, start={self._start_point_id})"
