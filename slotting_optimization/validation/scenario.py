from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..order_book import OrderBook
from ..warehouse import Warehouse
from ..item_locations import ItemLocations
from ..simulator import Simulator


@dataclass
class ValidationScenario:
    """Container for a validation scenario with known optimal assignment."""

    name: str
    order_book: OrderBook
    warehouse: Warehouse
    initial_assignment: ItemLocations  # Random starting point
    optimal_assignment: Dict[str, str]  # item_id -> location_id
    optimal_distance: float
    items: List[str]
    storage_locations: List[str]

    def compute_distance(self, assignment: ItemLocations) -> float:
        """Compute travel distance for any assignment.

        Args:
            assignment: ItemLocations to evaluate

        Returns:
            Total travel distance
        """
        simulator = Simulator()
        total, _ = simulator.simulate(self.order_book, self.warehouse, assignment)
        return total

    def compute_gap(self, assignment: ItemLocations) -> float:
        """Compute optimality gap as percentage.

        Args:
            assignment: ItemLocations to evaluate

        Returns:
            Gap as percentage: (actual - optimal) / optimal * 100
        """
        actual = self.compute_distance(assignment)
        return (actual - self.optimal_distance) / self.optimal_distance * 100

    def assignment_accuracy(self, assignment_dict: Dict[str, str]) -> float:
        """Compute percentage of items in optimal positions.

        Args:
            assignment_dict: Dictionary mapping item_id -> location_id

        Returns:
            Percentage of items correctly assigned (0-100)
        """
        correct = sum(
            1 for item in self.items
            if assignment_dict.get(item) == self.optimal_assignment.get(item)
        )
        return (correct / len(self.items)) * 100 if self.items else 0.0
