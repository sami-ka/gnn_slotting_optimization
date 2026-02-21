from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import numpy as np

from .scenario import ValidationScenario
from ..item_locations import ItemLocations


@dataclass
class ConvergencePoint:
    """Single point in convergence history."""

    epoch: int
    phase: str  # "phase1" or "phase2"
    actual_distance: float
    optimal_distance: float
    optimality_gap_pct: float
    assignment_accuracy_pct: float


class ConvergenceTracker:
    """Track convergence during training."""

    def __init__(self, scenario: ValidationScenario):
        """Initialize tracker.

        Args:
            scenario: The validation scenario to track
        """
        self.scenario = scenario
        self.history: List[ConvergencePoint] = []

    def record(
        self,
        epoch: int,
        phase: str,
        optimized_assignment: np.ndarray,
    ) -> ConvergencePoint:
        """Record a convergence measurement.

        Args:
            epoch: Current training epoch
            phase: Training phase ("phase1" or "phase2")
            optimized_assignment: Assignment as numpy array where
                                 optimized_assignment[i] = location_index for item i

        Returns:
            ConvergencePoint with computed metrics
        """
        # Convert array assignment to ItemLocations
        assignment_dict = {}
        for i, item in enumerate(self.scenario.items):
            loc_idx = int(optimized_assignment[i])
            assignment_dict[item] = self.scenario.storage_locations[loc_idx]

        assignment = ItemLocations.from_records([
            {"item_id": item, "location_id": loc}
            for item, loc in assignment_dict.items()
        ])

        # Compute metrics
        actual_distance = self.scenario.compute_distance(assignment)
        optimality_gap_pct = (actual_distance - self.scenario.optimal_distance) / self.scenario.optimal_distance * 100
        assignment_accuracy_pct = self.scenario.assignment_accuracy(assignment_dict)

        point = ConvergencePoint(
            epoch=epoch,
            phase=phase,
            actual_distance=actual_distance,
            optimal_distance=self.scenario.optimal_distance,
            optimality_gap_pct=optimality_gap_pct,
            assignment_accuracy_pct=assignment_accuracy_pct,
        )

        self.history.append(point)
        return point

    def get_history(self) -> List[ConvergencePoint]:
        """Get full convergence history.

        Returns:
            List of all recorded ConvergencePoints
        """
        return self.history.copy()
