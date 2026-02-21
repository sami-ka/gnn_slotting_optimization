"""Tests for validation framework."""

from slotting_optimization.validation import (
    compute_brute_force_optimal,
    generate_frequency_optimal_scenario,
    ValidationScenario,
)
from slotting_optimization.item_locations import ItemLocations


def test_brute_force_finds_optimal_small():
    """Verify brute force works for small 4-item problem."""
    # Generate a small scenario
    order_book, warehouse, _, items, storage_locations, expected_optimal, expected_dist = (
        generate_frequency_optimal_scenario(n_items=4, n_locations=4, n_orders=50, seed=42)
    )

    # Compute brute force optimal
    bf_optimal, bf_dist = compute_brute_force_optimal(
        order_book, warehouse, items, storage_locations
    )

    # The brute force should find at least as good or better solution
    assert bf_dist <= expected_dist + 1e-6, f"Brute force found worse solution: {bf_dist} > {expected_dist}"


def test_frequency_scenario_optimal_verified():
    """Cross-validate analytical optimal against brute force for 5-item problem."""
    # Generate scenario
    order_book, warehouse, _, items, storage_locations, optimal_assignment, optimal_distance = (
        generate_frequency_optimal_scenario(n_items=5, n_locations=5, n_orders=100, seed=42)
    )

    # Verify with brute force
    bf_optimal, bf_dist = compute_brute_force_optimal(
        order_book, warehouse, items, storage_locations
    )

    # The analytical optimal should match brute force (within small tolerance)
    assert abs(bf_dist - optimal_distance) < 1e-6, (
        f"Analytical optimal ({optimal_distance:.2f}) != brute force ({bf_dist:.2f})"
    )


def test_optimal_assignment_has_zero_gap():
    """Verify gap=0 when using the optimal assignment."""
    # Generate scenario
    order_book, warehouse, _, items, storage_locations, optimal_assignment, optimal_distance = (
        generate_frequency_optimal_scenario(n_items=5, n_locations=5, n_orders=100, seed=42)
    )

    # Create scenario object
    random_initial = ItemLocations.from_records([
        {"item_id": item, "location_id": loc}
        for item, loc in zip(items, storage_locations)  # Random assignment
    ])

    scenario = ValidationScenario(
        name="test",
        order_book=order_book,
        warehouse=warehouse,
        initial_assignment=random_initial,
        optimal_assignment=optimal_assignment,
        optimal_distance=optimal_distance,
        items=items,
        storage_locations=storage_locations,
    )

    # Create optimal ItemLocations
    optimal_il = ItemLocations.from_records([
        {"item_id": item, "location_id": loc}
        for item, loc in optimal_assignment.items()
    ])

    # Gap should be 0
    gap = scenario.compute_gap(optimal_il)
    assert abs(gap) < 1e-6, f"Optimal assignment should have 0% gap, got {gap:.4f}%"

    # Accuracy should be 100%
    accuracy = scenario.assignment_accuracy(optimal_assignment)
    assert abs(accuracy - 100.0) < 1e-6, f"Optimal assignment should have 100% accuracy, got {accuracy:.2f}%"


def test_assignment_accuracy_computation():
    """Test assignment accuracy calculation."""
    # Generate scenario
    order_book, warehouse, _, items, storage_locations, optimal_assignment, optimal_distance = (
        generate_frequency_optimal_scenario(n_items=5, n_locations=5, n_orders=100, seed=42)
    )

    random_initial = ItemLocations.from_records([
        {"item_id": item, "location_id": loc}
        for item, loc in zip(items, storage_locations)
    ])

    scenario = ValidationScenario(
        name="test",
        order_book=order_book,
        warehouse=warehouse,
        initial_assignment=random_initial,
        optimal_assignment=optimal_assignment,
        optimal_distance=optimal_distance,
        items=items,
        storage_locations=storage_locations,
    )

    # Test partial correctness
    # Create assignment with 3 out of 5 correct
    partial_assignment = {}
    for i, item in enumerate(items):
        if i < 3:
            partial_assignment[item] = optimal_assignment[item]  # Correct
        else:
            # Wrong assignment
            wrong_idx = (i + 1) % len(storage_locations)
            partial_assignment[item] = storage_locations[wrong_idx]

    accuracy = scenario.assignment_accuracy(partial_assignment)
    assert abs(accuracy - 60.0) < 1e-6, f"Expected 60% accuracy (3/5), got {accuracy:.2f}%"
