"""Tests for L17_533 instance loader."""

import os
import pytest

from slotting_optimization.instance_loader import (
    load_l17_instance,
    load_all_instances,
    get_storage_location_ids,
)
from slotting_optimization.simulator import Simulator
from slotting_optimization.order_book import OrderBook
from slotting_optimization.item_locations import ItemLocations
from slotting_optimization.warehouse import Warehouse


LAYOUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "L17_533",
    "Conventional",
)

INSTANCE_NAME = "c10_8502"


@pytest.fixture
def instance():
    return load_l17_instance(LAYOUT_DIR, INSTANCE_NAME)


def test_load_returns_correct_types(instance):
    assert isinstance(instance["warehouse"], Warehouse)
    assert isinstance(instance["order_book"], OrderBook)
    assert isinstance(instance["item_locations"], ItemLocations)
    assert isinstance(instance["solution_item_locations"], ItemLocations)
    assert isinstance(instance["skus_to_slot"], list)
    assert isinstance(instance["items"], list)
    assert isinstance(instance["storage_locations"], list)


def test_storage_locations_count(instance):
    assert len(instance["storage_locations"]) == 220


def test_order_count_matches_num_visits(instance):
    # c10_8502 has NUM_VISITS=10
    assert len(instance["order_book"]) == 10


def test_solution_assigns_all_skus(instance):
    sol = instance["solution_item_locations"]
    items = instance["items"]
    sol_dict = sol.to_dict()
    for item in items:
        assert item in sol_dict, f"SKU {item} missing from solution"


def test_skus_to_slot(instance):
    assert instance["skus_to_slot"] == ["6"]


def test_item_locations_excludes_null(instance):
    il = instance["item_locations"]
    # SKU 6 has null location, so item_locations should have 9 entries
    assert len(il) == 9
    assert il.get_location("6") is None


def test_simulate_with_solution(instance):
    """Simulate with solution assignment and verify valid distance."""
    sim = Simulator()
    sol_il = instance["solution_item_locations"]
    ob = instance["order_book"]
    w = instance["warehouse"]
    distance, _ = sim.simulate(ob, w, sol_il)
    assert distance > 0


def test_best_objective_is_positive(instance):
    """Best known objective should be a positive number."""
    assert instance["best_objective"] > 0
    assert instance["best_objective"] == 234.0


def test_warehouse_has_start_end(instance):
    w = instance["warehouse"]
    assert w.start_point == "0"
    assert w.end_point == "1"


def test_warehouse_distances_symmetric(instance):
    w = instance["warehouse"]
    locs = instance["storage_locations"][:5]
    for a in locs:
        for b in locs:
            if a != b:
                d1 = w.get_distance(a, b)
                d2 = w.get_distance(b, a)
                assert d1 is not None
                assert d1 == d2


def test_get_storage_location_ids():
    import json

    with open(os.path.join(LAYOUT_DIR, "tsplib_parent.json")) as f:
        parent = json.load(f)
    ids = get_storage_location_ids(parent)
    assert len(ids) == 220
    # Should not contain depots
    assert "0" not in ids
    assert "1" not in ids
