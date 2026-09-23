import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.routing.optimizer import min_cost_center, travel_metrics  # noqa: E402

WEIGHTS = {"travel_km": 0.2, "travel_h": 40.0, "out_of_pocket": 0.05, "center_load": 0.02}
BAMAKO = {"lat": 12.639, "lon": -8.003}


def test_travel_metrics_same_point_is_zero():
    d, h = travel_metrics(BAMAKO, dict(BAMAKO))
    assert d < 1e-6
    assert h < 1e-6


def test_travel_metrics_without_network_uses_detour():
    dist, hours = travel_metrics(BAMAKO, {"lat": 12.85, "lon": -8.00})
    straight = 23.6  # approx haversine km of that displacement
    assert dist > straight >= 0
    assert hours > 0


def test_min_cost_prefers_nearest_capable():
    centers = [
        {"name": "near", "lat": 12.65, "lon": -8.01, "out_of_pocket": 5000, "load": 0.5},
        {"name": "far", "lat": 15.0, "lon": -9.0, "out_of_pocket": 100, "load": 0.5},
    ]
    best = min_cost_center(BAMAKO, centers, WEIGHTS, max_travel_ratio=1.5)
    assert best["name"] == "near"
    assert "distance_km" in best and "travel_h" in best and "_cost" in best


def test_out_of_pocket_includes_transport():
    centers = [{"name": "a", "lat": 12.70, "lon": -8.05, "out_of_pocket": 0, "load": 0}]
    best = min_cost_center(BAMAKO, centers, WEIGHTS, max_travel_ratio=None)
    assert best["out_of_pocket"] > 0  # base + per-km transport


def test_empty_candidates_returns_none():
    assert min_cost_center(BAMAKO, [], WEIGHTS) is None