import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk.thresholds import (  # noqa: E402
    DEFAULT_BASE,
    load,
    resolve_threshold,
    row_thresholds,
)

TABLE = {
    "region": {"Bamako": 0.25, "Koulikoro": 0.37, "Gao": 0.73},
    "wealth_quartile": {"1": 0.46, "4": 0.65},
}


def test_load_missing_file_returns_empty():
    assert load("/nonexistent/group_thresholds.json") == {}


def test_load_corrupt_file_returns_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert load(str(p)) == {}


def test_load_roundtrip(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(TABLE))
    assert load(str(p)) == TABLE


def test_pooled_strategy_ignores_table():
    assert resolve_threshold(region="Bamako", table=TABLE, strategy="pooled") == DEFAULT_BASE
    assert resolve_threshold(region="Bamako", table={}, strategy="pooled") == DEFAULT_BASE


def test_missing_group_falls_back_to_base():
    assert resolve_threshold(region="Timbuktu", wealth_quartile=3, table=TABLE, strategy="group") == DEFAULT_BASE


def test_region_and_wealth_resolve():
    assert resolve_threshold(region="Bamako", wealth_quartile=1, table=TABLE, strategy="group") == pytest.approx(0.25)
    assert resolve_threshold(region="Gao", wealth_quartile=4, table=TABLE, strategy="group") == pytest.approx(0.65)


def test_min_rule_across_groups():
    # Bamako (0.25) wins over wealth q1 (0.46); Gao (0.73) loses to wealth q4 (0.65)
    assert resolve_threshold(region="Bamako", wealth_quartile=1, table=TABLE, strategy="group") == pytest.approx(0.25)
    assert resolve_threshold(region="Gao", wealth_quartile=4, table=TABLE, strategy="group") == pytest.approx(0.65)


def test_bounds_clip():
    t = {"region": {"X": 0.01}, "wealth_quartile": {"1": 0.99}}
    assert resolve_threshold(region="X", table=t, strategy="group", bounds=(0.02, 0.95)) == pytest.approx(0.02)
    assert resolve_threshold(wealth_quartile=1, table=t, strategy="group", bounds=(0.02, 0.95)) == pytest.approx(0.95)


def test_row_thresholds_vectorized():
    df = __import__("pandas").DataFrame(
        {"region": ["Bamako", "Gao", "Timbuktu"], "wealth_quartile": [1, 4, 2], "rural_flag": [1, 0, 0]}
    )
    out = row_thresholds(df, TABLE, strategy="group")
    assert np.allclose(out, [0.25, 0.65, DEFAULT_BASE])


def test_rural_in_table_participates_in_min():
    t = {**TABLE, "rural_flag": {"0": 0.30}}
    # Gao + wealth q4 -> 0.65, but rural 0 lowers it to 0.30
    assert resolve_threshold(region="Gao", wealth_quartile=4, rural_flag=0, table=t, strategy="group") == pytest.approx(0.30)


def test_empty_table_group_returns_base():
    assert resolve_threshold(region="Bamako", table={}, strategy="group") == DEFAULT_BASE


def test_loaded_shipped_table_resolves():
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "group_thresholds.json")
    table = load(path)
    assert table, "shipped config/group_thresholds.json not found"
    assert resolve_threshold(region="Koulikoro", wealth_quartile=1, table=table, strategy="group") == pytest.approx(0.37)
    assert resolve_threshold(region="Gao", wealth_quartile=4, table=table, strategy="group") == pytest.approx(0.65)
    assert resolve_threshold(region="Nowhere", wealth_quartile=2, table=table, strategy="group") == pytest.approx(0.48)