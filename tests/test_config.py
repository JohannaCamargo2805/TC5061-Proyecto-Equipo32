"""Pruebas de la configuración central (config.py)."""

import pandas as pd

from mibici import config


def _months(start, end):
    return [f"{p.year:04d}_{p.month:02d}" for p in pd.period_range(start, end, freq="M")]


def test_splits_are_ordered_disjoint_and_cover_the_date_range():
    starts = [pd.Timestamp(a) for a, _ in config.SPLITS.values()]
    ends = [pd.Timestamp(b) for _, b in config.SPLITS.values()]
    assert starts == sorted(starts)
    assert all(e < s for e, s in zip(ends[:-1], starts[1:]))  # sin traslape
    assert starts[0] == pd.Timestamp(config.DATE_START)
    assert ends[-1] == pd.Timestamp(config.DATE_END)
    gaps = [(s - e).days for e, s in zip(ends[:-1], starts[1:])]
    assert gaps == [1, 1]  # contiguos: ningún día queda fuera


def test_every_month_in_range_has_a_pinned_url():
    months = _months(config.DATE_START, config.DATE_END)
    assert len(months) == 24
    assert set(months) <= set(config.TRIP_URLS)


def test_experiment_name_includes_team_number():
    assert config.MLFLOW_EXPERIMENT_NAME == f"mibici-equipo{config.TEAM_NUMBER}"


def test_risk_thresholds_are_consistent():
    assert 0 <= config.P2_LOW_FILL < config.P2_INITIAL_FILL < config.P2_HIGH_FILL <= 1
    assert config.P2_HORIZON_HOURS < config.SERVICE_HOURS
