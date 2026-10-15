"""Pruebas de features.py: ausencia de fuga temporal y construcción de etiquetas."""

import numpy as np
import pandas as pd
import pytest

from mibici import config
from mibici.features import (
    LeakageError,
    add_calendar_features,
    add_lag_features,
    add_same_hour_mean,
    assign_split,
    build_p2_target,
    estimate_occupancy,
    hourly_station_panel,
)


def _trips():
    t0 = pd.Timestamp("2025-04-07 05:00")
    rows = []
    for d in range(10):
        for h in (2, 3, 12):  # 07:xx, 08:xx y 17:xx
            s = t0 + pd.Timedelta(days=d, hours=h, minutes=10)
            rows.append((1, 2, s, s + pd.Timedelta(minutes=15)))
    return pd.DataFrame(rows, columns=["origin_station_id", "destination_station_id", "start_time", "end_time"])


def test_panel_is_contiguous_and_counts_match():
    trips = _trips()
    panel = hourly_station_panel(trips, "2025-04-07", "2025-04-16")
    assert panel["departures"].sum() == len(trips)
    assert panel["arrivals"].sum() == len(trips)
    for _, g in panel.groupby("station_id"):
        assert (g["ts"].diff().dropna() == pd.Timedelta(hours=1)).all()


def test_lag_smaller_than_horizon_raises():
    panel = hourly_station_panel(_trips(), "2025-04-07", "2025-04-16")
    with pytest.raises(LeakageError):
        add_lag_features(panel, "departures", [1, 24], min_lag=config.P1_HORIZON_HOURS)
    with pytest.raises(LeakageError):
        add_same_hour_mean(panel, "departures", 7, 12, config.P1_HORIZON_HOURS, "x")


def test_features_do_not_change_when_the_target_changes():
    """Si se altera y(t), ninguna variable de la fila t cambia (no hay fuga)."""
    panel = hourly_station_panel(_trips(), "2025-04-07", "2025-04-16")
    a = add_lag_features(panel, "departures", [24, 168], min_lag=24)
    i = a.index[a["departures"] > 0][-1]
    modified = panel.copy()
    modified.loc[i, "departures"] += 50
    b = add_lag_features(modified, "departures", [24, 168], min_lag=24)
    cols = ["departures_lag_24", "departures_lag_168"]
    pd.testing.assert_series_equal(a.loc[i, cols], b.loc[i, cols])


def test_lag_does_not_cross_stations():
    panel = hourly_station_panel(_trips(), "2025-04-07", "2025-04-16")
    out = add_lag_features(panel, "departures", [24], min_lag=24)
    first_rows = out.groupby("station_id").head(24)
    assert first_rows["departures_lag_24"].isna().all()


def test_operating_day_starts_at_service_opening():
    df = pd.DataFrame({"ts": pd.to_datetime(["2025-04-07 04:00", "2025-04-07 05:00", "2025-04-08 00:00"])})
    out = add_calendar_features(df)
    assert out["op_date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-04-06", "2025-04-07", "2025-04-07"]
    assert out["op_hour"].tolist() == [23, 0, 19]


def test_occupancy_and_risk_label():
    ts = pd.date_range("2025-04-07 05:00", periods=6, freq="h")
    df = pd.DataFrame(
        {
            "station_id": 1,
            "ts": ts,
            "departures": [0, 4, 4, 0, 0, 0],
            "arrivals": [0, 0, 0, 0, 0, 0],
        }
    )
    df = add_calendar_features(df)
    out = estimate_occupancy(df, pd.Series(10, index=df.index), initial_fill=0.5)
    # Acotada a [0, capacidad]: no puede haber bicicletas negativas.
    assert out["est_bikes"].tolist() == [5, 1, 0, 0, 0, 0]
    assert out["net_cum_today"].tolist() == [0, -4, -8, -8, -8, -8]
    lab = build_p2_target(out, horizon=1, low=0.1, high=0.9)
    # En t=0 la siguiente hora deja 1 bici (10 %): riesgo de vaciado.
    assert lab["y"].tolist()[:2] == [1, 1]
    assert lab["risk_type"].iloc[0] == "vaciado"
    assert lab["y"].iloc[-1] == 0  # sin horas futuras en el día: no se inventa riesgo


def test_assign_split_uses_config_boundaries():
    s = assign_split(
        pd.Series(pd.to_datetime(["2026-03-31 23:00", "2026-04-01 00:00", "2026-07-01 05:00"]))
    )
    assert s.tolist() == ["train", "val", "test"]
