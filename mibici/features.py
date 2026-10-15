"""Feature engineering reutilizable para las tres preguntas del proyecto.

Principio rector: **ninguna variable usa información posterior al momento de la
predicción**. Cada función documenta en qué momento se considera emitida la
predicción y por qué la variable es pertinente para la pregunta que atiende.

* **P1 (regresión)**: viajes iniciados por estación y hora. La predicción se
  emite al cierre del día operativo D-1 (01:00) para todas las horas de servicio
  del día D, por lo que solo se usan rezagos de 24 h o más.
* **P2 (clasificación)**: riesgo de que una estación quede sin bicicletas o sin
  anclajes en las próximas ``P2_HORIZON_HOURS`` horas. La decisión se toma al
  cierre de la hora *t* con la información observada hasta *t*.
* **P3 (clustering)**: perfil de uso de cada estación, calculado solo con el
  periodo de entrenamiento.

Comando para construir las tablas analíticas en ``data/processed``::

    python -m mibici.features build
"""

from __future__ import annotations

import warnings

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici import config

app = typer.Typer(help="Construcción de variables para P1, P2 y P3.")


@app.callback()
def _cli() -> None:
    """Construcción de variables para P1, P2 y P3 (subcomando: build)."""


class LeakageError(ValueError):
    """Se lanza cuando una variable usaría información del futuro."""


# ---------------------------------------------------------------------------
# Carga de datos normalizados (salida de dataset.py)
# ---------------------------------------------------------------------------
def load_trips(start: str = config.DATE_START, end: str = config.DATE_END) -> pd.DataFrame:
    """Carga los viajes limpios (Parquet mensual) cuyo inicio cae en [start, end]."""
    from mibici.dataset import month_keys

    paths = [config.INTERIM_TRIPS_DIR / f"trips_{m}.parquet" for m in month_keys(start, end)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Faltan {missing}. Ejecuta: python -m mibici.dataset ingest")
    trips = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    end_excl = pd.Timestamp(end) + pd.Timedelta(days=1)
    mask = (trips["start_time"] >= pd.Timestamp(start)) & (trips["start_time"] < end_excl)
    return trips.loc[mask].reset_index(drop=True)


def load_stations() -> pd.DataFrame:
    """Carga la nomenclatura normalizada de estaciones."""
    return pd.read_parquet(config.STATIONS_INTERIM_PATH)


def filter_known_stations(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Conserva viajes cuyo origen y destino existen en la nomenclatura.

    Pertinencia: las tres preguntas necesitan atributos de la estación
    (capacidad o coordenadas); un viaje con estación desconocida no puede
    asignarse a una estación modelable.
    """
    known = set(stations["station_id"].astype(int))
    mask = trips["origin_station_id"].isin(known) & trips["destination_station_id"].isin(known)
    dropped = int((~mask).sum())
    if dropped:
        logger.info(f"Viajes con estación fuera de la nomenclatura descartados: {dropped:,}")
    return trips.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Agregación base: panel estación x hora
# ---------------------------------------------------------------------------
def hourly_station_panel(
    trips: pd.DataFrame, start: str = config.DATE_START, end: str = config.DATE_END
) -> pd.DataFrame:
    """Panel completo estación x hora con salidas (``departures``) y llegadas (``arrivals``).

    Las horas sin viajes se rellenan con 0 solo dentro del periodo en que la
    estación estuvo activa (desde su primer movimiento observado hasta el
    último), para no inventar demanda nula antes de que una estación existiera.
    El panel es contiguo por estación, lo que permite calcular rezagos con
    ``shift`` sin saltos de tiempo.

    Pertinencia: es la unidad de observación de P1 (variable objetivo =
    ``departures``) y la base del balance de bicicletas de P2.
    """
    end_excl = pd.Timestamp(end) + pd.Timedelta(days=1)
    dep = (
        trips.assign(ts=trips["start_time"].dt.floor("h"))
        .groupby(["origin_station_id", "ts"], observed=True)
        .size()
        .rename("departures")
    )
    dep.index = dep.index.set_names(["station_id", "ts"])
    arr_trips = trips.loc[trips["end_time"] < end_excl]
    arr = (
        arr_trips.assign(ts=arr_trips["end_time"].dt.floor("h"))
        .groupby(["destination_station_id", "ts"], observed=True)
        .size()
        .rename("arrivals")
    )
    arr.index = arr.index.set_names(["station_id", "ts"])
    counts = pd.concat([dep, arr], axis=1).fillna(0).reset_index()

    bounds = counts.groupby("station_id")["ts"].agg(["min", "max"])
    hours = pd.date_range(pd.Timestamp(start), end_excl - pd.Timedelta(hours=1), freq="h")
    frames = []
    for sid, (t0, t1) in bounds.iterrows():
        rng = hours[(hours >= t0) & (hours <= t1)]
        frames.append(pd.DataFrame({"station_id": np.int32(sid), "ts": rng}))
    grid = pd.concat(frames, ignore_index=True)
    grid["station_id"] = grid["station_id"].astype("int32")
    counts["station_id"] = counts["station_id"].astype("int32")
    panel = grid.merge(counts, on=["station_id", "ts"], how="left")
    panel[["departures", "arrivals"]] = panel[["departures", "arrivals"]].fillna(0).astype("int16")
    return panel.sort_values(["station_id", "ts"], kind="mergesort").reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """Agrega hora, día de la semana, fin de semana, mes, feriado y día/hora operativa.

    El día operativo inicia a las ``OPERATING_DAY_START_HOUR`` (05:00), porque
    MiBici presta servicio de 05:00 a 00:59. ``op_hour`` = horas transcurridas
    desde la apertura (0 = 05:00, 19 = 00:00).

    Pertinencia: la demanda (P1) y los desbalances (P2) siguen ciclos diarios y
    semanales muy marcados (picos de traslado al trabajo/escuela); los feriados
    rompen ese patrón. Son variables conocidas de antemano: no hay fuga.
    """
    ts = df[ts_col]
    shift = pd.Timedelta(hours=config.OPERATING_DAY_START_HOUR)
    holidays = pd.to_datetime(config.HOLIDAYS)
    out = df.copy()
    out["hour"] = ts.dt.hour.astype("int8")
    out["dow"] = ts.dt.dayofweek.astype("int8")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    out["month"] = ts.dt.month.astype("int8")
    out["is_holiday"] = ts.dt.normalize().isin(holidays).astype("int8")
    out["op_date"] = (ts - shift).dt.normalize()
    out["op_hour"] = ((out["hour"] - config.OPERATING_DAY_START_HOUR) % 24).astype("int8")
    return out


def add_lag_features(
    df: pd.DataFrame,
    col: str,
    lags: list[int],
    min_lag: int,
    group_col: str = "station_id",
    prefix: str | None = None,
) -> pd.DataFrame:
    """Agrega ``{col}_lag_{k}`` = valor de ``col`` hace ``k`` horas en la misma estación.

    Requiere un panel horario contiguo por estación (ver
    :func:`hourly_station_panel`). Lanza :class:`LeakageError` si algún rezago
    es menor que ``min_lag``: por ejemplo, en P1 la predicción se emite con 24 h
    de anticipación, así que un rezago de 1 h usaría información del futuro.

    Pertinencia: el uso reciente de la estación a la misma hora es el mejor
    predictor de su uso futuro (autocorrelación diaria y semanal).
    """
    bad = [k for k in lags if k < min_lag]
    if bad:
        raise LeakageError(f"Rezagos {bad} < {min_lag} h usarían información del futuro.")
    out = df.copy()
    grouped = out.groupby(group_col, sort=False)[col]
    name = prefix or col
    for k in lags:
        out[f"{name}_lag_{k}"] = grouped.shift(k).astype("float32")
    return out


def add_same_hour_mean(
    df: pd.DataFrame,
    col: str,
    n_periods: int,
    step_hours: int,
    min_lag: int,
    name: str,
    group_col: str = "station_id",
) -> pd.DataFrame:
    """Promedio de ``col`` a la misma hora en los ``n_periods`` periodos previos.

    Con ``step_hours=24`` es el promedio de la misma hora en los últimos
    ``n_periods`` días; con ``step_hours=168``, la misma hora y día de la
    semana en las últimas semanas. El primer rezago es ``step_hours``, que debe
    ser >= ``min_lag``.

    Pertinencia: suaviza el ruido de un solo día y captura el nivel típico de
    la estación a esa hora, lo que estabiliza la predicción de P1.
    """
    if step_hours < min_lag:
        raise LeakageError(f"El primer rezago ({step_hours} h) es menor que {min_lag} h.")
    grouped = df.groupby(group_col, sort=False)[col]
    stacked = np.column_stack(
        [grouped.shift(step_hours * k).to_numpy(dtype="float32") for k in range(1, n_periods + 1)]
    )
    out = df.copy()
    with warnings.catch_warnings():  # filas sin historia suficiente -> NaN (esperado)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out[name] = np.nanmean(stacked, axis=1).astype("float32")
    return out


def add_previous_day_stats(df: pd.DataFrame, col: str = "departures") -> pd.DataFrame:
    """Total del día operativo anterior y promedio diario de los 28 días previos.

    Se calculan por estación sobre totales diarios desplazados un día, de modo
    que para el día D solo intervienen días <= D-1.

    Pertinencia (P1): describen el nivel de actividad reciente de la estación,
    útil cuando la serie horaria tiene muchos ceros.
    """
    daily = df.groupby(["station_id", "op_date"], sort=True)[col].sum().rename("day_total")
    daily = daily.reset_index()
    g = daily.groupby("station_id", sort=False)["day_total"]
    daily[f"{col}_prev_day"] = g.shift(1).astype("float32")
    daily[f"{col}_mean_28d"] = g.transform(
        lambda s: s.shift(1).rolling(28, min_periods=7).mean()
    ).astype("float32")
    return df.merge(
        daily[["station_id", "op_date", f"{col}_prev_day", f"{col}_mean_28d"]],
        on=["station_id", "op_date"],
        how="left",
    )


def add_train_profile(
    df: pd.DataFrame,
    col: str,
    train_mask: pd.Series,
    keys: list[str],
    name: str,
) -> pd.DataFrame:
    """Promedio histórico de ``col`` por ``keys`` calculado **solo con entrenamiento**.

    Es una codificación de la estación (target encoding) que sustituye al
    identificador numérico, el cual no tiene orden ni magnitud útil. Al usar
    únicamente filas de entrenamiento, validación y prueba no influyen en ella.
    """
    profile = df.loc[train_mask].groupby(keys, observed=True)[col].mean().rename(name)
    out = df.merge(profile.reset_index(), on=keys, how="left")
    out[name] = out[name].astype("float32")
    return out


def assign_split(ts: pd.Series, splits: dict = config.SPLITS) -> pd.Series:
    """Etiqueta cada fila como train/val/test según la partición temporal de config."""
    out = pd.Series(pd.NA, index=ts.index, dtype="string")
    for name, (a, b) in splits.items():
        mask = (ts >= pd.Timestamp(a)) & (ts < pd.Timestamp(b) + pd.Timedelta(days=1))
        out[mask] = name
    return out


# ---------------------------------------------------------------------------
# P1: demanda por estación y hora
# ---------------------------------------------------------------------------
P1_FEATURES = [
    "hour", "dow", "is_weekend", "month", "is_holiday", "op_hour", "capacity",
    "departures_lag_24", "departures_lag_48", "departures_lag_168", "departures_lag_336",
    "dep_same_hour_mean_7d", "dep_same_hour_dow_mean_4w",
    "departures_prev_day", "departures_mean_28d", "dep_profile_station_hour",
]  # fmt: skip


def build_p1_table(panel: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Tabla de P1: una fila por estación y hora de servicio del día siguiente.

    Variable objetivo: ``y`` = viajes iniciados (``departures``) en la hora.
    Momento de predicción: cierre del día operativo anterior (rezagos >= 24 h).
    Se excluyen las horas sin servicio (01:00-04:59), cuando no se pueden
    iniciar viajes.
    """
    h = config.P1_HORIZON_HOURS
    df = add_calendar_features(panel)
    df = add_lag_features(df, "departures", config.P1_LAGS_HOURS, min_lag=h)
    df = add_same_hour_mean(df, "departures", 7, 24, h, "dep_same_hour_mean_7d")
    df = add_same_hour_mean(df, "departures", 4, 168, h, "dep_same_hour_dow_mean_4w")
    df = add_previous_day_stats(df, "departures")
    df = df.loc[df["op_hour"] < config.SERVICE_HOURS].copy()
    df["split"] = assign_split(df["ts"])
    df = add_train_profile(
        df, "departures", df["split"] == "train",
        ["station_id", "op_hour", "is_weekend"], "dep_profile_station_hour",
    )  # fmt: skip
    df = df.merge(
        stations[["station_id", "capacity"]].astype({"station_id": "int32"}),
        on="station_id",
        how="left",
    )
    df["capacity"] = df["capacity"].astype("float32")
    df = df.rename(columns={"departures": "y"})
    cols = ["station_id", "ts", "op_date", "split", "y"] + P1_FEATURES
    return df[cols].sort_values(["ts", "station_id"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# P2: riesgo de vaciado / llenado
# ---------------------------------------------------------------------------
P2_FEATURES = [
    "hour", "dow", "is_weekend", "is_holiday", "op_hour", "capacity",
    "fill_ratio_now", "net_cum_today", "departures", "arrivals",
    "departures_last3h", "arrivals_last3h", "fill_ratio_lag_24",
    "net_next_window_yesterday", "net_next_window_last_week",
    "risk_rate_station_hour",
]  # fmt: skip


def estimate_occupancy(
    df: pd.DataFrame,
    capacity: pd.Series,
    initial_fill: float = config.P2_INITIAL_FILL,
) -> pd.DataFrame:
    """Aproxima la ocupación de cada estación a partir de los viajes y su capacidad.

    MiBici no publica el inventario de bicicletas. Se supone que al abrir cada
    día operativo (05:00, tras la ventana nocturna sin préstamos en la que se
    rebalancea) la estación tiene ``initial_fill`` x capacidad bicicletas y se
    acumula el balance horario llegadas - salidas, acotado al rango físico
    [0, capacidad]::

        est_bikes(apertura - 1) = initial_fill * capacidad
        est_bikes(t) = min(capacidad, max(0, est_bikes(t-1) + llegadas_t - salidas_t))
        fill_ratio(t) = est_bikes(t) / capacidad

    La cota refleja que una estación no puede tener menos de 0 bicicletas ni
    más bicicletas que anclajes: si los viajes observados la "sacarían" de ese
    rango, hubo una reposición o retiro no observado, y el conteo se reinicia
    desde el límite. ``net_cum_today`` conserva el balance acumulado sin acotar.
    ``fill_ratio_now`` en la hora *t* solo usa movimientos ya ocurridos hasta *t*.

    Requiere ``df`` ordenado por estación y hora (salida de
    :func:`hourly_station_panel`).
    """
    out = df.copy()
    out["net"] = out["arrivals"].astype("int32") - out["departures"].astype("int32")
    out["net_cum_today"] = (
        out.groupby(["station_id", "op_date"], sort=False)["net"].cumsum().astype("float32")
    )
    cap = capacity.to_numpy(dtype="float64")
    net = out["net"].to_numpy(dtype="float64")
    sid = out["station_id"].to_numpy()
    day = out["op_date"].to_numpy()
    new_day = np.ones(len(out), dtype=bool)
    new_day[1:] = (sid[1:] != sid[:-1]) | (day[1:] != day[:-1])
    est = np.empty(len(out), dtype="float64")
    level = 0.0
    for i in range(len(out)):  # recurrencia acotada (no vectorizable con cumsum)
        if new_day[i]:
            level = initial_fill * cap[i]
        level = min(cap[i], max(0.0, level + net[i]))
        est[i] = level
    out["est_bikes"] = est.astype("float32")
    out["fill_ratio_now"] = (est / cap).astype("float32")
    return out


def build_p2_target(
    df: pd.DataFrame,
    horizon: int = config.P2_HORIZON_HOURS,
    low: float = config.P2_LOW_FILL,
    high: float = config.P2_HIGH_FILL,
) -> pd.DataFrame:
    """Construye la etiqueta de riesgo para la ventana (t, t + horizon].

    ``y = 1`` si en alguna de las siguientes ``horizon`` horas del mismo día
    operativo la ocupación estimada es <= ``low`` (sin bicicletas) o >= ``high``
    (sin anclajes). ``risk_type`` distingue vaciado, llenado o ambos.

    **Solo para la etiqueta** se miran horas futuras (``shift(-k)``); ninguna
    variable explicativa usa estas columnas.
    """
    out = df.copy()
    g = out.groupby(["station_id", "op_date"], sort=False)["fill_ratio_now"]
    empty = np.zeros(len(out), dtype=bool)
    full = np.zeros(len(out), dtype=bool)
    for k in range(1, horizon + 1):
        fut = g.shift(-k).to_numpy()
        empty |= np.nan_to_num(fut, nan=np.inf) <= low
        full |= np.nan_to_num(fut, nan=-np.inf) >= high
    out["y"] = (empty | full).astype("int8")
    out["risk_type"] = np.select(
        [empty & full, empty, full], ["ambos", "vaciado", "llenado"], default="ninguno"
    )
    return out


def build_p2_table(panel: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Tabla de P2: una fila por estación y hora de decisión con etiqueta de riesgo.

    Solo se consideran horas de decisión cuya ventana futura completa cae dentro
    del horario de servicio (``op_hour + horizon <= 19``) y estaciones con
    capacidad conocida.
    """
    h = config.P2_HORIZON_HOURS
    st = stations.dropna(subset=["capacity"])[["station_id", "capacity"]].astype(
        {"station_id": "int32"}
    )
    df = panel.merge(st, on="station_id", how="inner")
    df = add_calendar_features(df)
    df = estimate_occupancy(df, df["capacity"])
    df = build_p2_target(df, horizon=h)

    g = df.groupby("station_id", sort=False)
    df["departures_last3h"] = g["departures"].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    )
    df["arrivals_last3h"] = g["arrivals"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    df["fill_ratio_lag_24"] = g["fill_ratio_now"].shift(24).astype("float32")
    # Balance de la MISMA ventana futura (t+1..t+h) pero de ayer y de la semana
    # pasada: rezagos de 24-k y 168-k horas, todos >= 21 h, ya observados en t.
    df["net_next_window_yesterday"] = sum(g["net"].shift(24 - k) for k in range(1, h + 1)).astype(
        "float32"
    )
    df["net_next_window_last_week"] = sum(g["net"].shift(168 - k) for k in range(1, h + 1)).astype(
        "float32"
    )

    df = df.loc[df["op_hour"] + h <= config.SERVICE_HOURS - 1].copy()
    df["split"] = assign_split(df["ts"])
    df = add_train_profile(
        df,
        "y",
        df["split"] == "train",
        ["station_id", "op_hour", "is_weekend"],
        "risk_rate_station_hour",
    )
    df["capacity"] = df["capacity"].astype("float32")
    for c in ["departures", "arrivals", "departures_last3h", "arrivals_last3h"]:
        df[c] = df[c].astype("float32")
    cols = ["station_id", "ts", "op_date", "split", "y", "risk_type"] + P2_FEATURES
    return df[cols].sort_values(["ts", "station_id"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# P3: perfiles de uso por estación
# ---------------------------------------------------------------------------
def _block_of_hour(hours: pd.Series) -> pd.Series:
    """Asigna cada hora (0-23) a su bloque horario de ``config.P3_TIME_BLOCKS``."""
    edges = [a for a, _ in config.P3_TIME_BLOCKS.values()] + [24]
    return pd.cut(hours, bins=edges, right=False, labels=list(config.P3_TIME_BLOCKS))


def station_usage_profiles(
    trips: pd.DataFrame, stations: pd.DataFrame, start: str, end: str
) -> pd.DataFrame:
    """Vector de patrón de uso por estación en el periodo [start, end].

    Variables (todas proporciones o escalas comparables entre estaciones):

    * ``dep_<bloque>`` / ``arr_<bloque>``: proporción de salidas/llegadas de días
      hábiles en cada bloque horario (madrugada, pico mañana, mediodía, tarde,
      pico tarde, noche). Distinguen estaciones "origen" residenciales (salidas
      en la mañana) de estaciones "destino" de empleo/estudio (llegadas en la
      mañana).
    * ``weekend_share``: proporción de viajes en fin de semana (uso recreativo).
    * ``log_daily_trips``: volumen diario (salidas + llegadas) en escala log.
    * ``net_flow_ratio``: (llegadas - salidas) / (llegadas + salidas); indica si la
      estación tiende a acumular o a vaciarse (vínculo con P2).
    * ``mean_duration_min`` y ``round_trip_share``: duración media y proporción de
      viajes que regresan a la misma estación (uso recreativo vs. traslado).

    El volumen diario se normaliza por los días en que la estación estuvo activa
    en el periodo, para no subestimar estaciones inauguradas a mitad del periodo.

    Pertinencia (P3): describen *cómo* se usa la estación, no *dónde* está; la
    ubicación se usa después para interpretar y mapear los grupos.
    """
    end_excl = pd.Timestamp(end) + pd.Timedelta(days=1)
    t = trips.loc[(trips["start_time"] >= pd.Timestamp(start)) & (trips["start_time"] < end_excl)]
    # Días activos por estación dentro del periodo: desde su primer movimiento
    # observado (estaciones nuevas) hasta el fin del periodo.
    first_seen = (
        pd.concat(
            [
                t.groupby(t["origin_station_id"].astype("int32"))["start_time"].min(),
                t.groupby(t["destination_station_id"].astype("int32"))["start_time"].min(),
            ]
        )
        .groupby(level=0)
        .min()
    )
    active_days = ((end_excl - first_seen.dt.normalize()).dt.days).clip(lower=1)

    def _profile(station_col: str, time_col: str, prefix: str) -> pd.DataFrame:
        x = pd.DataFrame(
            {
                "station_id": t[station_col].astype("int32"),
                "weekend": t[time_col].dt.dayofweek >= 5,
                "block": _block_of_hour(t[time_col].dt.hour),
            }
        )
        wk = (
            x.loc[~x["weekend"]]
            .groupby(["station_id", "block"], observed=False)
            .size()
            .unstack(fill_value=0)
        )
        wk = wk.reindex(columns=list(config.P3_TIME_BLOCKS), fill_value=0)
        wk = wk.div(wk.sum(axis=1).replace(0, np.nan), axis=0).add_prefix(f"{prefix}_")
        tot = x.groupby("station_id").size().rename(f"{prefix}_total")
        wkd = x.groupby("station_id")["weekend"].sum().rename(f"{prefix}_weekend")
        return wk.join(tot, how="outer").join(wkd, how="outer")

    dep = _profile("origin_station_id", "start_time", "dep")
    arr = _profile("destination_station_id", "end_time", "arr")
    prof = dep.join(arr, how="outer").fillna(0)
    total = prof["dep_total"] + prof["arr_total"]
    prof["weekend_share"] = (prof["dep_weekend"] + prof["arr_weekend"]) / total.replace(0, np.nan)
    n_days = active_days.reindex(prof.index)
    prof["log_daily_trips"] = np.log1p(total / n_days)
    prof["net_flow_ratio"] = (prof["arr_total"] - prof["dep_total"]) / total.replace(0, np.nan)
    by_origin = t.assign(
        round_trip=(t["origin_station_id"] == t["destination_station_id"]).astype("float32"),
        duration_min=t["duration_s"] / 60,
    ).groupby(t["origin_station_id"].astype("int32"))
    prof["mean_duration_min"] = by_origin["duration_min"].mean()
    prof["round_trip_share"] = by_origin["round_trip"].mean()
    prof["daily_trips"] = total / n_days
    prof = prof.drop(columns=["dep_total", "arr_total", "dep_weekend", "arr_weekend"])
    prof.index.name = "station_id"
    prof = prof.reset_index()
    st = stations[["station_id", "name", "lat", "lon"]].astype({"station_id": "int32"})
    prof = prof.merge(st, on="station_id", how="inner")
    prof = prof.loc[prof["daily_trips"] >= config.P3_MIN_DAILY_TRIPS].dropna(subset=["lat", "lon"])
    return prof.fillna(0).sort_values("station_id").reset_index(drop=True)


# Variables de agrupamiento: describen la FORMA del patrón de uso (cuándo y en qué
# dirección se usa la estación), no su magnitud.
P3_FEATURES = (
    [f"dep_{b}" for b in config.P3_TIME_BLOCKS]
    + [f"arr_{b}" for b in config.P3_TIME_BLOCKS]
    + ["net_flow_ratio", "weekend_share"]
)
# Variables descriptivas: se reportan por grupo para interpretarlo (intensidad y tipo
# de viaje), pero no se usan para formar los grupos, para que el volumen no domine.
P3_DESCRIPTIVE = ["log_daily_trips", "mean_duration_min", "round_trip_share"]


# ---------------------------------------------------------------------------
# Comando
# ---------------------------------------------------------------------------
@app.command()
def build() -> None:
    """Construye y guarda las tablas analíticas de P1, P2 y P3 en data/processed."""
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    stations = load_stations()
    trips = filter_known_stations(load_trips(), stations)
    logger.info(f"Viajes para modelar: {len(trips):,}")

    panel = hourly_station_panel(trips)
    logger.info(
        f"Panel estación x hora: {len(panel):,} filas, {panel['station_id'].nunique()} estaciones"
    )

    p1 = build_p1_table(panel, stations)
    p1.to_parquet(config.P1_DATASET_PATH, index=False)
    logger.success(f"P1 -> {config.P1_DATASET_PATH.name}: {len(p1):,} filas")
    del p1

    p2 = build_p2_table(panel, stations)
    p2.to_parquet(config.P2_DATASET_PATH, index=False)
    logger.success(
        f"P2 -> {config.P2_DATASET_PATH.name}: {len(p2):,} filas, tasa de riesgo {p2['y'].mean():.3f}"
    )
    del p2, panel

    a, b = config.SPLITS["train"]
    p3_train = station_usage_profiles(trips, stations, a, b)
    a, b = config.SPLITS["test"]
    p3_test = station_usage_profiles(trips, stations, a, b)
    p3_train.to_parquet(config.P3_TRAIN_PROFILE_PATH, index=False)
    p3_test.to_parquet(config.P3_TEST_PROFILE_PATH, index=False)
    logger.success(f"P3 -> perfiles: {len(p3_train)} estaciones (train), {len(p3_test)} (test)")


if __name__ == "__main__":
    app()
