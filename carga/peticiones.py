"""Peticiones realistas y válidas para la prueba de carga de la API P1 (solo biblioteca estándar).

Cada petición sale de una **serie diaria simulada** de la estación (28 días, con patrón semanal y
ruido de Poisson), y los rezagos y las medias se calculan de esa serie exactamente como en
``mibici/features.py``. Así cumplen por construcción las reglas de consistencia de la API
(``media_7d`` contiene a ``lag_1``, ``lag_2`` y ``lag_7``, etc.) y el modelo trabaja con entradas
del mismo tipo que en producción, no con un único ejemplo repetido que el procesador cachearía.

Los niveles de demanda imitan los datos de MiBici 2025: mediana cercana a 25 viajes por
estación-día, cola larga hasta ~250, capacidad de 11 a 39 anclajes y unas 470 estaciones.
La semilla es fija: dos corridas de la prueba envían la misma secuencia de entradas.
"""

from datetime import date, timedelta
import math
import random

SEMILLA = 20261118
ESTACIONES = 470
INICIO = date(2025, 11, 1)  # primer día del conjunto de prueba de la v1
DIAS = 334  # hasta el 30 de septiembre de 2026 (datos de producción de la entrega 5)


def _poisson(rng: random.Random, lam: float) -> int:
    """Poisson por el método de Knuth (lam chico) o aproximación normal (lam grande)."""
    if lam <= 0:
        return 0
    if lam < 30:
        limite, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= rng.random()
            if p <= limite:
                return k
            k += 1
    return max(0, round(rng.gauss(lam, math.sqrt(lam))))


def peticion(rng: random.Random, estacion_id: int, dia: date) -> dict:
    """Variables de una estación para predecir el día ``dia`` (D), conocidas al cierre de D-1."""
    nivel = min(250.0, rng.lognormvariate(3.2, 0.7))  # viajes en un día hábil típico
    serie = []  # D-28 ... D-1
    for atras in range(28, 0, -1):
        d = dia - timedelta(days=atras)
        factor = 0.6 if d.weekday() >= 5 else 1.0
        serie.append(_poisson(rng, nivel * factor))
    dia_semana = dia.weekday()
    return {
        "estacion_id": estacion_id,
        "fecha": dia.isoformat(),
        "dia_semana": dia_semana,
        "fin_semana": int(dia_semana >= 5),
        "feriado": int(rng.random() < 0.03),
        "lag_1": serie[-1],
        "lag_2": serie[-2],
        "lag_7": serie[-7],
        "lag_14": serie[-14],
        "media_7d": round(sum(serie[-7:]) / 7, 4),
        "media_28d": round(sum(serie) / 28, 4),
        "media_mismo_dia_4sem": round((serie[-7] + serie[-14] + serie[-21] + serie[-28]) / 4, 4),
        "dias_activa": rng.randint(28, 700),
        "capacidad": rng.randint(11, 39),
    }


def generar(n: int = 2000, semilla: int = SEMILLA) -> list[dict]:
    """``n`` peticiones de estaciones y días al azar (reproducibles con la misma semilla)."""
    rng = random.Random(semilla)  # nosec B311: datos de prueba reproducibles, no criptografía
    return [
        peticion(rng, rng.randint(1, ESTACIONES), INICIO + timedelta(days=rng.randrange(DIAS)))
        for _ in range(n)
    ]


def invalida(base: dict) -> dict:
    """Entrada incoherente a propósito (fin de semana en un lunes): la API debe responder 422."""
    return {**base, "dia_semana": 0, "fin_semana": 1, "fecha": None}


def escalones(texto: str) -> list[tuple[int, int]]:
    """"5:60,10:60" -> [(5, 60), (10, 60)] (usuarios, segundos de cada escalón)."""
    pares = []
    for parte in texto.split(","):
        usuarios, segundos = parte.strip().split(":")
        pares.append((int(usuarios), int(segundos)))
    return pares
