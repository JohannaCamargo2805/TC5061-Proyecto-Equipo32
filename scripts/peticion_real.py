"""Construye una petición de ``POST /predict`` con los datos reales de una estación y un día.

Calcula las 12 variables con las mismas funciones del entrenamiento (``mibici.monitoreo``,
panel continuo 2025-2026) y guarda el JSON listo para la API, junto con los viajes que de verdad
se iniciaron ese día para comparar con la predicción::

    python scripts/peticion_real.py --estacion 2 --fecha 2026-09-30
    python scripts/peticion_real.py --fecha 2026-09-30 --salida docs/ejemplos/peticion_demo.json
    # sin --estacion: la estación con más viajes ese día

Requiere los Parquet de ``data/processed`` (2025 y producción 2026).
"""

import json
from pathlib import Path

import pandas as pd
import typer

from mibici import monitoreo as mon
from mibici.features import P1_FEATURES

app = typer.Typer(add_completion=False)
ENTEROS = {"dia_semana", "fin_semana", "feriado", "lag_1", "lag_2", "lag_7", "lag_14",
           "dias_activa", "capacidad"}


@app.command()
def main(
    fecha: str = typer.Option(..., help="Día D a predecir (AAAA-MM-DD)"),
    estacion: int = typer.Option(0, help="Estación; 0 = la de más viajes ese día"),
    salida: Path = typer.Option(None, help="Archivo JSON de la petición (opcional)"),
):
    """Petición real para la API y los viajes observados ese día."""
    dia = pd.Timestamp(fecha)
    panel = mon.construir_panel(fin=str((dia + pd.Timedelta(days=1)).date()))
    filas = mon.periodo(panel, str(dia.date()), str((dia + pd.Timedelta(days=1)).date())).reset_index()
    if filas.empty:
        raise typer.BadParameter(f"no hay estación-días completos el {fecha}")
    fila = (filas.loc[filas["estacion"] == estacion] if estacion
            else filas.sort_values("viajes", ascending=False).head(1))
    if fila.empty:
        raise typer.BadParameter(f"la estación {estacion} no tiene las 12 variables el {fecha}")
    f = fila.iloc[0]
    peticion = {"estacion_id": int(f["estacion"]), "fecha": str(dia.date()),
                **{v: (int(f[v]) if v in ENTEROS else round(float(f[v]), 4)) for v in P1_FEATURES}}
    texto = json.dumps(peticion, ensure_ascii=False, indent=2)
    if salida:
        salida.write_text(texto + "\n", encoding="utf-8")
    print(texto)
    print(f"VIAJES_REALES={int(f['viajes'])}")


if __name__ == "__main__":
    app()
