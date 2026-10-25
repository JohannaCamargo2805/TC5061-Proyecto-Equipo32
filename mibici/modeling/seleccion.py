"""Regla de selección de modelos por pregunta (ver ``SELECCION`` en config.py).

Se usa en la búsqueda de hiperparámetros (mejor combinación) y en el Model
Registry (qué corrida pasa a ``champion`` y cuál queda como ``challenger``).
"""

import numpy as np
import pandas as pd

from mibici.config import SELECCION


def _cumple(tabla: pd.DataFrame, restricciones: dict) -> pd.Series:
    ok = pd.Series(True, index=tabla.index)
    for col, (op, valor) in restricciones.items():
        serie = tabla.get(col, pd.Series(np.nan, index=tabla.index))
        ok &= serie >= valor if op == ">=" else serie <= valor
    return ok


def ordenar(tabla: pd.DataFrame, question: str, un_ee: bool = True) -> pd.DataFrame:
    """Ordena las corridas de mejor a peor y explica la decisión en la columna ``motivo``.

    ``tabla`` necesita una fila por corrida con las métricas como columnas.
    Con ``un_ee=True`` aplica la regla de un error estándar cuando la pregunta la define.
    """
    c = SELECCION[question]
    asc = c["orden"] == "min"
    t = tabla.copy()
    t["cumple_restricciones"] = _cumple(t, c.get("restricciones", {}))
    t["dentro_1ee"] = False
    if un_ee and c.get("error_estandar") and t[c["error_estandar"]].notna().any():
        base = t[t["cumple_restricciones"]].sort_values(c["metrica"], ascending=asc).iloc[0]
        limite = base[c["metrica"]] + (base[c["error_estandar"]] if asc else -base[c["error_estandar"]])
        t["dentro_1ee"] = t[c["metrica"]] <= limite if asc else t[c["metrica"]] >= limite
        t["limite_1ee"] = limite
        clave = [~t["cumple_restricciones"], ~t["dentro_1ee"], t[c["complejidad"]],
                 t[c["metrica"]] * (1 if asc else -1)]
    else:
        clave = [~t["cumple_restricciones"], t[c["metrica"]] * (1 if asc else -1)]
    orden = np.lexsort([np.asarray(k, dtype=float) for k in reversed(clave)])
    t = t.iloc[orden].reset_index(drop=True)
    t["motivo"] = [
        ("no cumple restricciones de negocio/estabilidad" if not r["cumple_restricciones"]
         else "dentro de 1 EE del mejor; se prefiere el de menor complejidad" if r["dentro_1ee"]
         else "mejor métrica de selección" if not c.get("error_estandar")
         else "fuera de 1 EE del mejor")
        for _, r in t.iterrows()
    ]
    return t
