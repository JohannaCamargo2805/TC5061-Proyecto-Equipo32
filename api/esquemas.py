"""Contratos de entrada y salida de la API (Pydantic v2).

La entrada son las 12 variables con las que se entrenó el modelo P1 (``P1_FEATURES``
en ``mibici/features.py``), todas conocidas al cierre del día anterior (D-1) al que se
predice. La validación tiene tres capas:

1. **Tipo y rango de cada campo**: enteros donde la variable es un conteo, números
   finitos, límites físicos (día 0-6, banderas 0/1, conteos no negativos) y
   ``extra="forbid"`` para rechazar campos desconocidos o mal escritos. Los textos y
   booleanos se rechazan aunque parezcan números (``"62"``, ``true``).
2. **Consistencia entre campos**, que se cumple por construcción en los datos reales:
   ``fin_semana`` debe corresponder a ``dia_semana`` (y a ``fecha``, si se envía) y las
   medias móviles deben contener los rezagos que caen dentro de su ventana.
3. **Advertencias** (no bloquean): valores fuera del rango visto en entrenamiento,
   donde el modelo extrapola.

La salida también se valida: la predicción es un número finito y no negativo.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Orden exacto de la firma del modelo registrado (MLmodel -> signature.inputs).
VARIABLES_P1: tuple[str, ...] = (
    "dia_semana",
    "fin_semana",
    "feriado",
    "lag_1",
    "lag_2",
    "lag_7",
    "lag_14",
    "media_7d",
    "media_28d",
    "media_mismo_dia_4sem",
    "dias_activa",
    "capacidad",
)

DESCRIPCIONES: dict[str, str] = {
    "dia_semana": "Día de la semana del día que se predice: 0 = lunes … 6 = domingo.",
    "fin_semana": "1 si el día que se predice es sábado o domingo; 0 en otro caso.",
    "feriado": "1 si el día que se predice es de descanso obligatorio (art. 74 LFT) o Jueves/Viernes Santo.",
    "lag_1": "Viajes iniciados en la estación el día anterior (D-1).",
    "lag_2": "Viajes iniciados hace 2 días (D-2).",
    "lag_7": "Viajes iniciados hace 7 días (D-7, mismo día de la semana anterior).",
    "lag_14": "Viajes iniciados hace 14 días (D-14).",
    "media_7d": "Promedio diario de viajes de D-7 a D-1.",
    "media_28d": "Promedio diario de viajes de D-28 a D-1.",
    "media_mismo_dia_4sem": "Promedio de D-7, D-14, D-21 y D-28 (mismo día de la semana, 4 semanas).",
    "dias_activa": "Días de historia de la estación hasta D-1 (mínimo 28, los que necesita media_28d).",
    "capacidad": "Anclajes instalados en la estación.",
}

MAX_CONTEO = 2000  # tope de plausibilidad por estación y día (el máximo observado en 2025 es 550)
LOTE_MAXIMO = (
    1000  # instancias por petición en /predict/lote (MiBici tiene ~370 estaciones activas)
)
TOLERANCIA = 0.5  # viajes; admite medias redondeadas por el cliente

# Rango de cada variable en TRAIN (ene-sep 2025, 90,230 estación-días). Fuera de él el
# modelo extrapola: la API responde, pero lo indica en ``advertencias``.
RANGO_ENTRENAMIENTO: dict[str, tuple[float, float]] = {
    "lag_1": (0, 550),
    "lag_2": (0, 550),
    "lag_7": (0, 550),
    "lag_14": (0, 550),
    "media_7d": (0, 273.857),
    "media_28d": (0, 263.464),
    "media_mismo_dia_4sem": (0, 295.25),
    "dias_activa": (28, 272),
    "capacidad": (11, 39),
}

# Estación 2, martes 4 de noviembre de 2025 (conjunto de prueba): se iniciaron 63 viajes.
EJEMPLO_ENTRADA = {
    "estacion_id": 2,
    "fecha": "2025-11-04",
    "dia_semana": 1,
    "fin_semana": 0,
    "feriado": 0,
    "lag_1": 61,
    "lag_2": 27,
    "lag_7": 65,
    "lag_14": 64,
    "media_7d": 56.5714,
    "media_28d": 55.0357,
    "media_mismo_dia_4sem": 64.0,
    "dias_activa": 307,
    "capacidad": 15,
}

_ENTEROS = (
    "dia_semana",
    "fin_semana",
    "feriado",
    "lag_1",
    "lag_2",
    "lag_7",
    "lag_14",
    "dias_activa",
    "capacidad",
)
_REALES = ("media_7d", "media_28d", "media_mismo_dia_4sem")


class EntradaP1(BaseModel):
    """Variables de una estación para predecir sus viajes del día D."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EJEMPLO_ENTRADA]})

    estacion_id: int | None = Field(
        None,
        ge=1,
        description="Identificador de la estación (opcional; se devuelve con la predicción).",
    )
    fecha: date | None = Field(
        None,
        description="Día D que se predice, AAAA-MM-DD (opcional). Si se envía, debe coincidir con dia_semana.",
    )
    dia_semana: int = Field(..., ge=0, le=6, description=DESCRIPCIONES["dia_semana"])
    fin_semana: int = Field(..., ge=0, le=1, description=DESCRIPCIONES["fin_semana"])
    feriado: int = Field(..., ge=0, le=1, description=DESCRIPCIONES["feriado"])
    lag_1: int = Field(..., ge=0, le=MAX_CONTEO, description=DESCRIPCIONES["lag_1"])
    lag_2: int = Field(..., ge=0, le=MAX_CONTEO, description=DESCRIPCIONES["lag_2"])
    lag_7: int = Field(..., ge=0, le=MAX_CONTEO, description=DESCRIPCIONES["lag_7"])
    lag_14: int = Field(..., ge=0, le=MAX_CONTEO, description=DESCRIPCIONES["lag_14"])
    media_7d: float = Field(
        ..., ge=0, le=MAX_CONTEO, allow_inf_nan=False, description=DESCRIPCIONES["media_7d"]
    )
    media_28d: float = Field(
        ..., ge=0, le=MAX_CONTEO, allow_inf_nan=False, description=DESCRIPCIONES["media_28d"]
    )
    media_mismo_dia_4sem: float = Field(
        ...,
        ge=0,
        le=MAX_CONTEO,
        allow_inf_nan=False,
        description=DESCRIPCIONES["media_mismo_dia_4sem"],
    )
    dias_activa: int = Field(..., ge=28, le=3650, description=DESCRIPCIONES["dias_activa"])
    capacidad: int = Field(..., ge=1, le=100, description=DESCRIPCIONES["capacidad"])

    @field_validator(*_ENTEROS, *_REALES, mode="before")
    @classmethod
    def _solo_numeros(cls, valor):
        """Rechaza textos y booleanos que Pydantic convertiría en número en modo flexible."""
        if isinstance(valor, (str, bool)):
            # ValueError (no TypeError): Pydantic solo convierte ValueError en un error 422.
            raise ValueError("debe ser un número (no texto ni booleano)")  # noqa: TRY004
        return valor

    @model_validator(mode="after")
    def _consistencia(self):
        errores = []
        if self.fin_semana != int(self.dia_semana >= 5):
            errores.append(
                f"fin_semana={self.fin_semana} no corresponde a dia_semana={self.dia_semana} "
                "(fin_semana es 1 solo para 5 = sábado y 6 = domingo)"
            )
        if self.fecha is not None and self.fecha.weekday() != self.dia_semana:
            errores.append(
                f"fecha {self.fecha.isoformat()} es día {self.fecha.weekday()} de la semana, "
                f"pero dia_semana={self.dia_semana}"
            )
        # La ventana D-7..D-1 contiene D-1, D-2 y D-7.
        suma = self.lag_1 + self.lag_2 + self.lag_7
        if 7 * self.media_7d + TOLERANCIA < suma:
            errores.append(
                f"media_7d={self.media_7d} es imposible: 7 × media_7d debe ser al menos "
                f"lag_1 + lag_2 + lag_7 = {suma}"
            )
        # media_mismo_dia_4sem = (D-7 + D-14 + D-21 + D-28) / 4.
        suma = self.lag_7 + self.lag_14
        if 4 * self.media_mismo_dia_4sem + TOLERANCIA < suma:
            errores.append(
                f"media_mismo_dia_4sem={self.media_mismo_dia_4sem} es imposible: 4 × "
                f"media_mismo_dia_4sem debe ser al menos lag_7 + lag_14 = {suma}"
            )
        # La ventana D-28..D-1 contiene D-1..D-7 y además D-14, D-21 y D-28.
        minimo = 7 * self.media_7d + 4 * self.media_mismo_dia_4sem - self.lag_7
        if 28 * self.media_28d + TOLERANCIA < minimo:
            errores.append(
                f"media_28d={self.media_28d} es imposible: 28 × media_28d debe ser al menos "
                f"7 × media_7d + 4 × media_mismo_dia_4sem − lag_7 = {minimo:.2f}"
            )
        if errores:
            raise ValueError("; ".join(errores))
        return self

    def fila(self) -> dict[str, float]:
        """Las 12 variables en el orden y tipo (double) de la firma del modelo."""
        return {v: float(getattr(self, v)) for v in VARIABLES_P1}

    def advertencias(self) -> list[str]:
        avisos = []
        for variable, (minimo, maximo) in RANGO_ENTRENAMIENTO.items():
            valor = getattr(self, variable)
            if not minimo <= valor <= maximo:
                avisos.append(
                    f"{variable}={valor} está fuera del rango de entrenamiento "
                    f"[{minimo:g}, {maximo:g}]: el modelo extrapola"
                )
        return avisos


class SolicitudLote(BaseModel):
    """Varias estaciones o días en una sola petición (una sola llamada al modelo)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "instancias": [
                        EJEMPLO_ENTRADA,
                        {
                            **EJEMPLO_ENTRADA,
                            "fecha": "2025-11-05",
                            "dia_semana": 2,
                            "lag_1": 63,
                            "lag_2": 61,
                            "lag_7": 62,
                            "lag_14": 56,
                            "media_7d": 56.2857,
                            "media_28d": 55.25,
                            "media_mismo_dia_4sem": 65.0,
                            "dias_activa": 308,
                        },
                    ]
                }
            ]
        },
    )

    instancias: list[EntradaP1] = Field(..., min_length=1, max_length=LOTE_MAXIMO)


class ModeloResumen(BaseModel):
    nombre: str = Field(..., description="Nombre del modelo en el Model Registry de MLflow.")
    version: int = Field(..., ge=1, description="Versión registrada que sirve la API.")
    alias: list[str] = Field(
        ..., description="Aliases que apuntaban a esa versión al recuperarla."
    )
    run_id: str = Field(..., description="Corrida de MLflow que entrenó el modelo.")


class PrediccionItem(BaseModel):
    estacion_id: int | None = None
    fecha: date | None = None
    viajes_esperados: float = Field(
        ...,
        ge=0,
        allow_inf_nan=False,
        description="Viajes que se espera inicien en la estación el día D.",
    )
    viajes_esperados_redondeado: int = Field(
        ..., ge=0, description="viajes_esperados redondeado al entero."
    )
    advertencias: list[str] = Field(
        default_factory=list,
        description="Avisos no bloqueantes (p. ej. valores fuera del rango de entrenamiento).",
    )


class PrediccionP1(PrediccionItem):
    modelo: ModeloResumen


class RespuestaLote(BaseModel):
    n: int = Field(
        ..., ge=1, description="Número de predicciones (mismo orden que las instancias)."
    )
    modelo: ModeloResumen
    predicciones: list[PrediccionItem]


class EstadoSalud(BaseModel):
    estado: Literal["ok"]
    modelo: str
    version: int
    origen: Literal["empaquetado", "registry"]


class Variable(BaseModel):
    nombre: str
    tipo: str
    descripcion: str


class InfoModelo(BaseModel):
    nombre: str
    version: int
    alias: list[str]
    run_id: str
    model_id: str | None = None
    origen: Literal["empaquetado", "registry"]
    uri_registry: str = Field(
        ..., description="URI del modelo en el registry por nombre y versión."
    )
    git_commit: str | None = None
    data_version: str | None = None
    tipo_modelo: str | None = None
    metricas: dict[str, float] = Field(default_factory=dict)
    variables: list[Variable]
    salida: str
    recuperado: str | None = Field(
        None, description="Fecha (UTC) en que se descargó del registry."
    )
    integridad_verificada: bool
