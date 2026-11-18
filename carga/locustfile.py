"""Prueba de carga de la API MiBici P1 con Locust (entrega 6).

Escenario: el tráfico que tendría la API si una app de MiBici y el equipo de operación la usaran a
la vez.

* ``UsuarioApp`` (peso 9): una persona consulta en la app cuántos viajes se esperan hoy en una
  estación. ``POST /predict`` con entradas realistas y distintas en cada petición
  (``peticiones.py``), más consultas ocasionales a ``/health`` y ``/modelo`` y un 4 % de entradas
  inválidas que la API debe rechazar con **422** (se cuentan como éxito si llega el 422).
* ``UsuarioOperacion`` (peso 1): el área de operación planea el rebalanceo y pide 100 estaciones
  en una sola llamada a ``POST /predict/lote``.
* ``UsuarioAsincrono`` (solo en ``ESCENARIO=asincrono``): encola lotes de 100 en
  ``POST /predict/asincrono`` y consulta ``GET /tareas/{id}`` hasta que el worker de RQ termina;
  el tiempo total se reporta como la petición ``ASYNC lote[100] completo``.

La forma de la carga la da ``ESCENARIO`` (variable de entorno):

* ``escalones`` (por defecto): escalones de usuarios de ``ESCALONES`` ("usuarios:segundos,...")
  para encontrar el punto en que la latencia se dispara (capacidad de una réplica).
* ``humo``: 20 usuarios durante ``DURACION`` segundos (60): la usa el pipeline de CI/CD.
* ``lote``: ``USUARIOS_LOTE`` usuarios (4) que piden lotes de 100 sin pausa durante ``DURACION``:
  mide el costo por inferencia cuando se predice en lote.
* ``asincrono``: ``USUARIOS_ASINC`` usuarios (10) del flujo asíncrono durante ``DURACION``.

Ejecución (desde la raíz del repositorio): ``python scripts/prueba_carga.py`` levanta el sistema,
corre este archivo dentro del contenedor ``locust`` y mide CPU y memoria con ``docker stats``.
Interfaz web: ``docker compose --profile carga up locust`` y abrir http://localhost:8089.
"""

import csv
import gzip
import itertools
import os
import random
import time

from locust import FastHttpUser, LoadTestShape, between, events, task
from peticiones import escalones, generar, invalida

POOL = generar(int(os.getenv("PETICIONES", "2000")))
TAMANO_LOTE = int(os.getenv("TAMANO_LOTE", "100"))
ESCALONES_DEFECTO = "10:45,25:45,50:45,100:45,150:45,200:45,250:45,300:45"
_ciclo = itertools.cycle(range(len(POOL)))


def siguiente() -> dict:
    return POOL[next(_ciclo)]


def lote(n: int) -> list[dict]:
    inicio = random.randrange(len(POOL) - n)  # nosec B311: carga de prueba, no criptografía
    return POOL[inicio : inicio + n]


def _respuesta_valida(respuesta, n: int | None = None) -> str | None:
    """Devuelve el motivo del fallo o None. Valida el contenido, no solo el código HTTP."""
    if respuesta.status_code != 200:
        return f"HTTP {respuesta.status_code}"
    try:
        cuerpo = respuesta.json()
    except ValueError:
        return "respuesta no es JSON"
    if n is None:
        v = cuerpo.get("viajes_esperados")
        return None if isinstance(v, (int, float)) and v >= 0 else "sin viajes_esperados"
    return None if cuerpo.get("n") == n else f"lote incompleto: {cuerpo.get('n')} de {n}"


class UsuarioApp(FastHttpUser):
    weight = 9
    wait_time = between(0.5, 1.5)

    @task(20)
    def predecir(self):
        with self.client.post("/predict", json=siguiente(), name="/predict",
                              catch_response=True) as r:
            motivo = _respuesta_valida(r)
            r.failure(motivo) if motivo else r.success()

    @task(1)
    def entrada_invalida(self):
        with self.client.post("/predict", json=invalida(siguiente()),
                              name="/predict [inválida→422]", catch_response=True) as r:
            r.success() if r.status_code == 422 else r.failure(f"esperaba 422: {r.status_code}")

    @task(1)
    def salud(self):
        self.client.get("/health", name="/health")

    @task(1)
    def modelo(self):
        self.client.get("/modelo", name="/modelo")


class UsuarioOperacion(FastHttpUser):
    weight = 1
    wait_time = between(2, 5)

    @task
    def predecir_lote(self):
        with self.client.post("/predict/lote", json={"instancias": lote(TAMANO_LOTE)},
                              name=f"/predict/lote [{TAMANO_LOTE}]", catch_response=True) as r:
            motivo = _respuesta_valida(r, TAMANO_LOTE)
            r.failure(motivo) if motivo else r.success()


class UsuarioLoteIntensivo(UsuarioOperacion):
    """Solo en ``ESCENARIO=lote``: lotes seguidos para medir el costo por inferencia en lote."""

    wait_time = between(0.0, 0.2)


class UsuarioAsincrono(FastHttpUser):
    weight = 1
    wait_time = between(1, 2)
    espera_maxima = float(os.getenv("ESPERA_MAXIMA_ASINC", "30"))

    @task
    def lote_asincrono(self):
        inicio = time.perf_counter()
        r = self.client.post("/predict/asincrono", json={"instancias": lote(TAMANO_LOTE)},
                             name=f"/predict/asincrono [{TAMANO_LOTE}]")
        if r.status_code != 202:
            return
        consultar = r.json()["consultar"]
        excepcion = None
        while True:
            time.sleep(0.2)
            estado = self.client.get(consultar, name="/tareas/{id}").json().get("estado")
            if estado == "finished":
                break
            if estado in {"failed", "canceled", "stopped"}:
                excepcion = RuntimeError(f"tarea {estado}")
                break
            if time.perf_counter() - inicio > self.espera_maxima:
                excepcion = TimeoutError(f"sin terminar en {self.espera_maxima:.0f} s")
                break
        events.request.fire(
            request_type="ASYNC",
            name=f"lote[{TAMANO_LOTE}] completo",
            response_time=(time.perf_counter() - inicio) * 1000,
            response_length=0,
            exception=excepcion,
            context={},
        )


# Registro de cada petición (inicio, endpoint, ms, éxito) para calcular percentiles exactos por
# escalón en carga/graficas.py; los CSV de Locust solo traen ventanas móviles de ~10 s.
SALIDA_PETICIONES = os.getenv("SALIDA_PETICIONES", "")
_peticiones: list[tuple] = []


@events.request.add_listener
def _registrar(request_type, name, response_time, exception, start_time=None, **_):
    fin = time.time()
    inicio = start_time if start_time else fin - response_time / 1000
    _peticiones.append((round(inicio, 4), request_type, name, round(response_time, 3),
                        int(exception is None)))


@events.quitting.add_listener
def _guardar(environment, **_):
    if not SALIDA_PETICIONES:
        return
    with gzip.open(SALIDA_PETICIONES, "wt", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["inicio", "tipo", "nombre", "ms", "ok"])
        escritor.writerows(_peticiones)


class FormaDeCarga(LoadTestShape):
    """Escalones de usuarios; cada escalón es estable el tiempo suficiente para medirlo."""

    escenario = os.getenv("ESCENARIO", "escalones")

    def __init__(self):
        super().__init__()
        duracion = int(os.getenv("DURACION", "60"))
        if self.escenario == "humo":
            self.plan = [(20, duracion)]
            self.clases = [UsuarioApp, UsuarioOperacion]
        elif self.escenario == "lote":
            self.plan = [(int(os.getenv("USUARIOS_LOTE", "4")), duracion)]
            self.clases = [UsuarioLoteIntensivo]
        elif self.escenario == "asincrono":
            self.plan = [(int(os.getenv("USUARIOS_ASINC", "10")), duracion)]
            self.clases = [UsuarioAsincrono]
        else:
            self.plan = escalones(os.getenv("ESCALONES", ESCALONES_DEFECTO))
            self.clases = [UsuarioApp, UsuarioOperacion]

    def tick(self):
        transcurrido = self.get_run_time()
        acumulado = 0
        for usuarios, segundos in self.plan:
            acumulado += segundos
            if transcurrido < acumulado:
                # Tasa de arranque alta: el escalón se alcanza en ~1 s y el resto es estable.
                return usuarios, max(usuarios, 10), self.clases
        return None
