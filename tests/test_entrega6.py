"""Pruebas de la entrega 6: orquestación, prueba de carga y análisis FinOps.

* ``docker-compose.yml`` y los Dockerfile de MLflow y de Locust: decisiones que hacen al sistema
  reproducible y seguro (versiones fijas, usuario sin privilegios, salud, redes, puertos locales).
* ``carga/peticiones.py``: las peticiones de la prueba de carga son válidas para la API.
* ``carga/graficas.py`` y ``finops/costo_inferencia.py``: con una corrida sintética producen el
  resumen por escalón, la capacidad y el costo por inferencia con las fórmulas esperadas.
* ``docker/mlflow/registrar_modelo.py``: registra el modelo empaquetado y es idempotente.

No levantan contenedores: eso lo hace el pipeline de CI/CD (``docker compose up --wait`` y una
prueba de humo con Locust).
"""

import gzip
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys

from pydantic import ValidationError
import pytest
import yaml

from api.esquemas import EntradaP1
from mibici.config import PROJ_ROOT, SERVICIO_MODELO_DIR

COMPOSE = yaml.safe_load((PROJ_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
SERVICIOS = COMPOSE["services"]


def _modulo(ruta: Path, nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def _pines(ruta: Path) -> dict[str, str]:
    pines = {}
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.split("#")[0].strip()
        if "==" in linea:
            nombre, version = linea.split("==")
            pines[nombre.strip().lower()] = version.strip()
    return pines


def _imagen_base(ruta: Path) -> str:
    texto = ruta.read_text(encoding="utf-8")
    return re.search(r"^ARG IMAGEN_PYTHON=(\S+)", texto, re.MULTILINE).group(1)


# ---------------------------------------------------------------- docker-compose.yml


def test_servicios_del_sistema():
    assert set(SERVICIOS) == {"mlflow", "registrar-modelo", "api", "redis", "worker", "locust"}
    assert SERVICIOS["locust"]["profiles"] == ["carga"]  # no arranca con "docker compose up"
    sin_perfil = [n for n, s in SERVICIOS.items() if "profiles" not in s]
    assert set(sin_perfil) == set(SERVICIOS) - {"locust"}


def test_orden_de_arranque_con_condiciones():
    dep = {n: s.get("depends_on", {}) for n, s in SERVICIOS.items()}
    assert dep["registrar-modelo"]["mlflow"]["condition"] == "service_healthy"
    for servicio in ("api", "worker"):
        assert dep[servicio]["registrar-modelo"]["condition"] == "service_completed_successfully"
        assert dep[servicio]["redis"]["condition"] == "service_healthy"
    assert dep["locust"]["api"]["condition"] == "service_healthy"


def test_servicios_permanentes_con_healthcheck_reinicio_y_limites():
    for nombre in ("mlflow", "api", "redis", "worker"):
        s = SERVICIOS[nombre]
        assert s["healthcheck"]["test"], nombre
        assert s["restart"] == "unless-stopped", nombre
        assert "cpus" in s and "mem_limit" in s, nombre
        assert s["logging"]["options"]["max-size"], nombre
    assert SERVICIOS["registrar-modelo"]["restart"] == "no"


def test_redes_y_volumenes():
    assert COMPOSE["networks"]["cola"]["internal"] is True  # Redis no sale de la red interna
    assert SERVICIOS["redis"]["networks"] == ["cola"] and "ports" not in SERVICIOS["redis"]
    assert set(SERVICIOS["api"]["networks"]) == {"frontal", "registro", "cola"}
    assert set(COMPOSE["volumes"]) == {"mlflow-datos", "redis-datos"}
    assert "mlflow-datos:/mlflow" in SERVICIOS["mlflow"]["volumes"]
    assert "./models/servicio:/modelo:ro" in SERVICIOS["registrar-modelo"]["volumes"]


def test_puertos_solo_en_localhost():
    for nombre, s in SERVICIOS.items():
        for puerto in s.get("ports", []):
            assert str(puerto).startswith("127.0.0.1:"), (nombre, puerto)


def test_sin_secretos_en_el_compose():
    texto = (PROJ_ROOT / "docker-compose.yml").read_text(encoding="utf-8").lower()
    for palabra in ("password", "token", "secret", "api_key"):
        assert f"{palabra}:" not in texto and f"{palabra}=" not in texto


def test_la_api_toma_el_modelo_del_registry_de_compose():
    entorno = SERVICIOS["api"]["environment"]
    assert entorno["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    assert "models:/mibici-equipo32-p1-demanda@champion" in entorno["MODELO_URI"]
    assert entorno["REDIS_URL"] == "redis://redis:6379/0"
    assert entorno["VERIFICAR_INTEGRIDAD"] == "1"


def test_imagenes_con_version_fija():
    base = _imagen_base(PROJ_ROOT / "Containerfile")
    for dockerfile in ("docker/mlflow/Dockerfile", "carga/Dockerfile"):
        assert _imagen_base(PROJ_ROOT / dockerfile) == base, dockerfile
    args = SERVICIOS["api"]["build"]["args"]
    assert base in args["IMAGEN_PYTHON"]
    assert re.search(r"redis:\d+\.\d+\.\d+-alpine", SERVICIOS["redis"]["image"])


@pytest.mark.parametrize("dockerfile", ["docker/mlflow/Dockerfile", "carga/Dockerfile"])
def test_dockerfiles_sin_root(dockerfile):
    texto = (PROJ_ROOT / dockerfile).read_text(encoding="utf-8")
    usuarios = re.findall(r"^USER\s+(\S+)", texto, re.MULTILINE)
    assert usuarios and usuarios[-1].split(":")[0] not in {"root", "0"}


@pytest.mark.parametrize("archivo", ["docker/mlflow/requirements-mlflow.txt",
                                     "carga/requirements-carga.txt"])
def test_dependencias_fijas(archivo):
    for linea in (PROJ_ROOT / archivo).read_text(encoding="utf-8").splitlines():
        linea = linea.split("#")[0].strip()
        if linea:
            assert re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==[\w.]+", linea), linea


def test_mlflow_del_servidor_igual_al_del_proyecto():
    servidor = _pines(PROJ_ROOT / "docker/mlflow/requirements-mlflow.txt")
    proyecto = _pines(PROJ_ROOT / "requirements.txt")
    assert servidor["mlflow"] == proyecto["mlflow"]
    modelo = _pines(SERVICIO_MODELO_DIR / "requirements.txt")
    for paquete in ("scikit-learn", "numpy", "pandas", "scipy", "skops"):
        assert servidor[paquete] == modelo[paquete], paquete


def test_mlflow_sin_ejecutor_de_trabajos():
    entorno = SERVICIOS["mlflow"]["environment"]
    assert "false" in entorno["MLFLOW_SERVER_ENABLE_JOB_EXECUTION"]
    comando = SERVICIOS["mlflow"]["command"]
    assert "--allowed-hosts" in comando and "mlflow" in comando[comando.index("--allowed-hosts") + 1]


# ---------------------------------------------------------------- prueba de carga

sys.path.insert(0, str(PROJ_ROOT / "carga"))
peticiones = _modulo(PROJ_ROOT / "carga" / "peticiones.py", "peticiones")


def test_peticiones_validas_para_la_api():
    for p in peticiones.generar(500):
        EntradaP1.model_validate(p)


def test_peticiones_reproducibles_y_variadas():
    a, b = peticiones.generar(200), peticiones.generar(200)
    assert a == b
    assert len({p["estacion_id"] for p in a}) > 100
    assert len({p["lag_1"] for p in a}) > 30


def test_peticion_invalida_es_rechazada():
    with pytest.raises(ValidationError):
        EntradaP1.model_validate(peticiones.invalida(peticiones.generar(1)[0]))


def test_escalones_de_la_forma_de_carga():
    assert peticiones.escalones("5:60, 10:30") == [(5, 60), (10, 30)]
    defecto = re.search(r'ESCALONES_DEFECTO = "([^"]+)"',
                        (PROJ_ROOT / "carga" / "locustfile.py").read_text(encoding="utf-8"))[1]
    plan = peticiones.escalones(defecto)
    assert [u for u, _ in plan] == sorted(u for u, _ in plan)  # escalones crecientes
    assert defecto in SERVICIOS["locust"]["environment"]["ESCALONES"]


def _corrida_sintetica(carpeta: Path, nombre="base", rps_por_usuario=1.0, ms_base=10.0):
    """Tres escalones de 30 s (10, 50, 100 usuarios); la latencia crece al saturar en 100."""
    carpeta.mkdir(parents=True)
    t0 = 1_700_000_000.0
    historial, filas, recursos = [], [], []
    for i, usuarios in enumerate((10, 50, 100)):
        inicio = t0 + 30 * i
        for s in range(30):
            historial.append({"Timestamp": inicio + s, "User Count": usuarios, "Type": "",
                              "Name": "Aggregated"})
            recursos.append({"Name": "mibici-equipo32-api-1", "ts": inicio + s,
                             "CPUPerc": f"{min(100, usuarios)}%", "MemUsage": "200MiB / 1GiB"})
            for k in range(int(usuarios * rps_por_usuario)):
                ms = ms_base * (40 if usuarios == 100 else 1) + k % 7
                nombre_ep = "/predict/lote [100]" if k == 0 else "/predict"
                filas.append((inicio + s + k / 1000, "POST", nombre_ep, ms, 1))
    for s in range(5):  # reposo antes de la carga
        recursos.append({"Name": "mibici-equipo32-redis-1", "ts": t0 - 10 + s,
                         "CPUPerc": "0.2%", "MemUsage": "6MiB / 256MiB"})
    import pandas as pd

    pd.DataFrame(historial).to_csv(carpeta / "locust_stats_history.csv", index=False)
    with gzip.open(carpeta / "peticiones.csv.gz", "wt", encoding="utf-8") as f:
        f.write("inicio,tipo,nombre,ms,ok\n")
        f.writelines(f"{a},{b},{c},{d},{e}\n" for a, b, c, d, e in filas)
    (carpeta / "recursos.jsonl").write_text("\n".join(json.dumps(r) for r in recursos))
    (carpeta / "entorno.json").write_text(json.dumps({
        "nombre": nombre, "escenario": "escalones", "slo_p95_ms": 300,
        "api": {"workers": 1, "cpus": 1.0, "memoria": "1g"}}))


def test_graficas_resume_escalones_y_capacidad(tmp_path):
    pytest.importorskip("matplotlib")
    graficas = _modulo(PROJ_ROOT / "carga" / "graficas.py", "graficas_prueba")
    carpeta = tmp_path / "reports" / "carga" / "base"
    _corrida_sintetica(carpeta)
    assert graficas.main(carpeta) == 0
    resumen = json.loads((carpeta / "resumen.json").read_text())
    escalones = {e["usuarios"]: e for e in resumen["escalones"]}
    assert set(escalones) == {10, 50, 100}
    assert escalones[50]["rps"] == pytest.approx(50, rel=0.05)
    assert escalones[100]["p95_ms"] > 300  # saturado: no cumple el SLO
    assert resumen["capacidad"]["usuarios"] == 50  # el mayor escalón que cumple
    assert resumen["cpu_s_por_inferencia"] is not None
    for figura in ("01_desempeno.png", "02_capacidad.png", "03_recursos.png", "04_endpoints.png"):
        assert (carpeta / figura).stat().st_size > 10_000, figura


def test_costo_por_inferencia(tmp_path):
    pytest.importorskip("matplotlib")
    graficas = _modulo(PROJ_ROOT / "carga" / "graficas.py", "graficas_prueba2")
    finops = _modulo(PROJ_ROOT / "finops" / "costo_inferencia.py", "costo_prueba")
    _corrida_sintetica(tmp_path / "reports" / "carga" / "base")
    graficas.main(tmp_path / "reports" / "carga" / "base")
    (tmp_path / "finops").mkdir()
    shutil.copy(PROJ_ROOT / "finops" / "precios_referencia.toml", tmp_path / "finops")
    finops.RAIZ, finops.CARGA = tmp_path, tmp_path / "reports" / "carga"
    finops.SALIDA = tmp_path / "reports" / "finops"
    assert finops.main() == 0
    costos = json.loads((finops.SALIDA / "costos.json").read_text())
    rps = costos["medido"]["capacidad_rps"]
    hora = costos["precio_hora_replica"]["Fargate x86"]
    assert hora == pytest.approx(0.04048 + 2 * 0.004445, rel=1e-3)  # 1 vCPU exige 2 GB
    esperado = 1e6 * hora / (rps * 0.6 * 3600)
    assert costos["usd_por_millon_en_linea"]["Fargate x86"] == pytest.approx(esperado, rel=1e-3)
    assert (finops.SALIDA / "05_costo_por_volumen.png").exists()
    assert {e["escenario"] for e in costos["escenarios"]} >= {"App: 1 M peticiones/mes"}


# ---------------------------------------------------------------- registro en MLflow


@pytest.mark.skipif(not (SERVICIO_MODELO_DIR / "MLmodel").is_file(), reason="falta models/servicio")
def test_registrar_modelo_es_idempotente(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    registrar = _modulo(PROJ_ROOT / "docker" / "mlflow" / "registrar_modelo.py", "registrar_prueba")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setattr(registrar, "TRACKING", uri)
    monkeypatch.setattr(registrar, "MODELO_DIR", SERVICIO_MODELO_DIR)
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("mibici-equipo32", artifact_location=(tmp_path / "art").as_uri())
    assert registrar.main() == 0
    assert registrar.main() == 0  # segunda vez: no duplica
    cliente = mlflow.MlflowClient(uri)
    versiones = cliente.search_model_versions("name = 'mibici-equipo32-p1-demanda'")
    assert len(versiones) == 1
    mv = cliente.get_model_version_by_alias("mibici-equipo32-p1-demanda", "champion")
    registro = json.loads((SERVICIO_MODELO_DIR / "registro.json").read_text(encoding="utf-8"))
    assert mv.tags["version_origen"] == str(registro["version"]) and mv.tags["git_commit"]
    # La API servida desde ese registry declara la versión del registry del proyecto
    from api.config import Ajustes
    from api.modelo import cargar_modelo

    servicio = cargar_modelo(Ajustes(modelo_uri="models:/mibici-equipo32-p1-demanda@champion",
                                     tracking_uri=uri, verificar_integridad=True))
    assert servicio.info.version == 1 and servicio.info.version_origen == registro["version"]
    assert servicio.info.integridad_verificada
