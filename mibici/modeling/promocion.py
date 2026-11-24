"""Proyecto final: promoción a producción de la versión que ganó el A/B (Continuous Training).

Cierra el ciclo que empezó en la entrega 5::

    monitor (Evidently detecta drift) -> retrain (nueva versión @challenger)
        -> abtest (A = @champion contra B = @challenger) -> promote (este módulo)

Uso::

    python -m mibici.modeling.promocion              # decide con la corrida de A/B más reciente
    python -m mibici.modeling.promocion --simular    # muestra las compuertas y el plan, no cambia nada
    python -m mibici.modeling.promocion --rollback   # regresa @champion a la versión @anterior
    mlflow run . -e promote --env-manager local

La versión B solo pasa a producción si se cumplen **todas** las compuertas:

1. Existe una corrida de A/B terminada (``stage_of_work = ab_test``) y su decisión es promover B:
   las cuatro reglas de ``config.py`` (IC 95 % del ΔMAE bajo cero, ningún mes peor por más de 2 %,
   sesgo no peor y costo de inferencia aceptable), fijadas antes de ver el resultado.
2. La versión A de ese A/B sigue siendo ``@champion``: la decisión no está desactualizada.
3. La versión B está ``READY``, es de P1 y su firma de entrada es exactamente la de la API.
4. Si B viene de un reentrenamiento, pasó su compuerta de validación (``pasa_compuerta = 1``).

Al promover: ``@champion`` -> B, ``@anterior`` -> A (rollback en un paso), se quita
``@challenger`` de B, cada versión recibe la etiqueta ``estado_produccion`` (producción, retador,
respaldo o archivada) y se registra la corrida ``p1-promocion-vA-a-vB`` con las compuertas, el
plan y la evidencia del A/B. La API toma la nueva versión al reconstruir su imagen
(``scripts/exportar_modelo_registry.py --alias champion``) o al reiniciar con
``MODELO_URI=models:/...@champion``: un alias se resuelve al arrancar, nunca a mitad del servicio.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import typer

from mibici.config import (
    ALIAS_ANTERIOR,
    ALIAS_PRODUCCION,
    ALIAS_RETADOR,
    ESTADOS_PRODUCCION,
    MLFLOW_EXPERIMENT,
    PROMOCION_RUN_NAME,
    REGISTRY_NAMES,
)
from mibici.features import P1_FEATURES
from mibici.modeling import seguimiento as seg

app = typer.Typer(add_completion=False)
NOMBRE = REGISTRY_NAMES["p1"]
ETIQUETA_ESTADO = "estado_produccion"
# Primera línea de la descripción de cada versión: su estado actual (se reemplaza en cada cambio).
_PREFIJO_ESTADO = re.compile(r"^\[Estado: [^\]]*\]\n\n", re.DOTALL)


class PromocionRechazada(RuntimeError):
    """Alguna compuerta no se cumple: @champion no se mueve."""


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _version_por_alias(client: MlflowClient, nombre: str, alias: str):
    try:
        return client.get_model_version_by_alias(nombre, alias)
    except MlflowException:
        return None


# =============================================================================
# Evidencia: la corrida de A/B
# =============================================================================
def ultima_corrida_ab(client: MlflowClient, experimento: str = MLFLOW_EXPERIMENT):
    """La corrida de A/B terminada más reciente del experimento (o None)."""
    exp = client.get_experiment_by_name(experimento)
    if exp is None:
        return None
    corridas = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.stage_of_work = 'ab_test' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return corridas[0] if corridas else None


def leer_decision(client: MlflowClient, corrida) -> dict:
    """Versiones comparadas, reglas y decisión de una corrida de A/B (artefacto + métricas)."""
    t, m = corrida.data.tags, corrida.data.metrics
    decision = {"run_id": corrida.info.run_id, "run_name": t.get("mlflow.runName", ""),
                "version_A": int(t["version_A"]), "version_B": int(t["version_B"]),
                "ganadora": t.get("ganadora", ""), "promover_B": bool(m.get("promover_B", 0.0)),
                "reglas": {}, "metricas": {}}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(client.download_artifacts(corrida.info.run_id, "ab_test/ab_resultado.json", tmp))
            resultado = json.loads(ruta.read_text(encoding="utf-8"))
        decision["reglas"] = resultado["decision"]["reglas"]
        decision["metricas"] = {
            "mae_A": resultado["metricas"]["A"]["mae"], "mae_B": resultado["metricas"]["B"]["mae"],
            "ic95_delta_mae": resultado["bootstrap"]["ic95_delta_mae"],
            "wilcoxon_p": resultado["wilcoxon_mae_diario"]["p_valor"],
            "evaluacion": resultado["evaluacion"],
        }
    except (MlflowException, OSError, KeyError) as error:  # corridas sin el artefacto
        logger.warning(f"La corrida de A/B no trae ab_test/ab_resultado.json ({error}); "
                       "se usan sus métricas")
        decision["metricas"] = {k: v for k, v in m.items() if k.startswith(("A_mae", "B_mae", "ic95"))}
    return decision


# =============================================================================
# Compuertas
# =============================================================================
def firma(nombre: str, version: int) -> list[str]:
    info = mlflow.models.get_model_info(f"models:/{nombre}/{version}")
    return list(info.signature.inputs.input_names()) if info.signature else []


def evaluar_compuertas(client: MlflowClient, decision: dict, nombre: str = NOMBRE) -> dict:
    """Cada compuerta con su resultado y la razón; ``todas`` es True solo si pasan todas."""
    va, vb = decision["version_A"], decision["version_B"]
    campeon = _version_por_alias(client, nombre, ALIAS_PRODUCCION)
    mv_b = client.get_model_version(nombre, str(vb))
    reglas = decision.get("reglas") or {}
    compuertas = {
        "1_ab_decide_promover_B": {
            "ok": decision["promover_B"] and decision["ganadora"] == "B"
            and (all(reglas.values()) if reglas else True),
            "detalle": f"corrida {decision['run_name']} ({decision['run_id'][:8]}): ganadora "
                       f"{decision['ganadora']}, reglas {sum(map(bool, reglas.values()))}/{len(reglas)}",
        },
        "2_A_sigue_en_produccion": {
            "ok": campeon is not None and int(campeon.version) == va,
            "detalle": f"@{ALIAS_PRODUCCION} = v{campeon.version if campeon else '-'}; el A/B comparó v{va}",
        },
    }
    variables = firma(nombre, vb) if mv_b.status == "READY" else []
    pregunta = (mv_b.tags or {}).get("question", "p1")
    compuertas["3_B_lista_y_con_el_contrato_de_la_API"] = {
        "ok": mv_b.status == "READY" and pregunta == "p1" and variables == list(P1_FEATURES),
        "detalle": f"v{vb} {mv_b.status}, pregunta {pregunta}, {len(variables)} variables de entrada",
    }
    metricas_b = client.get_run(mv_b.run_id).data.metrics
    if "pasa_compuerta" in metricas_b:
        compuertas["4_B_paso_su_compuerta_de_validacion"] = {
            "ok": bool(metricas_b["pasa_compuerta"]),
            "detalle": (f"VAL: v{vb} MAE {metricas_b.get('val_mae_nuevo', float('nan')):.4f} contra "
                        f"{metricas_b.get('val_mae_champion', float('nan')):.4f} del champion"),
        }
    return {"compuertas": compuertas, "todas": all(c["ok"] for c in compuertas.values())}


# =============================================================================
# Estado de producción de cada versión
# =============================================================================
def estado_de(aliases: list[str]) -> str:
    for alias in (ALIAS_PRODUCCION, ALIAS_RETADOR, ALIAS_ANTERIOR):
        if alias in aliases:
            return ESTADOS_PRODUCCION[alias]
    return ESTADOS_PRODUCCION[None]


def descripcion_con_estado(descripcion: str | None, estado: str, aliases: list[str]) -> str:
    """La descripción de la versión con su estado vigente en la primera línea."""
    base = _PREFIJO_ESTADO.sub("", descripcion or "", count=1)
    marca = f"[Estado: {estado}{' · ' + ', '.join('@' + a for a in aliases) if aliases else ''}]"
    return f"{marca}\n\n{base}" if base else marca


def actualizar_estados(client: MlflowClient, nombre: str = NOMBRE) -> dict[int, dict]:
    """Etiqueta ``estado_produccion`` y primera línea de la descripción de cada versión según sus
    aliases (así la interfaz de MLflow muestra el estado vigente); devuelve la tabla."""
    aliases = client.get_registered_model(nombre).aliases or {}
    por_version: dict[int, list[str]] = {}
    for alias, version in aliases.items():
        por_version.setdefault(int(version), []).append(alias)
    tabla = {}
    for mv in client.search_model_versions(f"name = '{nombre}'"):
        v = int(mv.version)
        estado = estado_de(por_version.get(v, []))
        if (mv.tags or {}).get(ETIQUETA_ESTADO) != estado:
            client.set_model_version_tag(nombre, str(v), ETIQUETA_ESTADO, estado)
        nueva = descripcion_con_estado(mv.description, estado, sorted(por_version.get(v, [])))
        if nueva != (mv.description or ""):
            client.update_model_version(nombre, str(v), description=nueva)
        tabla[v] = {"aliases": sorted(por_version.get(v, [])), "estado": estado,
                    "run_id": mv.run_id}
    campeon = aliases.get(ALIAS_PRODUCCION)
    if campeon is not None:
        client.set_registered_model_tag(nombre, "version_produccion", str(campeon))
    return dict(sorted(tabla.items()))


# =============================================================================
# Promoción y rollback
# =============================================================================
def plan(decision: dict) -> list[str]:
    va, vb = decision["version_A"], decision["version_B"]
    return [f"@{ALIAS_PRODUCCION}: v{va} -> v{vb}",
            f"@{ALIAS_ANTERIOR}: -> v{va} (rollback en un paso)",
            f"@{ALIAS_RETADOR}: se quita de v{vb} (ya es producción)",
            f"{ETIQUETA_ESTADO} en todas las versiones"]


def promover(client: MlflowClient, decision: dict, simular: bool = False,
             nombre: str = NOMBRE) -> dict:
    """Mueve @champion a B si pasan las compuertas. Idempotente si B ya es @champion."""
    va, vb = decision["version_A"], decision["version_B"]
    campeon = _version_por_alias(client, nombre, ALIAS_PRODUCCION)
    if campeon is not None and int(campeon.version) == vb:
        logger.info(f"v{vb} ya es @{ALIAS_PRODUCCION}: nada que promover")
        return {"promovida": False, "ya_en_produccion": True, "version_produccion": vb,
                "estados": actualizar_estados(client, nombre) if not simular else {}}
    evaluacion = evaluar_compuertas(client, decision, nombre)
    for clave, c in evaluacion["compuertas"].items():
        (logger.success if c["ok"] else logger.error)(f"{clave}: {'sí' if c['ok'] else 'NO'} · {c['detalle']}")
    resultado = {"decision_ab": decision, **evaluacion, "plan": plan(decision)}
    if not evaluacion["todas"]:
        raise PromocionRechazada(
            "no se promueve: " + ", ".join(k for k, c in evaluacion["compuertas"].items() if not c["ok"]))
    if simular:
        logger.info("Simulación: " + " · ".join(resultado["plan"]))
        return {**resultado, "promovida": False, "simulacion": True}
    fecha = _ahora()
    client.set_registered_model_alias(nombre, ALIAS_PRODUCCION, str(vb))
    client.set_registered_model_alias(nombre, ALIAS_ANTERIOR, str(va))
    retador = _version_por_alias(client, nombre, ALIAS_RETADOR)
    if retador is not None and int(retador.version) == vb:
        client.delete_registered_model_alias(nombre, ALIAS_RETADOR)
    client.set_model_version_tag(nombre, str(vb), "promovida", f"{fecha} por A/B {decision['run_id']}")
    client.set_model_version_tag(nombre, str(va), "retirada_de_produccion",
                                 f"{fecha}: reemplazada por v{vb} (A/B {decision['run_id']})")
    client.set_registered_model_tag(nombre, "ultima_promocion", f"v{va} -> v{vb} el {fecha}")
    estados = actualizar_estados(client, nombre)
    logger.success(f"{nombre}: @{ALIAS_PRODUCCION} -> v{vb}, @{ALIAS_ANTERIOR} -> v{va}")
    return {**resultado, "promovida": True, "fecha": fecha, "version_produccion": vb,
            "estados": estados}


def rollback(client: MlflowClient, nombre: str = NOMBRE) -> dict:
    """Regresa @champion a la versión @anterior (y la que estaba queda como @anterior)."""
    campeon = _version_por_alias(client, nombre, ALIAS_PRODUCCION)
    anterior = _version_por_alias(client, nombre, ALIAS_ANTERIOR)
    if anterior is None or campeon is None:
        raise PromocionRechazada(f"no hay @{ALIAS_ANTERIOR} o @{ALIAS_PRODUCCION} para hacer rollback")
    fecha = _ahora()
    client.set_registered_model_alias(nombre, ALIAS_PRODUCCION, anterior.version)
    client.set_registered_model_alias(nombre, ALIAS_ANTERIOR, campeon.version)
    client.set_model_version_tag(nombre, campeon.version, "retirada_de_produccion", f"{fecha}: rollback")
    client.set_registered_model_tag(nombre, "ultima_promocion",
                                    f"rollback v{campeon.version} -> v{anterior.version} el {fecha}")
    estados = actualizar_estados(client, nombre)
    logger.warning(f"Rollback: @{ALIAS_PRODUCCION} -> v{anterior.version}")
    return {"rollback": True, "fecha": fecha, "version_produccion": int(anterior.version),
            "estados": estados}


def registrar_corrida(resultado: dict) -> str:
    """Corrida de MLflow con la decisión, las compuertas y los estados resultantes."""
    d = resultado["decision_ab"]
    manifiesto = seg.leer_manifiesto()
    etiquetas = seg.etiquetas_base("p1", "registry", "promotion", manifiesto["data_version"])
    etiquetas.update({"entrega": "proyecto-final", "version_A": str(d["version_A"]),
                      "version_B": str(d["version_B"]), "ab_run_id": d["run_id"],
                      "version_produccion": str(resultado["version_produccion"])})
    with seg.corrida(PROMOCION_RUN_NAME.format(a=d["version_A"], b=d["version_B"]), etiquetas) as run:
        mlflow.log_params({"registered_model": NOMBRE, "alias_produccion": ALIAS_PRODUCCION,
                           "alias_respaldo": ALIAS_ANTERIOR, "ab_run": d["run_name"],
                           "de_version": d["version_A"], "a_version": d["version_B"]})
        mlflow.log_metrics({"promovida": 1.0,
                            **{f"compuerta_{k}": float(c["ok"]) for k, c in resultado["compuertas"].items()}})
        mlflow.log_dict(resultado, "promocion/promocion.json")
        run_id = run.info.run_id
    MlflowClient().set_model_version_tag(NOMBRE, str(d["version_B"]), "promocion_run_id", run_id)
    return run_id


ACCIONES = ("promover", "simular", "rollback")


@app.command()
def main(
    ab_run_id: str = typer.Option("auto", help="Corrida de A/B que decide; auto = la más reciente"),
    accion: str = typer.Option("promover", help="promover | simular (solo compuertas y plan) | rollback"),
    simular: bool = typer.Option(False, "--simular", help="Atajo de --accion simular"),
    hacer_rollback: bool = typer.Option(False, "--rollback", help="Atajo de --accion rollback"),
):
    """Promueve a producción (@champion) la versión que ganó el A/B, o hace rollback."""
    accion = "rollback" if hacer_rollback else "simular" if simular else accion.strip().lower()
    if accion not in ACCIONES:
        raise typer.BadParameter(f"accion debe ser una de {', '.join(ACCIONES)}")
    simular = accion == "simular"
    seg.configurar_tracking()
    client = MlflowClient()
    if accion == "rollback":
        logger.success(json.dumps(rollback(client), ensure_ascii=False, default=str))
        return
    corrida = ultima_corrida_ab(client) if seg.es_defecto(ab_run_id) else client.get_run(ab_run_id)
    if corrida is None:
        raise typer.BadParameter("No hay una corrida de A/B terminada: corre primero `-e abtest`")
    decision = leer_decision(client, corrida)
    try:
        resultado = promover(client, decision, simular=simular)
    except PromocionRechazada as error:
        logger.error(str(error))
        raise typer.Exit(code=1) from error
    if resultado.get("promovida"):
        resultado["promocion_run_id"] = registrar_corrida(resultado)
        logger.success(f"Corrida de la promoción: {resultado['promocion_run_id']}")
    for v, e in resultado.get("estados", {}).items():
        logger.info(f"v{v}: {e['estado']:<10} {', '.join('@' + a for a in e['aliases']) or '-'}")


if __name__ == "__main__":
    app()
