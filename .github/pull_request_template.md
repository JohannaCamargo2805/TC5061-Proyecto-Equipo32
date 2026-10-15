## ¿Qué cambia?

<!-- Describe brevemente el cambio y la pregunta que atiende (P1, P2, P3 o infraestructura). -->

## Tipo de cambio (Conventional Commits)

- [ ] feat · [ ] fix · [ ] docs · [ ] refactor · [ ] test · [ ] chore · [ ] data · [ ] exp

## Lista de verificación

- [ ] El título del PR sigue Conventional Commits (`tipo(alcance): descripción`).
- [ ] Rutas, fechas y semillas nuevas están en `mibici/config.py` (no repetidas en otros archivos).
- [ ] No agrego datos de MiBici, `mlruns/`, `mlartifacts/`, `mlflow.db`, modelos ni credenciales.
- [ ] `python -m pytest tests` pasa en mi máquina.
- [ ] Si cambié la ingesta, actualicé `references/data_manifest.json` (`python -m mibici.dataset ingest`).
- [ ] Si cambié variables o modelos, ninguna variable usa información posterior al momento de la predicción.

## Evidencia

<!-- Run de MLflow, captura, salida de pruebas o comando para revisar. -->
