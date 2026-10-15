# Machine Learning Canvas — Proyecto MiBici (TC5061, Equipo 32)

Plantilla: *Machine Learning Canvas* de Louis Dorard (OWNML). Versión gráfica: `references/ml_canvas.png`.

## Propuesta de valor
*¿Qué, por qué y para quién?*

- **Usuarios**: operador del sistema (BKT/AMIM), equipo de rebalanceo y planeación urbana del AMG.
- **Valor**: menos estaciones vacías o llenas (mejor experiencia y confianza del ciclista), uso más eficiente de flota y vehículos de rebalanceo, y evidencia para decidir dónde crecer.
- **Hoy**: decisiones basadas en experiencia y reportes; MiBici no publica inventario por estación, por lo que el riesgo se infiere de los viajes y la capacidad.

## Tarea de ML
*Entrada, salida y tipo de problema*

- **P1 · Regresión de conteo**: estación × hora de servicio del día D → nº de viajes iniciados (entero ≥ 0). Horizonte: D+1 (se predice al cierre de D-1).
- **P2 · Clasificación binaria**: estación × hora de decisión *t* → riesgo (1/0) de ocupación estimada ≤ 10 % (sin bicis) o ≥ 90 % (sin anclajes) en (t, t+3 h], con probabilidad.
- **P3 · Clustering**: patrón de uso de cada estación (cuándo y en qué dirección se usa) → grupo (k elegido por silueta) y su ubicación en el mapa.

## Decisiones
*¿Cómo se usan las predicciones para decidir?*

- **P1**: asignar bicicletas y personal por estación/zona para el día siguiente (dónde reforzar flota al abrir a las 05:00).
- **P2**: priorizar la ruta de los vehículos de rebalanceo: visitar primero las estaciones con mayor probabilidad de vaciarse o llenarse en 3 h (umbral elegido con F2).
- **P3**: definir estrategias por tipo de estación y priorizar zonas para densificar o expandir el sistema.

## Hacer predicciones
*¿Cuándo y con qué latencia?*

- **P1**: lote diario a la 01:00 (cierre del servicio) para las 20 horas de servicio del día siguiente; minutos de holgura.
- **P2**: cada hora durante el servicio (05:00–00:59) con datos hasta *t*; latencia < 5 min.
- **P3**: recalcular por trimestre o semestre (lote).

## Evaluación offline
*Métodos y métricas antes de desplegar*

- Partición temporal: train oct-24→mar-26, validación abr→jun-26, prueba jul→sep-26 (sin barajar).
- **P1**: MAE, RMSE, WAPE y MAE diario por estación vs. ingenuo estacional (t − 168 h).
- **P2**: PR-AUC, recall, precisión y F2 vs. regla de ocupación actual.
- **P3**: silueta, Davies-Bouldin, Calinski-Harabasz y estabilidad temporal (ARI) vs. zonificación geográfica.

## Fuentes de datos
*¿Qué datos crudos se usan?*

- Portal de datos abiertos MiBici: viajes mensuales en CSV (dic-2014 a sep-2026); se usan **oct-2024 a sep-2026** (24 meses).
- Nomenclatura de estaciones (sep-2026): id, nombre, coordenadas y capacidad.
- Calendario de días de descanso obligatorio (LFT) y Semana Santa.
- *Futuro*: clima (lluvia), eventos masivos, telemetría de anclajes (GBFS).

## Recolección de datos
*¿Cómo se obtienen datos nuevos (entradas y salidas)?*

- Ingesta reproducible `dataset.py`: descarga, normaliza (codificación, columnas, fechas), Parquet y manifiesto con SHA-256 → `data_version`.
- Etiquetas P1: se observan solas (conteo real) al publicarse cada mes.
- Etiquetas P2: aproximadas con balance llegadas − salidas y capacidad (supuesto 50 % al abrir); validar con inventario real cuando exista.

## Variables (features)
*Representaciones extraídas de los datos crudos*

- Calendario: hora, día, fin de semana, feriado, hora operativa.
- P1: rezagos 24/48/168/336 h, promedios misma hora (7 d, 4 sem), total del día previo, media 28 d, perfil histórico de la estación (solo train), capacidad.
- P2: ocupación estimada actual, flujo neto acumulado, salidas/llegadas última hora y 3 h, flujo de la misma ventana ayer y hace una semana, tasa histórica de riesgo (solo train).
- P3: proporción de salidas y llegadas por bloque horario, balance neto y fin de semana (agrupamiento); volumen, duración media y viajes circulares (descriptivas).

## Construir modelos
*¿Cuándo se crean o actualizan?*

- Reentrenamiento mensual al publicarse un mes nuevo (nueva `data_version`).
- scikit-learn: baselines simples + HistGradientBoosting (P1 Poisson, P2 clasificador balanceado) y K-Means (P3).
- Cada run en MLflow con parámetros, métricas, artefactos, versión de datos y commit de Git; < 10 min en una laptop.

## Evaluación en vivo y monitoreo
*Métodos y métricas después del despliegue*

- P1: MAE/WAPE por estación y zona al publicarse el mes real; alerta si el error supera al baseline.
- P2: tasa de alertas atendidas y de estaciones vacías/llenas reportadas; recall mensual.
- Deriva: volumen mensual, estaciones nuevas o dadas de baja, cambios en el perfil horario; estabilidad de grupos (ARI) entre periodos.
