# Guion del video de evidencia (TC5061 · Proyecto final · Equipo 32)

**Archivo:** `TC5061_Proyecto_Final_Equipo32.mp4` · **Duración objetivo:** 4 min 30 s (la rúbrica pide de 2 a 5 min) ·
**Toma única, sin cortes.** Presenta: Leydy Johanna Camargo Medina (A01840540), entrega individual.

## Antes de grabar (fuera del video, ~5 min)

1. Docker Desktop abierto. En la carpeta del repositorio, con PowerShell:
   `$env:MLFLOW_PUERTO="5050"; docker compose up -d --build --wait` (el 5000 lo usa el MLflow del proyecto).
2. Servidor de MLflow del proyecto: `powershell -ExecutionPolicy Bypass -File scripts\servidor_mlflow.ps1`
   (deja esa ventana minimizada).
3. Comprueba que todo responde: `curl.exe -s localhost:8000/health`.
4. Abre y deja listas, en este orden: la presentación `TC5061_Proyecto_Final_Equipo32.pdf` en pantalla completa,
   una ventana de PowerShell en la carpeta del repositorio (letra grande: Ctrl + rueda del mouse) y el navegador con
   la pestaña de `reports/monitoreo/drift_datos.html` (reporte de Evidently) abierta.
5. Grabadora: Recortes de Windows (**Win + Shift + R**, pantalla completa, con micrófono) u OBS Studio. Haz una
   prueba de audio de 10 segundos.
6. Cierra notificaciones (Win + N → No molestar).

## Guion

| Tiempo | En pantalla | Qué decir |
|---|---|---|
| 0:00–0:20 | Diapositiva 1 (portada) | «Hola, soy Leydy Johanna Camargo Medina, matrícula A01840540. Presento el proyecto final de Operaciones de aprendizaje automático: un sistema de machine learning de extremo a extremo que pronostica la demanda de MiBici, la bicicleta pública de Guadalajara.» |
| 0:20–0:45 | Diapositiva 2 (problema) | «Cada noche, el área de operación decide a qué estaciones llevar bicicletas. La pregunta es cuántos viajes se iniciarán mañana en cada estación. Usamos los datos abiertos de MiBici: 4.5 millones de viajes en 2025 y una red que creció a 472 estaciones. El modelo en producción se equivoca en 5.7 viajes por estación y día.» |
| 0:45–1:10 | Diapositiva 3 (arquitectura) | «Esta es la solución completa: datos versionados con SHA-256, un MLflow Project con 10 puntos de entrada, MLflow Tracking y el Model Registry. La versión en producción se exporta verificada, GitHub Actions la prueba y publica, y Docker Compose levanta la API. El monitoreo cierra el ciclo de Continuous Training.» |
| 1:10–1:30 | Diapositivas 4 y 5 | «El repositorio sigue Cookiecutter Data Science y el feature engineering usa solo información del día anterior. Elegimos un Poisson lineal con la regla de un error estándar. En el Model Registry, la v3 es @champion, la v1 queda como respaldo y cada versión guarda su linaje y su estado de producción.» |
| 1:30–2:00 | Diapositivas 6 y 7 | «Evidently detectó drift en 10 de 14 columnas y la v1 se degradó. Reentrenamos con datos hasta mayo de 2026 y un A/B pareado mostró que la v3 reduce el error 0.2 viajes, con un intervalo de confianza completamente bajo cero. La promoción pasó sus cuatro compuertas y movió @champion a la v3.» |
| 2:00–2:20 | Diapositivas 8 y 9 | «Todo se levanta con un comando. El pipeline tiene seis jobs, incluida una prueba de carga con Locust. Una réplica de 1 vCPU atiende 22 peticiones por segundo, 89 con el motor en NumPy, a 26 centavos de dólar por millón de peticiones.» |
| 2:20–2:35 | Diapositivas 10 y 11 | «La Model Card documenta el uso previsto, las limitaciones y las consideraciones éticas, como la minimización de datos personales. Estas son las decisiones de MLOps y por qué las tomamos.» |
| 2:35–2:45 | Cambia a PowerShell | «Ahora, la demo en mi computadora.» Ejecuta: `powershell -ExecutionPolicy Bypass -File scripts\demo_final.ps1` |
| 2:45–3:00 | Paso 1: `docker compose ps` | «Los cinco servicios están arriba: MLflow, el registro del modelo, la API, Redis y el worker.» Enter. |
| 3:00–3:15 | Pasos 2 y 3: `/health` y `/modelo` | «La API está sana y sirve la versión 3 del registry del proyecto, con su corrida de origen, commit, versión de datos e integridad verificada.» Enter. |
| 3:15–3:45 | Paso 4: `POST /predict` | «Esta es una petición real: la estación 286, la de demanda mediana del 30 de septiembre de 2026, con sus 12 variables calculadas de los datos abiertos. La API predice 28 viajes y ese día se iniciaron 31. También avisa que dias_activa está fuera del rango de entrenamiento: el modelo indica cuándo extrapola.» Enter. |
| 3:45–3:55 | Paso 5: entrada inválida | «Si la entrada es inconsistente, la API responde 422 y explica por qué.» Enter. |
| 3:55–4:20 | Navegador: Swagger, MLflow y GitHub Actions (se abren solos) | «Esta es la documentación de la API. En MLflow se ve el modelo con @champion en la v3 y @anterior en la v1. Y en GitHub Actions, el pipeline en verde.» |
| 4:20–4:40 | Diapositiva 12 (demo y conclusiones) | «En resumen: un sistema reproducible, desplegado, monitoreado y gobernado. Los siguientes pasos son agregar clima y eventos y automatizar el reentrenamiento mensual. Gracias.» |

## Después de grabar

* Comprueba la duración (entre 2:00 y 5:00) y que se escuche bien.
* Guarda como `TC5061_Proyecto_Final_Equipo32.mp4`.
* Sube a Canvas: el enlace al repositorio, `TC5061_Proyecto_Final_Equipo32.pdf` y el video (o el enlace al video dentro
  de la presentación).
