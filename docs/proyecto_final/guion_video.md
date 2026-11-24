# Guion y producción del video de evidencia (TC5061 · Proyecto final · Equipo 32)

**Archivo:** `TC5061_Proyecto_Final_Equipo32.mp4` · **Duración:** 4 min 26 s (la rúbrica pide de 2 a 5 min) ·
1920 × 1080, 30 cuadros/s · **Una sola toma de pantalla, sin cortes.** Entrega individual de Leydy Johanna Camargo
Medina (A01840540).

## Declaración: cómo se produjo

* **Pantalla: una toma continua.** `video/grabar_toma.ps1` grabó el escritorio completo de la computadora de la autora
  con ffmpeg (Desktop Duplication, `ddagrab`, 3840 × 2160 → 1920 × 1080) de principio a fin, sin pausas ni cortes:
  láminas 1 a 11 en pantalla completa, la demo en vivo, el navegador y las láminas 12 y 13.
* **La demo es real.** `scripts/demo_final.ps1` corrió en una consola contra la API del sistema levantado con
  `docker compose up` (imagen `modelo-v3`): cada respuesta que se ve es la que dio la API en ese momento. En el modo de
  toma sin intervención (`-PausaSegundos`, `-Bitacora`, `-EscalaFuente`) la demo espera lo que dura cada segmento de
  la narración en lugar de esperar Enter, anota la hora en que aparece cada resultado y agranda la letra.
* **Navegador.** Swagger UI, el Model Registry del proyecto (servidor de MLflow local) y GitHub Actions se abrieron en
  ventanas InPrivate de Edge justo antes de grabar, para que ya estuvieran cargadas, y durante la toma se trajeron al
  frente una por una (sin cuenta iniciada ni pestañas personales).
* **Narración con voz sintética.** La voz es de Microsoft Edge TTS (`es-MX-DaliaNeural`, velocidad +8 %), generada con
  `video/voz.py` a partir del texto de `video/narracion.json`. El texto se redactó con apoyo de Claude (Anthropic) a
  partir de los resultados del proyecto y se verificó contra la evidencia (MLflow, `reports/`, la API). La portada del
  video lo declara.
* **Montaje sin edición del video.** `video/montar.py` coloca cada segmento de voz en la hora que anotó la bitácora de
  la toma (`video/bitacora_toma.txt` y `video/bitacora_demo.txt`), con el primer cuadro de la lámina 1 como ancla y una
  segunda ancla en la lámina 12 para medir la deriva (39 ms en toda la toma). El video no se corta, no
  se reordena ni se acelera; solo se recortan 2.0 s de pantalla negra al inicio.
  `video/linea_de_tiempo.json` guarda dónde empieza cada segmento.

## Guion (tiempos del video final)

| Inicio | En pantalla | Narración |
|---|---|---|
| 0:01.6 | Lámina 1 · portada (con la declaración de la voz sintética) | Hola. Este es el proyecto final de Leydy Johanna Camargo Medina para Operaciones de aprendizaje automático: un sistema de machine learning de extremo a extremo que pronostica la demanda de MiBici, la bicicleta pública de Guadalajara. Narra una voz sintética; la demostración es en vivo, en una sola toma. |
| 0:20.8 | Lámina 2 · el problema | El problema: el área de operación necesita saber, desde la noche anterior, cuántos viajes se iniciarán mañana en cada estación, para planear a dónde llevar bicicletas. Con cuatro punto cinco millones de viajes de datos abiertos y una red que creció a cuatrocientas setenta y dos estaciones, el modelo en producción se equivoca en cinco punto siete viajes por estación y día. |
| 0:42.1 | Lámina 3 · arquitectura de extremo a extremo | La solución cubre el ciclo completo de MLOps: datos versionados con SHA doscientos cincuenta y seis, un MLflow Project con diez puntos de entrada, MLflow Tracking y el Model Registry. La versión en producción se exporta verificada, GitHub Actions la prueba y la publica, Docker Compose levanta la API, y el monitoreo cierra el ciclo de entrenamiento continuo. |
| 1:03.3 | Lámina 4 · estructura y experimentación | El repositorio sigue la estructura de Cookiecutter Data Science, y las variables usan solo información del cierre del día anterior, sin fuga de datos. Se eligió una regresión de Poisson con la regla de un error estándar: queda a menos de un error del mejor modelo y es mucho más simple. |
| 1:20.2 | Lámina 5 · Model Registry con linaje | En el Model Registry, la versión tres tiene el alias champion, y la versión uno queda como anterior, para revertir en un solo paso. Cada versión guarda su linaje: corrida, commit y versión de datos, y una etiqueta con su estado de producción. |
| 1:35.9 | Lámina 6 · drift con Evidently | Evidently detectó drift en diez de catorce columnas: la red pasó de trescientas sesenta y nueve a cuatrocientas setenta y dos estaciones, y el modelo original empezó a subestimar la demanda. |
| 1:47.3 | Lámina 7 · A/B pareado y promoción de la v3 | Se reentrenó con datos hasta mayo de dos mil veintiséis. Un A B pareado mostró que la versión tres reduce el error en dos décimas de viaje, con un intervalo de confianza completamente debajo de cero. La promoción pasó sus cuatro compuertas y movió champion a la versión tres. |
| 2:04.3 | Lámina 8 · Docker Compose y CI/CD | Todo el sistema se levanta con un solo comando de Docker Compose, y el pipeline de integración y despliegue continuos tiene seis jobs, desde la calidad del código hasta el despliegue. |
| 2:14.8 | Lámina 9 · Locust y FinOps | Con Locust, una réplica de un CPU virtual atiende veintidós peticiones por segundo, u ochenta y nueve con el motor en NumPy: un dólar por millón de peticiones en Fargate, o veintiséis centavos con NumPy. |
| 2:28.5 | Lámina 10 · Model Card | La Model Card documenta el uso previsto, lo que queda fuera de alcance, las limitaciones y las consideraciones éticas, como la minimización de datos personales. |
| 2:38.7 | Lámina 11 · decisiones de MLOps | Y estas son las decisiones de MLOps, cada una con la alternativa que se descartó. Ahora, la demostración en vivo. |
| 2:49.9 | Consola, en vivo · paso 1: `docker compose ps -a` | Primero, el sistema levantado con Docker Compose: MLflow, la API, Redis y el worker están sanos, y el servicio que registra el modelo terminó sin errores. |
| 3:03.3 | Consola, en vivo · paso 2: `GET /health` | La API responde que está sana y que carga el modelo desde el registry. |
| 3:08.7 | Consola, en vivo · paso 3: `GET /modelo` | Se sirve la versión tres del registry del proyecto, versión dos en el MLflow del Compose, con su commit, su versión de datos y la integridad verificada. |
| 3:19.0 | Consola, en vivo · paso 4: `POST /predict` con la petición real (`docs/ejemplos/peticion_demo.json`) | Ahora, una petición real: la estación doscientos ochenta y seis, el treinta de septiembre de dos mil veintiséis. La API predice veintiocho viajes, y ese día se iniciaron treinta y uno. Además advierte que días activa está fuera del rango de entrenamiento: el modelo avisa cuando extrapola. |
| 3:38.4 | Consola, en vivo · paso 5: entrada inválida → 422 | Si la entrada es inconsistente, la API responde cuatrocientos veintidós y explica por qué. |
| 3:46.0 | Edge · Swagger UI de la API (`localhost:8000/docs`) | Esta es la documentación interactiva de la API, generada por FastAPI. |
| 3:52.6 | Edge · Model Registry del proyecto en MLflow (`127.0.0.1:5000`) | En el Model Registry de MLflow, la versión tres tiene el alias champion, su estado de producción y su linaje en las etiquetas. |
| 4:02.1 | Edge · GitHub Actions del repositorio | Y en GitHub Actions, el pipeline de integración y despliegue continuos está en verde. |
| 4:08.7 | Lámina 12 · demo y conclusiones (después, lámina 13 · referencias) | En resumen: un sistema reproducible, desplegado, monitoreado y gobernado. Los siguientes pasos: agregar clima y eventos, y disparar el reentrenamiento de forma automática. Gracias. |

## Repetir la grabación

1. Levantar el sistema (`docker compose up -d --build --wait`, con `MLFLOW_PUERTO=5050`) y el servidor de MLflow del
   proyecto (`scripts/servidor_mlflow.ps1`); comprobar `curl.exe -s localhost:8000/health`.
2. Generar la voz: `python docs/proyecto_final/video/voz.py <carpeta> +8%` (requiere `edge-tts` y ffmpeg).
3. Exportar las láminas del PDF a PNG de 3840 × 2160 en `<carpeta>/laminas/l-01.png` … `l-13.png`.
4. Grabar: `powershell -File docs/proyecto_final/video/grabar_toma.ps1 -Base <carpeta> -Repo <repositorio>`
   (no tocar el equipo durante unos 5 minutos).
5. Montar: `python docs/proyecto_final/video/montar.py <carpeta> TC5061_Proyecto_Final_Equipo32.mp4`.
6. Subir a Canvas: el enlace al repositorio, `TC5061_Proyecto_Final_Equipo32.pdf` y el video.
