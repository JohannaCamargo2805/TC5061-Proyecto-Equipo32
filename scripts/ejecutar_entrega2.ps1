# Protocolo completo de la entrega 2 (MLflow Project) en Windows.
# Todas las corridas se lanzan con `mlflow run` contra el servidor local de tracking.
# Uso (desde la raiz del repositorio):
#   powershell -ExecutionPolicy Bypass -File scripts\ejecutar_entrega2.ps1
# Requisitos: .venv con requirements.txt instalado y datos de MiBici descargables
# (o una copia local en $env:MIBICI_RAW_SOURCE).
param([string]$Log = "")
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
if (-not $Log) { $Log = Join-Path $repo "reports\ejecucion_entrega2.log" }
New-Item -ItemType Directory -Force (Split-Path $Log -Parent) | Out-Null
"" | Out-File -Encoding utf8 $Log

# Entorno: el Python del proyecto (.venv) y el servidor de tracking.
$venv = Join-Path $repo ".venv\Scripts"
if (Test-Path $venv) { $env:PATH = "$venv;$env:PATH"; $env:VIRTUAL_ENV = (Split-Path $venv -Parent) }
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTHONWARNINGS = "ignore"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:MLFLOW_EXPERIMENT_NAME = "mibici-equipo32"

function Linea($t) { Write-Host $t; $t | Out-File -Encoding utf8 -Append $Log }
function Paso($t) { Write-Host ""; Write-Host "=== $t ===" -ForegroundColor Cyan; "" | Out-File -Encoding utf8 -Append $Log; "=== $t | $(Get-Date -Format 'HH:mm:ss') ===" | Out-File -Encoding utf8 -Append $Log }
function Ejecutar([string]$titulo, [scriptblock]$cmd) {
    Linea ">> $titulo"
    $t0 = Get-Date
    & $cmd *>&1 | ForEach-Object { "$_" } | Where-Object { $_ -notmatch "Downloading artifacts|UserWarning|warnings.warn|^\s*$" } | ForEach-Object { Write-Host $_; $_ | Out-File -Encoding utf8 -Append $Log }
    $ok = ($LASTEXITCODE -eq 0)
    Linea ("<< {0} | {1} | {2:N0} s" -f $titulo, $(if ($ok) { "OK" } else { "ERROR ($LASTEXITCODE)" }), ((Get-Date) - $t0).TotalSeconds)
    return $ok
}
function MlflowRun([string]$ep, [string[]]$params) {
    $argumentos = @("run", ".", "-e", $ep, "--env-manager", "local")
    foreach ($p in $params) { $argumentos += @("-P", $p) }
    return (Ejecutar ("mlflow " + ($argumentos -join " ")) { & mlflow @argumentos })
}

Linea "Entrega 2 (MLflow Project) - Equipo 32 - TC5061 | inicio: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Linea "Equipo: $env:COMPUTERNAME | Usuario: $env:USERNAME | Windows: $((Get-CimInstance Win32_OperatingSystem).Caption)"

Paso "1/9 Repositorio, rama y commit (las corridas se lanzan desde un arbol limpio)"
Ejecutar "git" { git branch --show-current; git log --oneline -1; git remote get-url origin; git status --short --untracked-files=no } | Out-Null

Paso "2/9 Python y librerias"
Ejecutar "versiones" { python -c "import sys, platform, pandas, numpy, sklearn, mlflow; print('Python', sys.version.split()[0], '|', platform.platform()); print('pandas', pandas.__version__, '| numpy', numpy.__version__, '| scikit-learn', sklearn.__version__, '| mlflow', mlflow.__version__)" } | Out-Null

Paso "3/9 Servidor de tracking http://127.0.0.1:5000"
$vivo = $false
try { $vivo = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "http://127.0.0.1:5000/health").StatusCode -eq 200 } catch {}
if (-not $vivo) {
    Linea "Levantando el servidor en otra ventana (scripts\servidor_mlflow.ps1)..."
    Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $repo "scripts\servidor_mlflow.ps1") -WorkingDirectory $repo
    for ($i = 0; $i -lt 40 -and -not $vivo; $i++) { Start-Sleep -Seconds 3; try { $vivo = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "http://127.0.0.1:5000/health").StatusCode -eq 200 } catch {} }
}
Linea "Servidor activo: $vivo | MLFLOW_TRACKING_URI=$env:MLFLOW_TRACKING_URI"
if (-not $vivo) { Linea "No se pudo levantar el servidor; se detiene."; exit 1 }

Paso "4/9 Pruebas (pytest)"
Ejecutar "pytest" { python -m pytest -q tests } | Out-Null

Paso "5/9 Ingesta reproducible: mlflow run . -e ingest"
Ejecutar "archivar corridas fallidas o con arbol sucio (borrado logico)" { python scripts\limpiar_corridas_fallidas.py } | Out-Null
$raw = if ($env:MIBICI_RAW_SOURCE) { "raw_source=$env:MIBICI_RAW_SOURCE" } else { "raw_source=config" }
MlflowRun "ingest" @($raw) | Out-Null
Ejecutar "estado de git tras la ingesta" { git status --short --untracked-files=no } | Out-Null
Ejecutar "etiquetar corridas de la propuesta" { python scripts\etiquetar_corridas_propuesta.py } | Out-Null

Paso "6/9 P1 - regresion de viajes por estacion"
MlflowRun "train" @("question=p1", "model=naive", "stage=baseline") | Out-Null
MlflowRun "train" @("question=p1", "model=poisson") | Out-Null
MlflowRun "train" @("question=p1", "model=hgb") | Out-Null
MlflowRun "train" @("question=p1", "model=rf", "n_estimators=200") | Out-Null
MlflowRun "search" @("question=p1") | Out-Null
MlflowRun "train" @("question=p1", "from_search=latest") | Out-Null

Paso "7/9 P2 - clasificacion de riesgo de disponibilidad"
MlflowRun "train" @("question=p2", "model=logreg", "stage=baseline") | Out-Null
MlflowRun "train" @("question=p2", "model=hgb") | Out-Null
MlflowRun "train" @("question=p2", "model=rf") | Out-Null
MlflowRun "search" @("question=p2") | Out-Null
MlflowRun "train" @("question=p2", "from_search=latest") | Out-Null

Paso "8/9 P3 - clustering de estaciones"
MlflowRun "train" @("question=p3", "stage=baseline") | Out-Null
MlflowRun "train" @("question=p3", "model=kmeans") | Out-Null
MlflowRun "train" @("question=p3", "model=agglomerative") | Out-Null
MlflowRun "train" @("question=p3", "model=gmm") | Out-Null
MlflowRun "search" @("question=p3") | Out-Null

Paso "9/9 Model Registry (aliases champion/challenger) y verificacion de carga por alias"
MlflowRun "register" @("question=p1") | Out-Null
MlflowRun "register" @("question=p2") | Out-Null
MlflowRun "register" @("question=p3") | Out-Null
MlflowRun "verify" @("question=all", "alias=champion") | Out-Null

Linea "Fin: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""
Write-Host "LISTO. Interfaz de MLflow: http://127.0.0.1:5000 (deja abierta la ventana del servidor)." -ForegroundColor Green
