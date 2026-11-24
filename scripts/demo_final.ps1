# Demo del proyecto final (video de evidencia, TC5061 - Equipo 32).
#
# Recorre, con una pausa entre pasos, lo que se muestra en el video: los servicios del Compose, la
# salud de la API, la versión del modelo con su linaje, una predicción con datos reales, una entrada
# inválida (422) y, al final, abre Swagger, el Model Registry de MLflow y GitHub Actions.
#
# Antes de grabar (fuera del video):
#   docker compose up -d --build --wait        # con MLFLOW_PUERTO=5050 si el MLflow del proyecto usa el 5000
#   powershell -ExecutionPolicy Bypass -File scripts\servidor_mlflow.ps1     # registry del proyecto (5000)
# Durante el video:
#   powershell -ExecutionPolicy Bypass -File scripts\demo_final.ps1
param(
    [string]$Api = "http://localhost:8000",
    [string]$MlflowProyecto = "http://127.0.0.1:5000",
    [string]$Peticion = "docs\ejemplos\peticion_demo.json",
    [int]$ViajesReales = 31,    # viajes iniciados en la estación 286 el 30-sep-2026 (scripts\peticion_real.py)
    [switch]$SinPausas,
    [switch]$SinNavegador
)
$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent $PSScriptRoot
Set-Location $raiz
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Paso([string]$titulo, [string]$comando) {
    Write-Host ""
    Write-Host "== $titulo" -ForegroundColor Cyan
    if ($comando) { Write-Host "> $comando" -ForegroundColor DarkGray }
}
function Pausa {
    if (-not $SinPausas) { [void](Read-Host "   (Enter para continuar)") }
}

Clear-Host
Write-Host "TC5061 · Proyecto final · Pronóstico de demanda de MiBici (P1)" -ForegroundColor Green
Write-Host "Leydy Johanna Camargo Medina (A01840540) · $(Get-Date -Format 'yyyy-MM-dd HH:mm')"

Paso "1. Sistema levantado con un solo comando" "docker compose ps"
docker compose ps --format "table {{.Service}}`t{{.Image}}`t{{.Status}}"
Pausa

Paso "2. Salud de la API" "GET $Api/health"
Invoke-RestMethod "$Api/health" | ConvertTo-Json
Pausa

Paso "3. Modelo que se sirve y su linaje" "GET $Api/modelo"
$m = Invoke-RestMethod "$Api/modelo"
[pscustomobject]@{
    modelo              = $m.nombre
    version_compose     = $m.version
    version_proyecto    = $m.version_origen
    alias               = ($m.alias -join ", ")
    commit              = $m.git_commit.Substring(0, 7)
    data_version        = $m.data_version
    test_mae            = $m.metricas.test_mae
    integridad_sha256   = $m.integridad_verificada
    motor               = $m.motor_inferencia
} | Format-List
Pausa

Paso "4. Predicción con datos reales: estación 286 (demanda mediana de ese día), 30 de septiembre de 2026" "POST $Api/predict  ($Peticion)"
Get-Content $Peticion -Raw | Write-Host -ForegroundColor DarkGray
$r = Invoke-WebRequest -Method Post -Uri "$Api/predict" -ContentType "application/json" -InFile $Peticion -UseBasicParsing
# La API responde en UTF-8; Windows PowerShell 5.1 supondría ISO-8859-1 sin el charset.
$p = [System.Text.Encoding]::UTF8.GetString($r.RawContentStream.ToArray()) | ConvertFrom-Json
Write-Host ""
Write-Host ("   Viajes esperados : {0}  (redondeado {1})" -f $p.viajes_esperados, $p.viajes_esperados_redondeado) -ForegroundColor Yellow
Write-Host ("   Viajes reales    : {0}" -f $ViajesReales) -ForegroundColor Yellow
Write-Host ("   Error            : {0:N1} viajes ({1:P1})" -f ($p.viajes_esperados - $ViajesReales), (($p.viajes_esperados - $ViajesReales) / $ViajesReales))
Write-Host ("   Modelo           : {0} v{1} del proyecto (v{2} en el registry del Compose) · {3} ms" -f $p.modelo.nombre, $m.version_origen, $p.modelo.version, $r.Headers["X-Tiempo-Respuesta-ms"])
if ($p.advertencias) { $p.advertencias | ForEach-Object { Write-Host "   Advertencia: $_" -ForegroundColor DarkYellow } }
Pausa

Paso "5. Entrada inválida: la API responde 422 y explica por qué" "POST $Api/predict  (docs\ejemplos\peticion_invalida.json)"
try {
    Invoke-WebRequest -Method Post -Uri "$Api/predict" -ContentType "application/json" -InFile "docs\ejemplos\peticion_invalida.json" -UseBasicParsing | Out-Null
} catch {
    $codigo = [int]$_.Exception.Response.StatusCode
    Write-Host "   HTTP $codigo" -ForegroundColor Yellow
    if ($_.ErrorDetails.Message) {
        ($_.ErrorDetails.Message | ConvertFrom-Json).detail | Select-Object -First 3 | ForEach-Object { Write-Host ("   {0}: {1}" -f ($_.loc -join "."), $_.msg) }
    }
}
Pausa

if (-not $SinNavegador) {
    Paso "6. Swagger, Model Registry y GitHub Actions" ""
    Start-Process "$Api/docs"
    Start-Process "$MlflowProyecto/#/models/mibici-equipo32-p1-demanda"
    Start-Process "https://github.com/JohannaCamargo2805/TC5061-Proyecto-Equipo32/actions"
}
Write-Host ""
Write-Host "Fin de la demo." -ForegroundColor Green
