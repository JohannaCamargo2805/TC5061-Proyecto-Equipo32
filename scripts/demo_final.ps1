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
# Para una toma sin intervención (video con narración sincronizada), en lugar de esperar Enter:
#   ... demo_final.ps1 -PausaSegundos "9,5,9,17,6" -Bitacora final_trabajo\video\bitacora_demo.txt -EscalaFuente 1.9
#   espera esos segundos después de cada paso, anota en la bitácora cuándo aparece cada resultado y
#   agranda la letra para que se lea en el video.
param(
    [string]$Api = "http://localhost:8000",
    [string]$MlflowProyecto = "http://127.0.0.1:5000",
    [string]$Peticion = "docs\ejemplos\peticion_demo.json",
    [int]$ViajesReales = 31,    # viajes iniciados en la estación 286 el 30-sep-2026 (scripts\peticion_real.py)
    [switch]$SinPausas,
    [switch]$SinNavegador,
    [string]$PausaSegundos = "",      # "9,5,9,17,6": una pausa por paso (1 a 5) en lugar de Enter
    [string]$Bitacora = "",           # archivo donde se anota la hora (ms Unix) en que aparece cada paso
    [double]$EscalaFuente = 0         # p. ej. 1.9: letra más grande, ventana maximizada y un paso por pantalla (video)
)
$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent $PSScriptRoot
Set-Location $raiz
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Encabezado {
    Write-Host "TC5061 · Proyecto final · Pronóstico de demanda de MiBici (P1)" -ForegroundColor Green
    Write-Host "Leydy Johanna Camargo Medina (A01840540) · $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
}
if ($EscalaFuente -gt 0) {
    # Modo video: letra más grande en la consola de Windows (conhost) y ventana maximizada.
    Add-Type @"
using System; using System.Runtime.InteropServices;
public static class ConsolaDemo {
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
  public struct FUENTE { public uint cbSize; public uint nFont; public short X; public short Y; public int FontFamily;
    public int FontWeight; [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string FaceName; }
  [DllImport("kernel32.dll")] static extern IntPtr GetStdHandle(int n);
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] static extern bool GetCurrentConsoleFontEx(IntPtr h, bool m, ref FUENTE f);
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] static extern bool SetCurrentConsoleFontEx(IntPtr h, bool m, ref FUENTE f);
  [DllImport("kernel32.dll")] static extern IntPtr GetConsoleWindow();
  [DllImport("user32.dll")] static extern bool ShowWindow(IntPtr h, int n);
  public static string Agrandar(double escala) {
    var f = new FUENTE(); f.cbSize = (uint)Marshal.SizeOf(f);
    IntPtr h = GetStdHandle(-11);
    GetCurrentConsoleFontEx(h, false, ref f);
    short antes = f.Y;
    f.Y = (short)Math.Round(f.Y * escala); f.X = 0; f.FontFamily = 54; f.FontWeight = 400; f.FaceName = "Consolas";
    bool ok = SetCurrentConsoleFontEx(h, false, ref f);
    ShowWindow(GetConsoleWindow(), 3);
    return antes + "->" + f.Y + (ok ? "" : " (sin cambio)");
  }
}
"@
    $fuente = [ConsolaDemo]::Agrandar($EscalaFuente)
    $Host.UI.RawUI.WindowTitle = "TC5061 · Proyecto final · Demo en vivo de la API (Docker Compose)"
    if ($Bitacora) { Add-Content -Path $Bitacora -Value "fuente $fuente" }
}

function Paso([string]$titulo, [string]$comando) {
    if ($EscalaFuente -gt 0) { Clear-Host; Encabezado }
    Write-Host ""
    Write-Host "== $titulo" -ForegroundColor Cyan
    if ($comando) { Write-Host "> $comando" -ForegroundColor DarkGray }
}
$script:paso = 0
$pausas = @($PausaSegundos -split "[,;\s]+" | Where-Object { $_ } |
    ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) })
function Pausa {
    $script:paso++
    if ($Bitacora) {
        Add-Content -Path $Bitacora -Value ("paso{0} {1}" -f $script:paso, [DateTimeOffset]::Now.ToUnixTimeMilliseconds())
    }
    if ($pausas.Count -ge $script:paso) { Start-Sleep -Milliseconds ([int](1000 * $pausas[$script:paso - 1])) }
    elseif (-not $SinPausas) { [void](Read-Host "   (Enter para continuar)") }
}

Clear-Host
if ($EscalaFuente -le 0) { Encabezado }

Paso "1. Sistema levantado con un solo comando" "docker compose ps -a"
docker compose ps -a --format "table {{.Service}}`t{{.Image}}`t{{.Status}}"
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
        # Un motivo por línea (el validador junta varios con "; ")
        ($_.ErrorDetails.Message | ConvertFrom-Json).detail | Select-Object -First 3 | ForEach-Object {
            $loc = $_.loc -join "."
            ($_.msg -replace "^Value error, ", "") -split "; " | ForEach-Object { Write-Host ("   {0}: {1}" -f $loc, $_) }
        }
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
