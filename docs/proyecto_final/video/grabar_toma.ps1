# Toma única del video del proyecto final (TC5061 · Equipo 32).
#
# Graba la pantalla completa con ffmpeg (Desktop Duplication) en UNA sola toma, sin cortes:
# láminas 1 a 11 en pantalla completa -> demo en vivo (scripts\demo_final.ps1 contra la API del
# Compose) -> Swagger, Model Registry de MLflow y GitHub Actions en el navegador -> láminas 12 y 13.
# Cada escena dura lo que dura su segmento de narración (voz\duraciones.json) y se anota en
# bitacora_toma.txt la hora en que aparece, para colocar después la narración sin mover el video.
param(
    [Parameter(Mandatory = $true)][string]$Base,   # ...\final_trabajo\video
    [Parameter(Mandatory = $true)][string]$Repo,   # ...\TC5061-Equipo32-equipo
    [int]$EsperaInicial = 12,
    [string]$Salida = "toma.mkv"
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
Add-Type @"
using System; using System.Runtime.InteropServices;
public static class W32 {
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, System.Text.StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr despues, int x, int y, int cx, int cy, uint f);
  public delegate bool Enumera(IntPtr h, IntPtr l);
  [DllImport("user32.dll")] public static extern bool EnumWindows(Enumera f, IntPtr l);
  public static string Titulo(IntPtr h) { var sb = new System.Text.StringBuilder(512); GetWindowText(h, sb, 512); return sb.ToString(); }
  public static IntPtr Busca(string a, string b) {
    IntPtr r = IntPtr.Zero;
    EnumWindows(delegate (IntPtr h, IntPtr l) {
      if (!IsWindowVisible(h)) return true;
      string t = Titulo(h);
      if (t.Contains(a) && t.Contains(b)) { r = h; return false; }
      return true;
    }, IntPtr.Zero);
    return r;
  }
  // Pone la ventana encima de todas (o la regresa a su nivel normal) sin enviarle clics ni teclas.
  public static void Encima(IntPtr h, bool si) { SetWindowPos(h, new IntPtr(si ? -1 : -2), 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040); }
}
"@
[void][W32]::SetProcessDPIAware()
$inv = [Globalization.CultureInfo]::InvariantCulture
$bit = Join-Path $Base "bitacora_toma.txt"
$bitDemo = Join-Path $Base "bitacora_demo.txt"
function Ms { [DateTimeOffset]::Now.ToUnixTimeMilliseconds() }
function Marca([string]$evento) { Add-Content -Path $bit -Value ("{0} {1}" -f $evento, (Ms)) -Encoding ASCII }
Set-Content -Path $bit -Value ("preparacion {0}" -f (Ms)) -Encoding ASCII
if (Test-Path $bitDemo) { Remove-Item $bitDemo }

$dur = (Get-Content (Join-Path $Base "voz\duraciones.json") -Raw -Encoding UTF8 | ConvertFrom-Json).duraciones
function D([string]$id) { [double]$dur.$id }

# --- Comprobaciones previas: sin servicios arriba no se graba ---
try {
    $h = Invoke-RestMethod "http://localhost:8000/health" -TimeoutSec 10
    Add-Content $bit ("api_health {0} v{1} origen v{2}" -f $h.estado, $h.version, $h.version_origen)
    $null = Invoke-WebRequest "http://127.0.0.1:5000/" -UseBasicParsing -TimeoutSec 10
    Add-Content $bit "mlflow_proyecto ok"
} catch {
    Add-Content $bit ("ERROR_PREVIO {0}" -f $_.Exception.Message)
    exit 2
}

# Navegador: Microsoft Edge (ventanas InPrivate con un perfil propio: sin cuenta, sin pestañas personales)
$progId = "msedge"
$nav = @("${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe", "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
Add-Content $bit ("navegador {0} | {1}" -f $progId, $nav)

# --- Ventana de láminas: pantalla completa, siempre encima ---
$pantalla = [Windows.Forms.Screen]::PrimaryScreen.Bounds
$form = New-Object Windows.Forms.Form
$form.FormBorderStyle = "None"; $form.StartPosition = "Manual"; $form.Bounds = $pantalla
$form.TopMost = $true; $form.BackColor = [Drawing.Color]::Black; $form.ShowInTaskbar = $false
$pb = New-Object Windows.Forms.PictureBox
$pb.Dock = "Fill"; $pb.SizeMode = "Zoom"; $pb.BackColor = [Drawing.Color]::Black
$form.Controls.Add($pb)
$laminas = @{}
foreach ($i in 1..13) { $laminas[$i] = [Drawing.Image]::FromFile((Join-Path $Base ("laminas\l-{0:D2}.png" -f $i))) }

function Bombea([double]$segundos) {
    $fin = (Ms) + [long](1000 * $segundos)
    while ((Ms) -lt $fin) { [Windows.Forms.Application]::DoEvents(); Start-Sleep -Milliseconds 15 }
}
function Lamina([int]$i, [string]$evento) {
    $pb.Image = $laminas[$i]; $pb.Refresh(); [Windows.Forms.Application]::DoEvents()
    Marca $evento
}
function Cierra-Navegador {
    Get-CimInstance Win32_Process -Filter "Name='msedge.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*perfil_edge*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
function AlFrente {
    $form.Show(); $form.TopMost = $false; $form.TopMost = $true
    $form.Activate(); [void][W32]::SetForegroundWindow($form.Handle); [Windows.Forms.Application]::DoEvents()
}

# Tiempo para que se libere el control remoto del equipo y se oculten las demás ventanas
Start-Sleep -Seconds $EsperaInicial

# --- Navegador: una ventana InPrivate por página, abiertas y cargadas antes de grabar ---
$urls = [ordered]@{
    w1 = @("http://localhost:8000/docs", "API MiBici")
    w2 = @("http://127.0.0.1:5000/#/models/mibici-equipo32-p1-demanda", "mibici-equipo32-p1-demanda")
    w3 = @("https://github.com/JohannaCamargo2805/TC5061-Proyecto-Equipo32/actions", "orkflow")
}
$perfil = Join-Path $Base "perfil_edge_video"
$argsNav = @("--user-data-dir=`"$perfil`"", "--no-first-run", "--no-default-browser-check",
    "--disable-features=Translate,TranslateUI,msEdgeTranslate", "--inprivate", "--start-maximized", "--new-window")
$ventanas = @{}
foreach ($k in $urls.Keys) {
    Start-Process -FilePath $nav -ArgumentList ($argsNav + @($urls[$k][0]))
    $h = [IntPtr]::Zero
    $limiteV = (Ms) + 20000
    while ((Ms) -lt $limiteV -and $h -eq [IntPtr]::Zero) {
        Start-Sleep -Milliseconds 500
        $fg = [W32]::GetForegroundWindow(); $t = [W32]::Titulo($fg)
        if ($t -like "*$($urls[$k][1])*" -and $t -like "*Edge*") { $h = $fg }
        else { $h = [W32]::Busca($urls[$k][1], "InPrivate") }
    }
    $ventanas[$k] = $h
    if ($h -ne [IntPtr]::Zero) { [void][W32]::ShowWindow($h, 3) }
    Add-Content $bit ("ventana {0} {1} | {2}" -f $k, $h, [W32]::Titulo($h))
    Start-Sleep -Seconds 3
}
Start-Sleep -Seconds 6   # que terminen de cargar (MLflow tarda más la primera vez)
(New-Object -ComObject Shell.Application).MinimizeAll()
Start-Sleep -Milliseconds 1500
AlFrente
Bombea 1.5

# --- ffmpeg: escritorio completo 3840x2160 -> 1920x1080, 30 fps, sin cursor ---
$archivo = Join-Path $Base $Salida
$psi = New-Object Diagnostics.ProcessStartInfo
$psi.FileName = "ffmpeg"
$psi.Arguments = "-hide_banner -y -init_hw_device d3d11va=d3d -init_hw_device qsv=qs@d3d -f lavfi -i ddagrab=output_idx=0:framerate=30:draw_mouse=0 -vf hwmap=derive_device=qsv,format=qsv,scale_qsv=w=1920:h=1080 -c:v h264_qsv -global_quality 18 `"$archivo`""
$psi.UseShellExecute = $false; $psi.RedirectStandardInput = $true; $psi.RedirectStandardError = $true
$psi.CreateNoWindow = $true
$ff = New-Object Diagnostics.Process; $ff.StartInfo = $psi
$errFf = New-Object Text.StringBuilder
$null = Register-ObjectEvent -InputObject $ff -EventName ErrorDataReceived -Action { if ($EventArgs.Data) { [void]$Event.MessageData.AppendLine($EventArgs.Data) } } -MessageData $errFf
[void]$ff.Start(); $ff.BeginErrorReadLine()
Marca "ffmpeg_inicio"
try {
Bombea 3.0

# --- Láminas 1 a 11 ---
Lamina 1 "s01"; Bombea (1.0 + (D "s01") + 0.6)
foreach ($i in 2..11) {
    $id = "s{0:D2}" -f $i
    Lamina $i $id
    Bombea (0.4 + (D $id) + 0.5)
}

# --- Demo en vivo en PowerShell ---
$pausas = @(((D "d1") + 1.1), ((D "d2") + 0.8), ((D "d3") + 0.9), ((D "d4") + 1.1), ((D "d5") + 0.9)) |
    ForEach-Object { ([double]$_).ToString("0.00", $inv) }
$demo = Join-Path $Repo "scripts\demo_final.ps1"
$argumentos = "-NoExit -ExecutionPolicy Bypass -File `"$demo`" -SinNavegador -EscalaFuente 1.9 -PausaSegundos `"$($pausas -join ',')`" -Bitacora `"$bitDemo`""
$form.Hide(); [Windows.Forms.Application]::DoEvents()
Marca "demo_inicio"
$pDemo = Start-Process -FilePath "conhost.exe" -ArgumentList "powershell.exe $argumentos" -WindowStyle Maximized -PassThru
$limite = (Ms) + 120000
while ((Ms) -lt $limite) {
    Start-Sleep -Milliseconds 100
    $listo = $false
    if (Test-Path $bitDemo) { try { $listo = [bool]((Get-Content $bitDemo -ErrorAction Stop) -match "^paso5 ") } catch {} }
    if ($listo) { break }
}
Bombea ([double]::Parse($pausas[4], $inv) + 0.3)
Marca "demo_fin"

# --- Navegador: Swagger, Model Registry del proyecto y GitHub Actions (ya cargadas) ---
$anterior = [IntPtr]::Zero
foreach ($k in $urls.Keys) {
    $h = $ventanas[$k]
    Marca $k
    if ($h -ne [IntPtr]::Zero) {
        [void][W32]::ShowWindow($h, 3); [W32]::Encima($h, $true)
        if ($anterior -ne [IntPtr]::Zero) { [W32]::Encima($anterior, $false) }
        $anterior = $h
    } else {
        Start-Process -FilePath $nav -ArgumentList ($argsNav + @($urls[$k][0]))   # respaldo: abrirla en vivo
    }
    Bombea (0.8 + (D $k) + 0.8)
}
if ($anterior -ne [IntPtr]::Zero) { [W32]::Encima($anterior, $false) }

# --- Cierre: láminas 12 y 13 ---
Lamina 12 "s12_previo"
AlFrente
Marca "s12"
Bombea (0.4 + (D "s12") + 1.0)
Lamina 13 "lamina13"
Bombea 3.0

} catch {
    Add-Content $bit ("ERROR_TOMA {0} | {1}" -f $_.Exception.Message, $_.InvocationInfo.PositionMessage)
} finally {
    # --- Fin de la grabación (también si algo falló) ---
    try { $ff.StandardInput.Write("q"); $ff.StandardInput.Flush() } catch {}
    if (-not $ff.WaitForExit(20000)) { $ff.Kill() }
    Marca "ffmpeg_fin"
}
Start-Sleep -Milliseconds 500
Set-Content -Path (Join-Path $Base "toma_ffmpeg_log.txt") -Value $errFf.ToString() -Encoding UTF8
try { & taskkill.exe /PID $pDemo.Id /T /F | Out-Null } catch {}
try { Cierra-Navegador } catch {}
$form.Close()
Add-Content $bit "TERMINADO"
