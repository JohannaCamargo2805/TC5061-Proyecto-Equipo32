"""Coloca la narración sobre la toma única (sin cortar ni mover el video) y exporta el MP4 final.

Uso: python montar.py <carpeta_con_toma_y_bitacoras> <salida.mp4>
- Ancla: el primer cuadro en que aparece la lámina 1 (detectado en el video) = marca s01 de la bitácora.
- Cada segmento de voz se coloca en su marca de la bitácora + un desfase fijo.
- Comprueba la deriva con una segunda ancla (la aparición de la lámina 12 al final).
"""
import json
import pathlib
import re
import subprocess
import sys

d = pathlib.Path(sys.argv[1])
salida = pathlib.Path(sys.argv[2])
toma = d / "toma.mkv"


def marcas(path):
    m = {}
    for linea in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        p = linea.split()
        if len(p) >= 2 and p[1].isdigit():
            m[p[0]] = int(p[1])
    return m


bt = marcas(d / "bitacora_toma.txt")
bd = marcas(d / "bitacora_demo.txt")
dur = json.loads((d / "voz" / "duraciones.json").read_text(encoding="utf-8"))["duraciones"]


def luma(inicio, fin):
    """Promedio de luma (YAVG) por cuadro entre inicio y fin (s)."""
    r = subprocess.run(
        ["ffmpeg", "-v", "info", "-ss", f"{inicio:.3f}", "-to", f"{fin:.3f}", "-i", str(toma), "-vf",
         "signalstats,metadata=print:key=lavfi.signalstats.YAVG", "-f", "null", "-"],
        capture_output=True, text=True)
    t, out = None, []
    for linea in r.stderr.splitlines():
        mt = re.search(r"pts_time:([\d.]+)", linea)
        if mt:
            t = float(mt.group(1))
        my = re.search(r"YAVG=([\d.]+)", linea)
        if my and t is not None:
            out.append((inicio + t, float(my.group(1))))
    return out


# Ancla 1: lámina 1 (de negro a la portada)
serie = luma(0, 15)
negro = min(y for _, y in serie)
t_anc = next(t for t, y in serie if y > negro + 15)
print(f"ancla lamina1 en {t_anc:.3f} s (negro {negro:.1f})")


def tv(evento_ms):
    return t_anc + (evento_ms - bt["s01"]) / 1000.0


# Ancla 2: lámina 12 (comprobación de deriva)
pred12 = tv(bt["s12"])
serie12 = luma(pred12 - 3, pred12 + 3)
y12 = serie12[-1][1]
t12 = next(t for t, y in serie12 if abs(y - y12) < 3)
deriva = t12 - pred12
print(f"lamina12 predicha {pred12:.3f} s, observada {t12:.3f} s, deriva {deriva*1000:.0f} ms")

# Escala lineal si la deriva es medible (reloj de captura vs. reloj de pared)
escala = 1.0
if abs(deriva) > 0.08:
    escala = (t12 - t_anc) / (pred12 - t_anc)
    print(f"corrigiendo escala {escala:.6f}")


def tv2(evento_ms, extra=0.0):
    return t_anc + escala * ((evento_ms - bt["s01"]) / 1000.0) + extra


coloc = {"s01": tv2(bt["s01"], 1.0)}
for i in range(2, 12):
    coloc[f"s{i:02d}"] = tv2(bt[f"s{i:02d}"], 0.4)
for i in range(1, 6):
    coloc[f"d{i}"] = tv2(bd[f"paso{i}"], 0.3)
for k in ("w1", "w2", "w3"):
    coloc[k] = tv2(bt[k], 0.8)
coloc["s12"] = tv2(bt["s12"], 0.4)

inicio = max(0.0, t_anc - 0.6)
fin_video = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                  str(toma)], capture_output=True, text=True).stdout)
orden = sorted(coloc, key=coloc.get)
for a, b in zip(orden, orden[1:]):
    hueco = coloc[b] - (coloc[a] + dur[a])
    if hueco < 0.15:
        print(f"AVISO: {a} se encima con {b} ({hueco:.2f} s)")
linea_tiempo = {k: round(coloc[k] - inicio, 3) for k in orden}
(salida.parent / "linea_de_tiempo.json").write_text(json.dumps(
    {"ancla_lamina1_s": round(t_anc, 3), "deriva_ms": round(deriva * 1000), "inicio_recorte_s": round(inicio, 3),
     "segmentos_inicio_s": linea_tiempo, "duraciones_s": dur}, ensure_ascii=False, indent=1), encoding="utf-8")

args = ["ffmpeg", "-y", "-hide_banner", "-v", "error", "-ss", f"{inicio:.3f}", "-i", str(toma)]
filtros = []
for n, k in enumerate(orden, start=1):
    args += ["-i", str(d / "voz" / f"{k}.mp3")]
    ms = int(round((coloc[k] - inicio) * 1000))
    filtros.append(f"[{n}:a]aresample=48000,adelay={ms}|{ms}[a{n}]")
mezcla = "".join(f"[a{n}]" for n in range(1, len(orden) + 1))
filtros.append(f"{mezcla}amix=inputs={len(orden)}:normalize=0:dropout_transition=0,volume=1.22,"
               f"alimiter=limit=0.89,apad[aout]")
args += ["-filter_complex", ";".join(filtros), "-map", "0:v", "-map", "[aout]",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
         "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", "-shortest", "-movflags", "+faststart",
         "-metadata", "title=TC5061 Proyecto final - Equipo 32 - Pronóstico de demanda de MiBici",
         "-metadata", "comment=Toma única sin cortes; narración con voz sintética Microsoft Edge TTS es-MX-DaliaNeural",
         str(salida)]
subprocess.run(args, check=True)
print("listo", salida, "duración video original", round(fin_video - inicio, 2))
