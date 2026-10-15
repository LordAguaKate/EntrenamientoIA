#!/usr/bin/env python3
"""
EcoSort - benchmark visual comparativo en Raspberry Pi (probado en la Pi 5 y la Pi 4).

Corre, uno tras otro, los modelos .tflite del proyecto (MobileNetV3-Large,
EfficientNetB0 y ShuffleNetV2 con pesos de ImageNet) sobre el mismo conjunto
de imágenes y muestra todo el proceso en la terminal, pensado para grabarse en
video o capturarse como evidencia:

    Etapa 1  Sistema y modelos      ficha del dispositivo y de cada .tflite
    Etapa 2  Carga de imágenes      mismo preprocesamiento que en el entrenamiento
    Por cada modelo:
        a) Latencia y precisión     una inferencia cronometrada por imagen
        b) Estabilidad              inferencia continua midiendo CPU, RAM y temperatura
    Etapa final  Comparación        tabla comparativa y matrices de confusión

Pensado para correrse EN la Raspberry Pi (también funciona en la laptop, donde
cae a tensorflow.lite). Uso típico en la Raspberry:

    python3 benchmark_visual.py --models-dir modelos --images-dir test_set

Para una demostración rápida (video):

    python3 benchmark_visual.py --models-dir modelos --images-dir test_set \
        --max-imagenes 200 --duracion-estabilidad 60 --enfriamiento 10

Salida (carpeta resultados_<fecha>/):
    resultados.json     todas las cifras y las series de tiempo, para graficar
                        con tests/graficar_benchmark_rpi.py
    capturas/*.svg      capturas de cada pantalla del proceso

Requiere: numpy, pillow, psutil, rich y un runtime de TFLite
(ai-edge-litert, tflite-runtime o tensorflow).
"""

import argparse
import gc
import io
import json
import platform
import statistics
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
from PIL import Image

try:
    import tflite_runtime.interpreter as tflite
    RUNTIME = "tflite_runtime"
except ImportError:
    try:
        import ai_edge_litert.interpreter as tflite
        RUNTIME = "ai_edge_litert"
    except ImportError:
        import tensorflow.lite as tflite
        RUNTIME = "tensorflow.lite"

# Modelos a comparar. El orden es el orden en que se ejecutan.
MODELOS = [
    {"clave": "mobilenet", "nombre": "MobileNetV3-Large",
     "archivo": "modelo_residuos_rpi_V3_05.tflite", "color": "dodger_blue1"},
    {"clave": "efficientnet", "nombre": "EfficientNetB0",
     "archivo": "modelo_residuos_EfficientNetB0_v1.tflite", "color": "dark_orange"},
    {"clave": "shufflenet", "nombre": "ShuffleNetV2 (ImageNet)",
     "archivo": "modelo_residuos_shufflenetV2_imagenet.tflite", "color": "spring_green3"},
]

CLASES = ["aluminio", "carton", "otros", "plastico", "vidrio"]
COLOR_CLASE = {"aluminio": "cyan", "carton": "yellow", "otros": "magenta", "plastico": "blue", "vidrio": "green"}
# Mismas extensiones que acepta ImageDataGenerator de Keras, para evaluar
# exactamente las mismas imágenes que los notebooks de evaluación.
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".tiff"}
UMBRAL_H1_MS = 200.0
MASCARA_THROTTLING = 0xF000F
BLOQUES = "▁▂▃▄▅▆▇█"
ANCHO_CAPTURA = 124


# --------------------------------------------------------------------------
# Medición de RAM en un proceso aislado
# --------------------------------------------------------------------------
def modo_medir_ram(ruta, num_threads):
    """Se ejecuta como subproceso: carga solo un modelo y reporta su memoria.

    Así la cifra no se mezcla con las imágenes en memoria ni con los otros
    modelos que ya pasaron por el proceso principal.
    """
    proceso = psutil.Process()
    base = proceso.memory_info().rss
    interp = tflite.Interpreter(model_path=ruta, num_threads=num_threads)
    interp.allocate_tensors()
    idet, odet = interp.get_input_details()[0], interp.get_output_details()[0]
    entrada = np.zeros(idet["shape"], dtype=idet["dtype"])
    for _ in range(30):
        interp.set_tensor(idet["index"], entrada)
        interp.invoke()
        interp.get_tensor(odet["index"])
    con_modelo = proceso.memory_info().rss
    print(json.dumps({"proceso_base_mb": round(base / 2**20, 2),
                      "proceso_con_modelo_mb": round(con_modelo / 2**20, 2),
                      "incremento_modelo_mb": round((con_modelo - base) / 2**20, 2)}))


if "--medir-ram" in sys.argv:
    _i = sys.argv.index("--medir-ram")
    _hilos = int(sys.argv[_i + 2]) if len(sys.argv) > _i + 2 else None
    modo_medir_ram(sys.argv[_i + 1], _hilos or None)
    raise SystemExit(0)


from rich import box  # noqa: E402  (después del modo --medir-ram para no inflar su memoria)
from rich.console import Console, Group  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn  # noqa: E402
from rich.progress_bar import ProgressBar  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

console = Console()
DIR_CAPTURAS = None  # se define en main(); None = no guardar capturas


def nombre_dispositivo():
    """'Raspberry Pi 5', 'Raspberry Pi 4'... según /proc/device-tree/model; en otro equipo, su nombre de host."""
    try:
        modelo = Path("/proc/device-tree/model").read_text().rstrip("\x00\n ")
        return " ".join(modelo.split()[:3]) if modelo.startswith("Raspberry Pi") else modelo
    except OSError:
        return platform.node()


DISPOSITIVO = nombre_dispositivo()


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def capturar(nombre, renderable, titulo):
    """Guarda en SVG una pantalla del proceso, tal como se ve en la terminal."""
    if DIR_CAPTURAS is None:
        return
    grabadora = Console(record=True, width=ANCHO_CAPTURA, file=io.StringIO(), force_terminal=True,
                        color_system="truecolor", legacy_windows=False)
    grabadora.print(renderable)
    grabadora.save_svg(str(DIR_CAPTURAS / f"{nombre}.svg"), title=titulo)


def mostrar(renderable, captura=None, titulo=None):
    titulo = titulo or f"EcoSort · benchmark en {DISPOSITIVO}"
    console.print(renderable)
    if captura:
        capturar(captura, renderable, titulo)


def sparkline(valores, ancho=60, minimo=None, maximo=None):
    v = list(valores)
    if not v:
        return ""
    if len(v) > ancho:  # se promedia por tramos para que quepa la prueba completa
        paso = len(v) / ancho
        v = [statistics.fmean(v[int(i * paso):max(int(i * paso) + 1, int((i + 1) * paso))]) for i in range(ancho)]
    lo = min(v) if minimo is None else minimo
    hi = max(v) if maximo is None else maximo
    rango = (hi - lo) or 1.0
    return "".join(BLOQUES[min(7, max(0, int((x - lo) / rango * 7.999)))] for x in v)


def barra_nivel(pct, ancho=24):
    n = round(max(0.0, min(100.0, pct)) / 100 * ancho)
    color = "green" if pct < 50 else "yellow" if pct < 80 else "red"
    return Text("█" * n, style=color) + Text("░" * (ancho - n), style="grey30")


def barra_valor(valor, maximo, color, ancho=30):
    n = max(1, round(valor / maximo * ancho)) if maximo else 0
    return Text("█" * n, style=color) + Text("░" * (ancho - n), style="grey23")


def resumen(valores):
    valores = [float(x) for x in valores]
    if not valores:
        return {}
    return {
        "media": round(statistics.mean(valores), 3),
        "std": round(statistics.pstdev(valores), 3) if len(valores) > 1 else 0.0,
        "min": round(min(valores), 3),
        "p50": round(float(np.percentile(valores, 50)), 3),
        "p95": round(float(np.percentile(valores, 95)), 3),
        "p99": round(float(np.percentile(valores, 99)), 3),
        "max": round(max(valores), 3),
    }


def leer_sysfs(ruta, divisor=1.0):
    try:
        return float(Path(ruta).read_text()) / divisor
    except (OSError, ValueError):
        return None


def temp_cpu():
    return leer_sysfs("/sys/class/thermal/thermal_zone0/temp", 1000)


def freq_mhz():
    return leer_sysfs("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1000)


def throttled():
    try:
        salida = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout
        return int(salida.strip().split("=")[1], 16)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def fmt_tiempo(seg):
    seg = int(max(0, seg))
    return f"{seg // 60:02d}:{seg % 60:02d}"


def etapa(etiqueta, titulo, descripcion, color="blue"):
    console.print()
    console.rule(f"[bold white on {color}] {etiqueta} [/] [bold cyan]{titulo}", style=color)
    console.print(f"[dim]{descripcion}[/]\n")


def info_sistema():
    placa = platform.node()
    try:
        placa = Path("/proc/device-tree/model").read_text().rstrip("\x00\n ")
    except OSError:
        pass
    so = platform.platform()
    try:
        for linea in Path("/etc/os-release").read_text().splitlines():
            if linea.startswith("PRETTY_NAME="):
                so = linea.split("=", 1)[1].strip('"')
    except OSError:
        pass
    return placa, so


def kpi(etiqueta, valor, estilo="bold white"):
    return Text.assemble((f"{etiqueta}\n", "dim"), (valor, estilo))


# --------------------------------------------------------------------------
# Imágenes
# --------------------------------------------------------------------------
def listar_imagenes(images_dir, max_imagenes):
    """Lista las imágenes alternando entre clases (0 = todas)."""
    por_clase = {}
    for p in sorted(images_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTENSIONES:
            por_clase.setdefault(p.parent.name, []).append(p)
    if not por_clase:
        raise SystemExit(f"No se encontraron imágenes en {images_dir}")
    grupos = [por_clase[k] for k in sorted(por_clase)]
    total = sum(len(g) for g in grupos)
    limite = total if max_imagenes <= 0 else min(max_imagenes, total)
    rutas, i = [], 0
    while len(rutas) < limite:
        for g in grupos:
            if i < len(g) and len(rutas) < limite:
                rutas.append(g[i])
        i += 1
    return rutas


def cargar_imagenes(rutas, images_dir, input_size, interpolacion):
    """Decodifica y redimensiona las imágenes y las deja en memoria como uint8.

    Se usa el mismo preprocesamiento que ImageDataGenerator.flow_from_directory
    en los notebooks: conversión a RGB y redimensionado con interpolación
    'nearest'. Los tres modelos traen su normalización integrada, así que
    reciben los pixeles en el rango 0-255.
    """
    filtro = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC}[interpolacion]
    items = []
    with Progress(SpinnerColumn(), TextColumn("[bold]{task.description}"), BarColumn(bar_width=40),
                  MofNCompleteColumn(), TimeElapsedColumn(), console=console) as prog:
        tarea = prog.add_task("Cargando y preprocesando imágenes", total=len(rutas))
        for r in rutas:
            with Image.open(r) as img:
                arr = np.asarray(img.convert("RGB").resize((input_size[1], input_size[0]), filtro), dtype=np.uint8)
            real = r.parent.name if r.parent.name in CLASES else None
            items.append({"arr": arr, "nombre": r.name, "ruta": str(r.relative_to(images_dir)), "real": real})
            prog.advance(tarea)
    return items


def inferir(interpreter, idet, odet, arr):
    """Cronometra set_tensor + invoke + get_tensor de una imagen ya preprocesada."""
    entrada = arr[None].astype(idet["dtype"])  # la conversión de tipo queda fuera del cronómetro
    t0 = time.perf_counter()
    interpreter.set_tensor(idet["index"], entrada)
    interpreter.invoke()
    salida = interpreter.get_tensor(odet["index"])[0]
    return salida, (time.perf_counter() - t0) * 1000


# --------------------------------------------------------------------------
# Métricas de clasificación (con numpy, para no depender de scikit-learn)
# --------------------------------------------------------------------------
def metricas_clasificacion(reales, predichas):
    n = len(CLASES)
    mat = np.zeros((n, n), dtype=int)
    for r, p in zip(reales, predichas):
        if r is not None:
            mat[CLASES.index(r), CLASES.index(p)] += 1
    total = int(mat.sum())
    if not total:
        return None
    por_clase, precs, recs, f1s = {}, [], [], []
    for i, c in enumerate(CLASES):
        vp, soporte, predichos = int(mat[i, i]), int(mat[i].sum()), int(mat[:, i].sum())
        prec = vp / predichos if predichos else 0.0
        rec = vp / soporte if soporte else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        por_clase[c] = {"aciertos": vp, "total": soporte, "accuracy": round(rec, 4),
                        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
        if soporte:
            precs.append(prec), recs.append(rec), f1s.append(f1)
    aciertos = int(np.trace(mat))
    return {
        "accuracy": round(aciertos / total, 6), "aciertos": aciertos, "errores": total - aciertos, "total": total,
        "macro": {"precision": round(float(np.mean(precs)), 4), "recall": round(float(np.mean(recs)), 4),
                  "f1": round(float(np.mean(f1s)), 4)},
        "por_clase": por_clase,
        "matriz_confusion": mat.tolist(),
    }


# --------------------------------------------------------------------------
# Pantallas en vivo
# --------------------------------------------------------------------------
def render_latencia(e, alto=None, ancho=None):
    alto = alto or console.size.height
    ancho = ancho or console.size.width
    filas = list(e["filas"])[-max(5, alto - 22):]
    ancho_img = max(12, min(44, ancho - 72))
    tabla = Table(box=box.SIMPLE_HEAD, expand=True, header_style="bold")
    for col, kw in (("#", {"justify": "right", "min_width": 4}), ("Fase", {"min_width": 7}),
                    ("Imagen", {"max_width": ancho_img, "min_width": 12}), ("Real", {"min_width": 8}),
                    ("Predicha", {"min_width": 8}), ("Confianza", {"min_width": 16}),
                    ("Latencia", {"justify": "right", "min_width": 10}), ("", {"min_width": 1})):
        tabla.add_column(col, no_wrap=True, overflow="ellipsis", **kw)
    for f in filas:
        conf = f["conf"]
        n = round(conf * 10)
        barra = Text("█" * n, style="green" if conf >= 0.8 else "yellow" if conf >= 0.5 else "red")
        barra += Text("░" * (10 - n), style="grey30") + Text(f" {conf * 100:5.1f}%")
        if f["real"] is None:
            marca = Text("·", style="dim")
        else:
            marca = Text("✔", style="bold green") if f["real"] == f["pred"] else Text("✘", style="bold red")
        tabla.add_row(
            str(f["i"] + 1),
            Text("CALENT.", style="yellow") if f["cal"] else Text("MEDIDA", style="green"),
            Text(f["nombre"]),
            Text(f["real"] or "-", style=COLOR_CLASE.get(f["real"], "dim")),
            Text(f["pred"], style=f"bold {COLOR_CLASE[f['pred']]}"),
            barra,
            f"{f['ms']:.2f} ms",
            marca,
        )

    lat = e["lat"]
    if lat:
        media = statistics.fmean(lat)
        std = statistics.pstdev(lat) if len(lat) > 1 else 0.0
        p95 = float(np.percentile(lat, 95))
        cifras = [kpi("Media", f"{media:.2f} ms", "bold green"), kpi("Desv. est.", f"{std:.2f} ms"),
                  kpi("Mín", f"{min(lat):.2f} ms"), kpi("Máx", f"{max(lat):.2f} ms"), kpi("P95", f"{p95:.2f} ms"),
                  kpi("Cuadros/s", f"{1000 / media:.1f}")]
    else:
        media = 0.0
        cifras = [kpi("Media", "calentando…", "yellow")] + [kpi(x, "-") for x in ("Desv. est.", "Mín", "Máx", "P95", "Cuadros/s")]
    prec = f"{100 * e['ok'] / e['con_real']:.2f}%  ({e['con_real'] - e['ok']} errores)" if e["con_real"] else "-"
    cifras.append(kpi("Precisión acumulada", prec, "bold cyan"))
    kpis = Table.grid(expand=True)
    for _ in cifras:
        kpis.add_column(ratio=1)
    kpis.add_row(*cifras)

    hechas = e["hechas"]
    en_calentamiento = hechas <= e["warm"]
    fase = "Calentamiento (no cuenta en las métricas)" if en_calentamiento else "Medición de latencia y precisión"
    limite = Table.grid(expand=True)
    limite.add_column(width=38)
    limite.add_column(ratio=1)
    limite.add_row(Text(f"Latencia media vs límite H1 ({UMBRAL_H1_MS:.0f} ms)", style="dim"),
                   ProgressBar(total=UMBRAL_H1_MS, completed=min(media, UMBRAL_H1_MS), complete_style="green"))
    return Group(
        Text.assemble((f" {e['modelo']} ", f"bold white on {e['color']}"), ("  "),
                      (f"{fase}  ", "bold yellow" if en_calentamiento else "bold green"), (f"{hechas}/{e['total']}", "bold")),
        ProgressBar(total=e["total"], completed=hechas, complete_style=e["color"]),
        tabla,
        Panel(Group(kpis, Text(), limite), title="Estadísticas en vivo", border_style=e["color"], box=box.ROUNDED),
    )


def render_estabilidad(e):
    t = min(e["t"], e["dur"])
    ultimo = lambda k, d=None: e[k][-1] if e[k] else d  # noqa: E731
    tiempo = Table.grid(expand=True)
    tiempo.add_column(ratio=1)
    tiempo.add_column(width=36, justify="right")
    tiempo.add_row(ProgressBar(total=e["dur"], completed=t, complete_style=e["color"]),
                   Text(f"{fmt_tiempo(t)} / {fmt_tiempo(e['dur'])}  (faltan {fmt_tiempo(e['dur'] - t)})", style="bold"))

    cuerpo = Table.grid(padding=(0, 2), expand=True)
    cuerpo.add_column(width=16, style="dim")
    cuerpo.add_column(no_wrap=True, ratio=1)
    cuerpo.add_row("Modelo", Text(e["modelo"], style=f"bold {e['color']}"))
    cuerpo.add_row("Inferencias", Text.assemble((f"{e['n']:,}", "bold white"),
                                                (f"   {ultimo('ips', 0):.1f} inf/s   latencia {ultimo('lat', 0):.2f} ms", "dim")))
    cuerpo.add_row("", "")
    cuerpo.add_row("CPU total", Text.assemble((f"{ultimo('cpu', 0):5.1f}%  ", "bold white"),
                                              (sparkline(e["cpu"], 60, 0, 100), "cyan")))
    for i, pct in enumerate(e["cores"]):
        cuerpo.add_row(f"  núcleo {i}", barra_nivel(pct) + Text(f" {pct:5.1f}%"))
    cuerpo.add_row("", "")
    cuerpo.add_row("RAM proceso", Text.assemble((f"{ultimo('ram', 0):7.2f} MB  ", "bold white"),
                                                (f"sistema {e['ram_sis']:.1f}% usada", "dim")))
    temp = ultimo("temp")
    cuerpo.add_row("Temperatura", Text.assemble((f"{temp:5.1f} °C  " if temp is not None else "   n/d   ", "bold white"),
                                                (sparkline(e["temp"], 60, 30, 85), "yellow")))
    freq = e["freq"][-1] if e["freq"] else None
    cuerpo.add_row("Frecuencia CPU", Text(f"{freq:.0f} MHz" if freq else "n/d", style="bold white"))
    thr = e["thr"]
    if thr is None:
        estado_thr = Text("no disponible", style="dim")
    elif thr & MASCARA_THROTTLING:
        estado_thr = Text(f"ATENCIÓN (0x{thr:X})", style="bold red")
    else:
        estado_thr = Text("sin throttling ni bajo voltaje", style="bold green")
    cuerpo.add_row("Estado energía", estado_thr)
    cuerpo.add_row("", "")
    cuerpo.add_row("Latencia / seg", Text(sparkline(e["lat"], 60, 0, 2 * (statistics.fmean(e["lat"]) if e["lat"] else 1)), style="green"))
    return Panel(Group(tiempo, Text(), cuerpo), title=f"Ejecución continua · {e['modelo']}", border_style=e["color"], box=box.ROUNDED)


def prueba_estabilidad(interpreter, idet, odet, items, dur, intervalo, modelo, color):
    proceso = psutil.Process()
    e = {"modelo": modelo, "color": color, "dur": dur, "t": 0.0, "n": 0, "tiempos": [], "ips": [], "cpu": [],
         "cores": [0.0] * psutil.cpu_count(), "ram": [], "ram_sis": 0.0, "lat": [], "temp": [], "freq": [], "thr": throttled()}
    thr_visto = e["thr"] or 0
    psutil.cpu_percent(None)
    psutil.cpu_percent(None, percpu=True)
    t0 = ultima = ult_thr = ult_render = time.time()
    acc, idx = [], 0
    with Live(get_renderable=lambda: render_estabilidad(e), console=console, auto_refresh=False) as live:
        while True:
            ahora = time.time()
            e["t"] = ahora - t0
            if e["t"] >= dur:
                break
            _, ms = inferir(interpreter, idet, odet, items[idx % len(items)]["arr"])
            idx += 1
            e["n"] += 1
            acc.append(ms)
            if ahora - ultima >= intervalo:
                e["tiempos"].append(round(ahora - t0, 2))
                e["cpu"].append(psutil.cpu_percent(None))
                e["cores"] = psutil.cpu_percent(None, percpu=True)
                e["ram"].append(round(proceso.memory_info().rss / 2**20, 3))
                e["ram_sis"] = psutil.virtual_memory().percent
                e["lat"].append(round(statistics.fmean(acc), 4))
                e["ips"].append(round(len(acc) / (ahora - ultima), 2))
                acc, ultima = [], ahora
                temp = temp_cpu()
                if temp is not None:
                    e["temp"].append(temp)
                freq = freq_mhz()
                if freq is not None:
                    e["freq"].append(freq)
                if ahora - ult_thr >= 10:
                    ult_thr = ahora
                    nuevo = throttled()
                    if nuevo is not None:
                        thr_visto |= nuevo
                        e["thr"] = thr_visto
            if ahora - ult_render >= 0.25:
                live.refresh()
                ult_render = ahora
    return e


def enfriar(segundos):
    """Pausa entre pruebas para que todos los modelos arranquen a temperatura parecida."""
    if segundos <= 0:
        return
    with Progress(TextColumn("[dim]Enfriamiento antes de la siguiente prueba"), BarColumn(bar_width=30),
                  TextColumn("{task.fields[temp]}"), console=console, transient=True) as prog:
        tarea = prog.add_task("", total=segundos, temp="")
        t0 = time.time()
        while time.time() - t0 < segundos:
            temp = temp_cpu()
            prog.update(tarea, completed=time.time() - t0, temp=f"{temp:.1f} °C" if temp is not None else "")
            time.sleep(0.5)
    temp = temp_cpu()
    console.print(f"[dim]Enfriamiento de {segundos:g} s terminado" + (f" · temperatura {temp:.1f} °C[/]" if temp is not None else "[/]"))


def medir_ram_aislada(ruta, num_threads):
    try:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--medir-ram", str(ruta), str(num_threads or 0)]
        salida = subprocess.run(cmd, capture_output=True, text=True, timeout=180).stdout
        return json.loads(salida.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def estado(ok):
    return Text("✔ CUMPLE", style="bold green") if ok else Text("✘ REVISAR", style="bold red")


def tabla_matriz(mat, titulo):
    conf = Table(title=titulo, box=box.ROUNDED, title_style="bold")
    conf.add_column("Real \\ Predicha")
    for c in CLASES:
        conf.add_column(Text(c, style=COLOR_CLASE[c]), justify="right")
    conf.add_column("Acierto", justify="right")
    for i, c in enumerate(CLASES):
        total = int(sum(mat[i]))
        fila = [Text(str(mat[i][j]), style="bold green" if i == j and mat[i][j] else "bold red" if mat[i][j] else "dim")
                for j in range(len(CLASES))]
        conf.add_row(Text(c, style=COLOR_CLASE[c]), *fila, f"{100 * mat[i][i] / total:.2f}%" if total else "-")
    return conf


# --------------------------------------------------------------------------
# Prueba de un modelo
# --------------------------------------------------------------------------
def probar_modelo(k, n_modelos, cfg, ruta, items, args):
    nombre, color = cfg["nombre"], cfg["color"]
    etiqueta = f"MODELO {k}/{n_modelos}"

    etapa(f"{etiqueta} · a", f"{nombre} — latencia y precisión",
          f"Por cada imagen se cronometra set_tensor + invoke + get_tensor. Las primeras {args.n_warmup} inferencias "
          "son de calentamiento y se descartan de la latencia (no de la precisión).", color)
    enfriar(args.enfriamiento)
    temp_inicio = temp_cpu()
    t0 = time.perf_counter()
    interpreter = tflite.Interpreter(model_path=str(ruta), num_threads=args.num_threads)
    interpreter.allocate_tensors()
    carga_ms = (time.perf_counter() - t0) * 1000
    idet, odet = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    console.print(f"[green]✔[/] {nombre} cargado en [bold]{carga_ms:.0f} ms[/]")

    e3 = {"modelo": nombre, "color": color, "filas": deque(maxlen=60), "lat": [], "total": len(items),
          "warm": args.n_warmup, "ok": 0, "con_real": 0, "hechas": 0}
    preds, confs, lats = [], [], []
    with Live(get_renderable=lambda: render_latencia(e3), console=console, auto_refresh=False) as live:
        ult_render = 0.0
        for i, it in enumerate(items):
            salida, ms = inferir(interpreter, idet, odet, it["arr"])
            pred = CLASES[int(np.argmax(salida))]
            cal = i < args.n_warmup
            if not cal:
                e3["lat"].append(ms)
            if it["real"] is not None:
                e3["con_real"] += 1
                e3["ok"] += it["real"] == pred
            preds.append(pred), confs.append(round(float(np.max(salida)), 5)), lats.append(round(ms, 4))
            e3["hechas"] = i + 1
            e3["filas"].append({"i": i, "cal": cal, "nombre": it["nombre"], "real": it["real"], "pred": pred,
                                "conf": float(np.max(salida)), "ms": ms})
            if time.time() - ult_render >= 0.1:
                live.refresh()
                ult_render = time.time()
            if args.pausa_visual:
                time.sleep(args.pausa_visual)
    capturar(f"{k}a_latencia_{cfg['clave']}", render_latencia(e3, alto=42, ancho=ANCHO_CAPTURA),
             f"{nombre} · prueba de latencia y precisión")

    etapa(f"{etiqueta} · b", f"{nombre} — estabilidad",
          f"Inferencia continua durante {fmt_tiempo(args.duracion_estabilidad)} (mm:ss) muestreando CPU, RAM, "
          f"temperatura y latencia cada {args.intervalo_muestreo:g} s.", color)
    e4 = prueba_estabilidad(interpreter, idet, odet, items, args.duracion_estabilidad, args.intervalo_muestreo, nombre, color)
    capturar(f"{k}b_estabilidad_{cfg['clave']}", render_estabilidad(e4), f"{nombre} · prueba de estabilidad")

    res_lat = resumen(e3["lat"])
    n10 = max(1, len(e4["lat"]) // 10)
    deriva = (statistics.fmean(e4["lat"][-n10:]) / statistics.fmean(e4["lat"][:n10]) - 1) * 100 if len(e4["lat"]) >= 10 else None
    delta_ram = e4["ram"][-1] - e4["ram"][0] if e4["ram"] else None
    thr_fin = e4["thr"]
    ips = e4["n"] / args.duracion_estabilidad if args.duracion_estabilidad else 0.0
    res_cpu, res_temp = resumen(e4["cpu"]), resumen(e4["temp"])
    clasif = metricas_clasificacion([it["real"] for it in items], preds)

    tabla = Table(title=f"Resumen · {nombre}", box=box.ROUNDED, title_style=f"bold {color}", header_style="bold")
    for col in ("Métrica", "Valor", "Criterio", "Estado"):
        tabla.add_column(col)
    if clasif:
        tabla.add_row("Precisión (accuracy)", f"{clasif['accuracy'] * 100:.2f}%  ({clasif['aciertos']}/{clasif['total']}, "
                      f"{clasif['errores']} errores)", "-", Text("-", style="dim"))
        tabla.add_row("F1 macro", f"{clasif['macro']['f1']:.4f}", "-", Text("-", style="dim"))
    tabla.add_row("Latencia media", f"{res_lat['media']:.2f} ms (±{res_lat['std']:.2f})  ·  {1000 / res_lat['media']:.1f} cuadros/s",
                  f"< {UMBRAL_H1_MS:.0f} ms (H1)",
                  Text.assemble(estado(res_lat["media"] < UMBRAL_H1_MS), (f"  margen x{UMBRAL_H1_MS / res_lat['media']:.1f}", "dim")))
    tabla.add_row("Latencia P95 / máx", f"{res_lat['p95']:.2f} ms / {res_lat['max']:.2f} ms", f"< {UMBRAL_H1_MS:.0f} ms (H1)",
                  estado(res_lat["max"] < UMBRAL_H1_MS))
    if e4["n"]:
        tabla.add_row("Inferencias continuas", f"{e4['n']:,} en {fmt_tiempo(args.duracion_estabilidad)}  ({ips:.1f} inf/s)", "-", Text("-", style="dim"))
    if res_cpu:
        tabla.add_row("CPU media / máx", f"{res_cpu['media']:.1f}% / {res_cpu['max']:.1f}%", "-", Text("-", style="dim"))
    if delta_ram is not None:
        tabla.add_row("RAM proceso (inicio → fin)", f"{e4['ram'][0]:.2f} → {e4['ram'][-1]:.2f} MB  (Δ {delta_ram:+.2f})", "Δ ≤ 2 MB",
                      estado(delta_ram <= 2))
    if deriva is not None:
        tabla.add_row("Deriva de latencia (primer vs último 10%)", f"{deriva:+.2f}%", "|Δ| ≤ 5%", estado(abs(deriva) <= 5))
    if res_temp:
        tabla.add_row("Temperatura media / máx", f"{res_temp['media']:.1f} °C / {res_temp['max']:.1f} °C", "< 80 °C", estado(res_temp["max"] < 80))
    if thr_fin is not None:
        tabla.add_row("Throttling / bajo voltaje", "ninguno" if not thr_fin & MASCARA_THROTTLING else f"flags 0x{thr_fin:X}", "ninguno",
                      estado(not thr_fin & MASCARA_THROTTLING))
    bloque = [tabla]
    if clasif:
        bloque.append(tabla_matriz(clasif["matriz_confusion"], f"Matriz de confusión · {nombre} (filas = real, columnas = predicha)"))
    mostrar(Group(*bloque), captura=f"{k}c_resumen_{cfg['clave']}", titulo=f"{nombre} · resumen")

    del interpreter
    gc.collect()
    return {
        "clave": cfg["clave"], "nombre": nombre, "archivo": ruta.name,
        "tam_tflite_mb": round(ruta.stat().st_size / 2**20, 3),
        "carga_ms": round(carga_ms, 1),
        "entrada": f"{idet['shape'].tolist()} {np.dtype(idet['dtype']).name}",
        "salida": f"{odet['shape'].tolist()} {np.dtype(odet['dtype']).name}",
        "temperatura_inicio_c": temp_inicio,
        "latencia_ms": res_lat,
        "cuadros_por_segundo": round(1000 / res_lat["media"], 2),
        "cumple_h1": bool(res_lat["max"] < UMBRAL_H1_MS),
        "clasificacion": clasif,
        "estabilidad": {
            "duracion_segundos": args.duracion_estabilidad,
            "n_inferencias": e4["n"],
            "inferencias_por_segundo": round(ips, 2),
            "cpu_percent": res_cpu,
            "ram_mb": resumen(e4["ram"]),
            "delta_ram_mb": round(delta_ram, 3) if delta_ram is not None else None,
            "latencia_ms_por_segundo": resumen(e4["lat"]),
            "deriva_latencia_pct": round(deriva, 2) if deriva is not None else None,
            "temperatura_c": res_temp,
            "frecuencia_mhz": resumen(e4["freq"]),
            "throttled_flags": f"0x{thr_fin:X}" if thr_fin is not None else None,
            "series": {"t_s": e4["tiempos"], "cpu_percent": e4["cpu"], "ram_mb": e4["ram"], "latencia_ms": e4["lat"],
                       "inferencias_por_segundo": e4["ips"], "temperatura_c": e4["temp"], "frecuencia_mhz": e4["freq"]},
        },
        "por_imagen": {"predicha": preds, "confianza": confs, "latencia_ms": lats},
    }


# --------------------------------------------------------------------------
# Comparación final
# --------------------------------------------------------------------------
def comparacion_final(resultados):
    color = {m["nombre"]: m["color"] for m in MODELOS}
    hay_clasif = all(r["clasificacion"] for r in resultados)
    tablas = []
    if hay_clasif:
        t_prec = Table(title=f"Precisión de clasificación en {DISPOSITIVO}", box=box.ROUNDED, title_style="bold", header_style="bold")
        for col, just in (("Modelo", "left"), ("Accuracy", "right"), ("Errores", "right"), ("Precisión macro", "right"),
                          ("Recall macro", "right"), ("F1 macro", "right")):
            t_prec.add_column(col, justify=just, no_wrap=True)
        for r in resultados:
            c = r["clasificacion"]
            t_prec.add_row(Text(r["nombre"], style=f"bold {color[r['nombre']]}"), f"{c['accuracy'] * 100:.2f}%",
                           f"{c['errores']} / {c['total']}", f"{c['macro']['precision']:.4f}", f"{c['macro']['recall']:.4f}",
                           f"{c['macro']['f1']:.4f}")
        tablas += [t_prec, Text()]

    tabla = Table(title=f"Rendimiento en {DISPOSITIVO}", box=box.ROUNDED, title_style="bold", header_style="bold")
    for col, just in (("Modelo", "left"), ("Latencia media", "right"), ("P95", "right"), ("Cuadros/s", "right"),
                      ("CPU", "right"), ("RAM modelo", "right"), ("T. máx", "right"), (".tflite", "right"),
                      ("H1", "left")):
        tabla.add_column(col, justify=just, no_wrap=True)
    for r in resultados:
        est, ram = r["estabilidad"], r.get("ram_aislada")
        tabla.add_row(
            Text(r["nombre"], style=f"bold {color[r['nombre']]}"),
            f"{r['latencia_ms']['media']:.2f} ms",
            f"{r['latencia_ms']['p95']:.2f} ms",
            f"{r['cuadros_por_segundo']:.1f}",
            f"{est['cpu_percent']['media']:.1f}%" if est["cpu_percent"] else "-",
            f"{ram['incremento_modelo_mb']:.1f} MB" if ram else "-",
            f"{est['temperatura_c']['max']:.1f} °C" if est["temperatura_c"] else "-",
            f"{r['tam_tflite_mb']:.2f} MB",
            estado(r["cumple_h1"]),
        )
    tablas.append(tabla)

    barras = Table.grid(padding=(0, 2))
    barras.add_column(width=32, no_wrap=True)
    barras.add_column()
    barras.add_column(justify="right")
    max_lat = max(r["latencia_ms"]["media"] for r in resultados)
    barras.add_row(Text("Latencia media (menos es mejor)", style="bold"), "", "")
    for r in resultados:
        barras.add_row(Text(r["nombre"], style=color[r["nombre"]]), barra_valor(r["latencia_ms"]["media"], max_lat, color[r["nombre"]], 50),
                       f"{r['latencia_ms']['media']:.2f} ms")
    if hay_clasif:
        barras.add_row("", "", "")
        barras.add_row(Text("Accuracy (más es mejor)", style="bold"), "", "")
        for r in resultados:
            barras.add_row(Text(r["nombre"], style=color[r["nombre"]]),
                           barra_valor(r["clasificacion"]["accuracy"], 1.0, color[r["nombre"]], 50),
                           f"{r['clasificacion']['accuracy'] * 100:.2f}%")
    mostrar(Group(*tablas, Text(), Panel(barras, box=box.ROUNDED, border_style="cyan")), captura="9a_comparacion_final",
            titulo="Comparación final de los modelos")

    if hay_clasif:
        clases = Table(title="Accuracy por clase", box=box.ROUNDED, title_style="bold", header_style="bold")
        clases.add_column("Clase")
        for r in resultados:
            clases.add_column(Text(r["nombre"], style=color[r["nombre"]]), justify="right")
        for c in CLASES:
            vals = [r["clasificacion"]["por_clase"][c]["accuracy"] for r in resultados]
            clases.add_row(Text(c, style=COLOR_CLASE[c]),
                           *[Text(f"{v * 100:.2f}%", style="bold green" if v == max(vals) else "") for v in vals])
        mostrar(clases, captura="9b_accuracy_por_clase", titulo=f"Accuracy por clase en {DISPOSITIVO}")

        mejor = max(resultados, key=lambda r: r["clasificacion"]["accuracy"])
        console.print(f"[green]✔[/] Mayor precisión : [bold]{mejor['nombre']}[/]  →  {mejor['clasificacion']['accuracy'] * 100:.2f}%")
    rapido = min(resultados, key=lambda r: r["latencia_ms"]["media"])
    console.print(f"[green]✔[/] Menor latencia  : [bold]{rapido['nombre']}[/]  →  {rapido['latencia_ms']['media']:.2f} ms")
    ligero = min(resultados, key=lambda r: r["tam_tflite_mb"])
    console.print(f"[green]✔[/] Más ligero      : [bold]{ligero['nombre']}[/]  →  {ligero['tam_tflite_mb']:.2f} MB")


def main():
    global DIR_CAPTURAS
    aqui = Path(__file__).resolve().parent
    # En la Raspberry los modelos van junto al script (modelos/); en el repo, en models/
    dir_modelos = aqui / "modelos" if (aqui / "modelos").is_dir() else aqui.parent / "models"
    dir_imagenes = aqui / "test_set" if (aqui / "test_set").is_dir() else aqui.parent / "data" / "raw" / "waste_classification" / "test"

    ap = argparse.ArgumentParser(description="Benchmark visual comparativo de los modelos .tflite en Raspberry Pi")
    ap.add_argument("--models-dir", type=Path, default=dir_modelos)
    ap.add_argument("--images-dir", type=Path, default=dir_imagenes, help="Carpeta con una subcarpeta por clase")
    ap.add_argument("--modelos", nargs="+", choices=[m["clave"] for m in MODELOS], default=[m["clave"] for m in MODELOS],
                    help="Modelos a probar (default: los tres)")
    ap.add_argument("--max-imagenes", type=int, default=0, help="0 = todas las imágenes de la carpeta (default)")
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--duracion-estabilidad", type=int, default=300, help="Segundos de inferencia continua por modelo")
    ap.add_argument("--intervalo-muestreo", type=float, default=1.0)
    ap.add_argument("--enfriamiento", type=float, default=60.0, help="Segundos de pausa antes de probar cada modelo")
    ap.add_argument("--num-threads", type=int, default=None, help="Hilos del intérprete (default: el del runtime)")
    ap.add_argument("--interpolacion", choices=["nearest", "bilinear", "bicubic"], default="nearest",
                    help="Redimensionado de las imágenes (default: nearest, igual que en los notebooks)")
    ap.add_argument("--pausa-visual", type=float, default=0.0,
                    help="pausa entre inferencias para que se lea en video; la CPU baja de frecuencia y la latencia "
                         "medida sube (usar 0 para cifras oficiales)")
    ap.add_argument("--salida-dir", type=Path, default=None, help="Carpeta de resultados (default: resultados_<fecha>)")
    ap.add_argument("--sin-capturas", action="store_true", help="No guardar las capturas SVG de cada pantalla")
    args = ap.parse_args()

    inicio = datetime.now()
    salida_dir = args.salida_dir or Path(f"resultados_{inicio:%Y%m%d_%H%M%S}")
    salida_dir.mkdir(parents=True, exist_ok=True)
    if not args.sin_capturas:
        DIR_CAPTURAS = salida_dir / "capturas"
        DIR_CAPTURAS.mkdir(exist_ok=True)

    seleccion = [m for m in MODELOS if m["clave"] in args.modelos]
    for m in seleccion:
        if not (args.models_dir / m["archivo"]).exists():
            raise SystemExit(f"No se encontró el modelo {args.models_dir / m['archivo']}")
    placa, so = info_sistema()

    console.clear()
    mostrar(Panel(Text.assemble(("EcoSort · Clasificación de residuos\n", "bold white"),
                                (f"Benchmark comparativo de inferencia en {DISPOSITIVO}\n", "cyan"),
                                (" · ".join(m["nombre"] for m in seleccion), "dim")),
                  border_style="bright_blue", box=box.DOUBLE, padding=(1, 4)))

    # ---------------- Etapa 1 ----------------
    etapa("ETAPA 1", "Sistema y modelos",
          "Se identifica el dispositivo y se inspecciona cada modelo .tflite: tamaño, tensores de entrada/salida "
          "y memoria que ocupa cargado en un proceso aislado.")
    thr0 = throttled()
    temp0 = temp_cpu()
    ficha = Table.grid(padding=(0, 2))
    ficha.add_column(style="dim")
    ficha.add_column(style="bold white")
    for k, v in (("Placa", placa), ("Sistema", so), ("Kernel", platform.release()), ("Arquitectura", platform.machine()),
                 ("Python", platform.python_version()), ("RAM total", f"{psutil.virtual_memory().total / 2**30:.1f} GiB"),
                 ("Núcleos", str(psutil.cpu_count())), ("Runtime", RUNTIME),
                 ("Hilos de inferencia", str(args.num_threads) if args.num_threads else "valor por defecto del runtime"),
                 ("Temperatura inicial", f"{temp0:.1f} °C" if temp0 is not None else "n/d"),
                 ("Preprocesamiento", f"RGB, redimensionado '{args.interpolacion}', float32 en rango 0-255 (normalización dentro del modelo)"),
                 ("Inicio", f"{inicio:%Y-%m-%d %H:%M:%S}")):
        ficha.add_row(k, Text(v))

    tabla_modelos = Table(title="Modelos a evaluar", box=box.ROUNDED, title_style="bold", header_style="bold")
    for col, just in (("#", "right"), ("Modelo", "left"), ("Archivo", "left"), ("Tamaño", "right"), ("Entrada", "left"),
                      ("Salida", "left"), ("RAM del modelo", "right")):
        tabla_modelos.add_column(col, justify=just)
    ram_aislada, input_size = {}, None
    with console.status("Inspeccionando modelos…"):
        for i, m in enumerate(seleccion, 1):
            ruta = args.models_dir / m["archivo"]
            interp = tflite.Interpreter(model_path=str(ruta))
            idet, odet = interp.get_input_details()[0], interp.get_output_details()[0]
            tam = (int(idet["shape"][1]), int(idet["shape"][2]))
            if input_size and tam != input_size:
                raise SystemExit("Los modelos no comparten el mismo tamaño de entrada")
            input_size = tam
            del interp
            ram_aislada[m["clave"]] = medir_ram_aislada(ruta, args.num_threads)
            ram = ram_aislada[m["clave"]]
            tabla_modelos.add_row(str(i), Text(m["nombre"], style=f"bold {m['color']}"), m["archivo"],
                                  f"{ruta.stat().st_size / 2**20:.2f} MB",
                                  f"{idet['shape'].tolist()} {np.dtype(idet['dtype']).name}",
                                  f"{odet['shape'].tolist()} {np.dtype(odet['dtype']).name}",
                                  f"{ram['incremento_modelo_mb']:.1f} MB" if ram else "n/d")
    mostrar(Group(Panel(ficha, title="Ficha del entorno", border_style="green", box=box.ROUNDED), tabla_modelos),
            captura="0a_sistema_y_modelos", titulo="Etapa 1 · sistema y modelos")
    if thr0 is not None:
        console.print(("[green]✔[/] Energía correcta al iniciar" if not thr0 & MASCARA_THROTTLING
                       else f"[yellow]![/] Flags de energía al iniciar: 0x{thr0:X}"))
    if args.pausa_visual:
        console.print(f"[yellow]![/] Modo demostración: pausa de {args.pausa_visual:g} s entre inferencias. "
                      "La latencia medida sale más alta; para cifras oficiales usa --pausa-visual 0.")

    # ---------------- Etapa 2 ----------------
    rutas = listar_imagenes(args.images_dir, args.max_imagenes)
    etapa("ETAPA 2", "Carga de imágenes de prueba",
          f"Se toman {len(rutas)} imágenes alternando entre clases, se redimensionan a {input_size[0]}x{input_size[1]} "
          "y se dejan en memoria. Los tres modelos reciben exactamente las mismas imágenes.")
    t0 = time.perf_counter()
    items = cargar_imagenes(rutas, args.images_dir, input_size, args.interpolacion)
    carga_img_s = time.perf_counter() - t0
    dist = Table(title="Imágenes por clase", box=box.SIMPLE_HEAD, title_style="bold")
    dist.add_column("Clase")
    dist.add_column("Imágenes", justify="right")
    dist.add_column("")
    por_clase = {c: sum(1 for it in items if it["real"] == c) for c in CLASES}
    for c in CLASES:
        dist.add_row(Text(c, style=COLOR_CLASE[c]), str(por_clase[c]), barra_valor(por_clase[c], max(por_clase.values()) or 1, COLOR_CLASE[c], 40))
    dist.add_row(Text("Total", style="bold"), Text(str(len(items)), style="bold"), "")
    mostrar(dist, captura="0b_imagenes_de_prueba", titulo="Etapa 2 · imágenes de prueba")
    console.print(f"[green]✔[/] {len(items)} imágenes preprocesadas en [bold]{carga_img_s:.1f} s[/] "
                  f"({1000 * carga_img_s / len(items):.1f} ms por imagen)")

    # ---------------- Modelos ----------------
    resultados = []
    for k, m in enumerate(seleccion, 1):
        r = probar_modelo(k, len(seleccion), m, args.models_dir / m["archivo"], items, args)
        r["ram_aislada"] = ram_aislada[m["clave"]]
        resultados.append(r)

    # ---------------- Comparación ----------------
    etapa("ETAPA FINAL", "Comparación de los modelos",
          f"Mismas {len(items)} imágenes, mismo dispositivo y mismo procedimiento para los {len(resultados)} modelos.")
    comparacion_final(resultados)

    fin = datetime.now()
    reporte = {
        "dispositivo": {**platform.uname()._asdict(), "placa": placa, "sistema": so,
                        "ram_total_mb": round(psutil.virtual_memory().total / 2**20), "nucleos": psutil.cpu_count()},
        "runtime": RUNTIME,
        "python": platform.python_version(),
        "fecha_inicio": inicio.isoformat(timespec="seconds"),
        "fecha_fin": fin.isoformat(timespec="seconds"),
        "duracion_total_s": round((fin - inicio).total_seconds()),
        "parametros": {"max_imagenes": args.max_imagenes, "n_warmup": args.n_warmup,
                       "duracion_estabilidad_s": args.duracion_estabilidad, "intervalo_muestreo_s": args.intervalo_muestreo,
                       "enfriamiento_s": args.enfriamiento, "num_threads": args.num_threads,
                       "interpolacion": args.interpolacion, "pausa_visual_s": args.pausa_visual,
                       "umbral_h1_ms": UMBRAL_H1_MS},
        "preprocesamiento": f"RGB, resize {input_size[0]}x{input_size[1]} '{args.interpolacion}', float32 0-255 (normalización dentro del modelo)",
        "clases": CLASES,
        "imagenes": {"total": len(items), "por_clase": por_clase, "carga_s": round(carga_img_s, 2),
                     "ruta": [it["ruta"] for it in items], "real": [it["real"] for it in items]},
        "modelos": resultados,
    }
    salida_json = salida_dir / "resultados.json"
    salida_json.write_text(json.dumps(reporte, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[green]✔[/] Resultados guardados en [bold]{salida_json.resolve()}[/]")
    if DIR_CAPTURAS:
        console.print(f"[green]✔[/] Capturas de pantalla en  [bold]{DIR_CAPTURAS.resolve()}[/]")
    console.print(f"[dim]Duración total de la prueba: {fmt_tiempo((fin - inicio).total_seconds())} (mm:ss)[/]\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Prueba interrumpida por el usuario.[/]")
        raise SystemExit(130)
