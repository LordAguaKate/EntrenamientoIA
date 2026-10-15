"""
Genera las gráficas e imágenes de evidencia a partir de los resultados del
benchmark hecho EN la Raspberry Pi 5 (tests/benchmark_visual.py).

Todas las cifras que se grafican vienen del resultados.json medido en la
Raspberry; este script solo dibuja, por eso puede correrse en la laptop
(la Raspberry no tiene matplotlib instalado).

Uso (desde la raíz del repo):
    python tests/graficar_benchmark_rpi.py

Entradas (por defecto en reports/benchmark_rpi/):
    resultados_rpi_1hilo.json     corrida oficial
    resultados_rpi_4hilos.json    corrida complementaria con 4 hilos (opcional)
    capturas_svg/                 capturas de pantalla del proceso (opcional)

Salida:
    reports/figures/raspberry/*.png

Requiere matplotlib, numpy y pillow (ya están en requirements.txt). Para
convertir las capturas SVG a PNG hace falta el comando rsvg-convert (librsvg).
"""

import argparse
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402
from PIL import Image  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DIR_DATOS = BASE_DIR / "reports" / "benchmark_rpi"
DIR_FIGURAS = BASE_DIR / "reports" / "figures" / "raspberry"
DIR_TEST = BASE_DIR / "data" / "raw" / "waste_classification" / "test"
DISPOSITIVO = "Raspberry Pi 5"  # se reemplaza en main() por la placa que aparece en el JSON

# Un color fijo por modelo en todas las gráficas (paleta validada para daltonismo)
COLOR = {"mobilenet": "#2a78d6", "efficientnet": "#eb6834", "shufflenet": "#1baf7a"}
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TENUE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"
CRITICO = "#d03b3b"
# Rampa secuencial de un solo tono para las matrices de confusión
RAMPA_AZUL = LinearSegmentedColormap.from_list("azul", [SUPERFICIE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": TINTA,
    "axes.edgecolor": EJE, "axes.labelcolor": TINTA_2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": TENUE, "ytick.color": TENUE, "xtick.labelcolor": TINTA_2, "ytick.labelcolor": TINTA_2,
    "grid.color": REJILLA, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
})


def titulo(fig, texto, sub, y=0.98):
    """Título y subtítulo alineados a la izquierda, fuera del área de datos."""
    fig.text(0.02, y, texto, fontsize=13, fontweight="bold", ha="left", va="top", color=TINTA)
    fig.text(0.02, y - 0.055 * (4.5 / fig.get_figheight()), sub, fontsize=9.5, ha="left", va="top", color=TINTA_2)


def mostrar_ruta(ruta):
    try:
        return ruta.relative_to(BASE_DIR)
    except ValueError:
        return ruta


def guardar(fig, nombre):
    ruta = DIR_FIGURAS / nombre
    fig.savefig(ruta)
    plt.close(fig)
    print(f"  {mostrar_ruta(ruta)}")


def wilson(aciertos, n, z=1.96):
    """Intervalo de confianza de Wilson al 95% para una proporción."""
    p = aciertos / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    mitad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centro - mitad, centro + mitad


def contexto(d):
    hilos = d["parametros"]["num_threads"] or 1
    return (f"{DISPOSITIVO} · {d['imagenes']['total']:,} imágenes de prueba · {d['runtime']} · "
            f"{hilos} hilo{'s' if hilos > 1 else ''} · modelos .tflite cuantizados")


def leyenda_modelos(fig, modelos, y=0.9, marcador="s"):
    manijas = [plt.Line2D([], [], marker=marcador, ls="", color=COLOR[m["clave"]], markersize=8, label=m["nombre"]) for m in modelos]
    fig.legend(handles=manijas, loc="upper left", bbox_to_anchor=(0.015, y), ncols=len(modelos), handletextpad=0.3, columnspacing=1.6)


# --------------------------------------------------------------------------
# Gráficas
# --------------------------------------------------------------------------
def fig_accuracy(d):
    modelos = d["modelos"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    fig.subplots_adjust(top=0.74, left=0.27, right=0.9)
    for i, m in enumerate(modelos):
        c = m["clasificacion"]
        lo, hi = wilson(c["aciertos"], c["total"])
        acc = c["accuracy"] * 100
        ax.plot([lo * 100, hi * 100], [i, i], color=COLOR[m["clave"]], lw=2, solid_capstyle="round")
        ax.plot(acc, i, "o", color=COLOR[m["clave"]], markersize=10, markeredgecolor=SUPERFICIE, markeredgewidth=2)
        ax.text(acc, i - 0.28, f"{acc:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.text(hi * 100 + 0.12, i, f"{c['errores']} errores", va="center", fontsize=9, color=TINTA_2)
    ax.set_yticks(range(len(modelos)), [m["nombre"] for m in modelos], fontsize=10.5)
    ax.set_ylim(len(modelos) - 0.5, -0.6)
    ax.set_xlim(92, 98.5)
    ax.set_xlabel("Accuracy sobre el conjunto de prueba (%)  —  el eje no inicia en cero")
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    titulo(fig, f"Precisión de clasificación en la {DISPOSITIVO}",
           contexto(d) + "\nPunto: accuracy medida. Barra: intervalo de confianza del 95% (Wilson).")
    guardar(fig, "01_accuracy_modelos.png")


def fig_accuracy_por_clase(d):
    modelos, clases = d["modelos"], d["clases"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    fig.subplots_adjust(top=0.74, left=0.13)
    for j, c in enumerate(clases):
        vals = [m["clasificacion"]["por_clase"][c]["accuracy"] * 100 for m in modelos]
        desplaz = np.linspace(-0.22, 0.22, len(modelos))  # cada modelo en su propio carril para que no se encimen
        for m, v, dy in zip(modelos, vals, desplaz):
            ax.plot([0, v], [j + dy, j + dy], color=REJILLA, lw=1, zorder=1)
            ax.plot(v, j + dy, "o", color=COLOR[m["clave"]], markersize=10, markeredgecolor=SUPERFICIE, markeredgewidth=2, zorder=3)
            ax.text(v + 0.3, j + dy, f"{v:.1f}%", ha="left", va="center", fontsize=8.5, color=TINTA_2)
    soporte = [modelos[0]["clasificacion"]["por_clase"][c]["total"] for c in clases]
    ax.set_yticks(range(len(clases)), [f"{c}\n(n = {n})" for c, n in zip(clases, soporte)])
    ax.set_ylim(len(clases) - 0.5, -0.5)
    minimo = min(m["clasificacion"]["por_clase"][c]["accuracy"] for m in modelos for c in clases) * 100
    ax.set_xlim(math.floor(minimo) - 1, 101.6)
    ax.set_xlabel("Accuracy por clase (%)  —  el eje no inicia en cero")
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    leyenda_modelos(fig, modelos, y=0.865, marcador="o")
    titulo(fig, f"Accuracy por clase en la {DISPOSITIVO}", contexto(d))
    guardar(fig, "02_accuracy_por_clase.png")


def fig_matrices(d):
    modelos, clases = d["modelos"], d["clases"]
    fig, axes = plt.subplots(1, len(modelos), figsize=(4.6 * len(modelos), 5.2))
    fig.subplots_adjust(top=0.78, wspace=0.28, left=0.07, right=0.99, bottom=0.14)
    for ax, m in zip(np.atleast_1d(axes), modelos):
        mat = np.array(m["clasificacion"]["matriz_confusion"])
        norm = mat / mat.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap=RAMPA_AZUL, norm=PowerNorm(gamma=0.35, vmin=0, vmax=1))
        for i in range(len(clases)):
            for j in range(len(clases)):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=10.5,
                        fontweight="bold" if i == j else "normal",
                        color="white" if norm[i, j] > 0.25 else (TINTA if mat[i, j] else TENUE))
        ax.set_xticks(range(len(clases)), clases, rotation=35, ha="right")
        ax.set_yticks(range(len(clases)), clases)
        ax.set_xticks(np.arange(-0.5, len(clases)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(clases)), minor=True)
        ax.grid(which="minor", color=SUPERFICIE, linewidth=2)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        c = m["clasificacion"]
        ax.set_title(f"{m['nombre']}\n{c['accuracy'] * 100:.2f}% · {c['errores']} errores", fontsize=11, loc="left", color=TINTA)
        ax.set_xlabel("Clase predicha")
    np.atleast_1d(axes)[0].set_ylabel("Clase real")
    titulo(fig, f"Matrices de confusión en la {DISPOSITIVO}",
           contexto(d) + "\nCada celda es el número de imágenes; el tono indica la proporción dentro de la clase real (escala realzada para que se vean los errores).", y=0.985)
    guardar(fig, "03_matrices_confusion.png")


def barras_h(ax, modelos, valores, etiquetas, unidad="", xmax=None):
    y = np.arange(len(modelos))
    ax.barh(y, valores, height=0.5, color=[COLOR[m["clave"]] for m in modelos])
    xmax = xmax or max(valores) * 1.3
    ax.set_xlim(0, xmax)
    for yi, v, e in zip(y, valores, etiquetas):
        ax.text(v + xmax * 0.012, yi, e, va="center", ha="left", fontsize=10)
    ax.set_yticks(y, [m["nombre"] for m in modelos], fontsize=10.5)
    ax.set_ylim(len(modelos) - 0.4, -0.6)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    if unidad:
        ax.set_xlabel(unidad)


def fig_latencia(d):
    modelos = d["modelos"]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.subplots_adjust(top=0.74, left=0.27, right=0.95)
    medias = [m["latencia_ms"]["media"] for m in modelos]
    barras_h(ax, modelos, medias,
             [f"{m['latencia_ms']['media']:.2f} ms  ·  {m['cuadros_por_segundo']:.0f} cuadros/s" for m in modelos],
             "Latencia media por imagen (ms)  —  menos es mejor", xmax=max(medias) * 1.45)
    n = d["imagenes"]["total"] - d["parametros"]["n_warmup"]
    titulo(fig, f"Latencia de inferencia en la {DISPOSITIVO}",
           contexto(d) + f"\nMedia de {n:,} inferencias por modelo (se descartan {d['parametros']['n_warmup']} de calentamiento).")
    guardar(fig, "04_latencia_media.png")


def fig_latencia_h1(d):
    modelos = d["modelos"]
    umbral = d["parametros"]["umbral_h1_ms"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    fig.subplots_adjust(top=0.72, left=0.27, right=0.95)
    maximos = [m["latencia_ms"]["max"] for m in modelos]
    etiquetas = [f"máx. {m['latencia_ms']['max']:.1f} ms  ·  margen ×{umbral / m['latencia_ms']['max']:.1f}" for m in modelos]
    dentro = [v > umbral * 0.55 for v in maximos]  # la etiqueta no cabe antes de la línea del límite: va dentro de la barra
    barras_h(ax, modelos, maximos, [("" if adentro else e) for e, adentro in zip(etiquetas, dentro)],
             "Latencia máxima observada (ms)", xmax=max(umbral, *maximos) * 1.06)
    for i, (v, e, adentro) in enumerate(zip(maximos, etiquetas, dentro)):
        if adentro:
            ax.text(v - umbral * 0.015, i, e, va="center", ha="right", fontsize=10, color="white", fontweight="bold")
    ax.axvline(umbral, color=CRITICO, lw=1.5)
    ax.text(umbral - 2, -0.52, f"Límite H1: {umbral:.0f} ms", ha="right", va="center", fontsize=9.5, color=TINTA, fontweight="bold")
    titulo(fig, "Latencia frente al límite de la hipótesis H1",
           contexto(d) + "\nSe grafica el peor caso (latencia máxima) de cada modelo contra el límite de 200 ms.")
    guardar(fig, "05_latencia_vs_limite_H1.png")


def fig_distribucion_latencia(d):
    modelos = d["modelos"]
    w = d["parametros"]["n_warmup"]
    fig, axes = plt.subplots(len(modelos), 1, figsize=(8, 1.9 * len(modelos) + 1.5))
    fig.subplots_adjust(top=0.82, hspace=0.75, left=0.08, right=0.96, bottom=0.1)
    for ax, m in zip(np.atleast_1d(axes), modelos):
        lat = np.array(m["por_imagen"]["latencia_ms"][w:])
        r = m["latencia_ms"]
        lo, hi = np.percentile(lat, 0.5), np.percentile(lat, 99.5)
        margen = max((hi - lo) * 0.15, 0.02)
        ax.hist(lat, bins=np.linspace(lo - margen, hi + margen, 60), color=COLOR[m["clave"]], edgecolor=SUPERFICIE, linewidth=0.6)
        ax.axvline(r["media"], color=TINTA, lw=1)
        ax.set_title(f"{m['nombre']}   media {r['media']:.2f} ms · desv. est. {r['std']:.2f} ms · P50 {r['p50']:.2f} · "
                     f"P95 {r['p95']:.2f} · P99 {r['p99']:.2f} · máx {r['max']:.2f}", fontsize=9.5, loc="left", color=TINTA)
        ax.grid(axis="y")
        ax.set_ylabel("Inferencias")
    np.atleast_1d(axes)[-1].set_xlabel("Latencia por imagen (ms)  —  cada panel tiene su propia escala; se muestra el 99% central")
    titulo(fig, "Distribución de la latencia por imagen", contexto(d) + "\nLa línea vertical marca la media.",
           y=0.985)
    guardar(fig, "06_distribucion_latencia.png")


def fig_latencia_secuencia(d):
    modelos = d["modelos"]
    w = d["parametros"]["n_warmup"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    fig.subplots_adjust(top=0.74, right=0.97)
    for m in modelos:
        lat = np.array(m["por_imagen"]["latencia_ms"])
        x = np.arange(1, len(lat) + 1)
        ax.plot(x[w:], lat[w:], color=COLOR[m["clave"]], lw=0.9, label=f"{m['nombre']} (media {m['latencia_ms']['media']:.2f} ms)")
    ax.set_xlim(0, len(lat))
    ax.set_ylim(0, None)
    ax.set_xlabel("Número de imagen evaluada")
    ax.set_ylabel("Latencia (ms)")
    ax.grid(axis="y")
    leyenda = fig.legend(loc="upper left", bbox_to_anchor=(0.015, 0.865), ncols=3, handlelength=1.4, columnspacing=1.4)
    for linea in leyenda.get_lines():
        linea.set_linewidth(3)
    titulo(fig, "Latencia de cada inferencia a lo largo de la prueba", contexto(d))
    guardar(fig, "07_latencia_por_imagen.png")


def fig_estabilidad(d):
    modelos = d["modelos"]
    paneles = [("latencia_ms", "Latencia media\npor segundo (ms)", True), ("cpu_percent", "Uso de CPU\ntotal (%)", True),
               ("temperatura_c", "Temperatura\ndel SoC (°C)", False), ("ram_mb", "Variación de\nRAM (MB)", False)]
    fig, axes = plt.subplots(len(paneles), 1, figsize=(9, 9), sharex=True)
    fig.subplots_adjust(top=0.865, hspace=0.22, left=0.12, right=0.97, bottom=0.07)
    for ax, (clave, etiqueta, desde_cero) in zip(axes, paneles):
        for m in modelos:
            s = m["estabilidad"]["series"]
            n = min(len(s["t_s"]), len(s[clave]))
            valores = np.array(s[clave][:n], dtype=float)
            if clave == "ram_mb":  # cada modelo corre en el mismo proceso: se grafica el cambio desde su propio inicio
                valores = valores - valores[0]
            ax.plot(s["t_s"][:n], valores, color=COLOR[m["clave"]], lw=1.8)
        ax.set_ylabel(etiqueta)
        ax.grid(axis="y")
        if desde_cero:
            ax.set_ylim(0, None)
        if clave == "cpu_percent":
            ax.set_ylim(0, 100)
        if clave == "ram_mb":
            ax.set_ylim(-5, 5)
    axes[-1].set_xlabel("Tiempo de ejecución continua (s)")
    axes[-1].set_xlim(0, modelos[0]["estabilidad"]["duracion_segundos"])
    leyenda_modelos(fig, modelos, y=0.925)
    dur = modelos[0]["estabilidad"]["duracion_segundos"]
    duracion = f"{dur // 60} minuto{'s' if dur // 60 > 1 else ''}" if dur % 60 == 0 else f"{dur} segundos"
    titulo(fig, f"Prueba de estabilidad: {duracion} de inferencia continua por modelo",
           contexto(d) + "\nUna muestra por segundo. RAM: cambio respecto al inicio de la prueba de cada modelo (0 = sin fugas).", y=0.99)
    guardar(fig, "08_estabilidad_series.png")


def fig_compromiso(d):
    modelos = d["modelos"]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(top=0.72)
    for m in modelos:
        x, y = m["latencia_ms"]["media"], m["clasificacion"]["accuracy"] * 100
        ax.scatter(x, y, s=m["tam_tflite_mb"] * 160, color=COLOR[m["clave"]], edgecolor=SUPERFICIE, linewidth=2, zorder=3)
        ax.annotate(f"{m['nombre']}\n{y:.2f}% · {x:.1f} ms · {m['tam_tflite_mb']:.2f} MB", (x, y), xytext=(0, -22),
                    textcoords="offset points", ha="center", va="top", fontsize=9.5, color=TINTA)
    xs = [m["latencia_ms"]["media"] for m in modelos]
    ys = [m["clasificacion"]["accuracy"] * 100 for m in modelos]
    ax.set_xlim(0, max(xs) * 1.25)
    ax.set_ylim(min(ys) - 1.6, max(ys) + 0.8)
    ax.set_xlabel("Latencia media por imagen (ms)  —  menos es mejor")
    ax.set_ylabel("Accuracy (%)  —  más es mejor")
    ax.grid(True)
    titulo(fig, f"Compromiso entre precisión y velocidad en la {DISPOSITIVO}",
           contexto(d) + "\nEl tamaño del círculo es proporcional al tamaño del archivo .tflite. Mejor: arriba a la izquierda.\nEl eje vertical no inicia en cero.")
    guardar(fig, "09_precision_vs_latencia.png")


def fig_recursos(d):
    modelos = d["modelos"]
    paneles = [
        ("Tamaño del archivo .tflite (MB)", [m["tam_tflite_mb"] for m in modelos], "{:.2f} MB"),
        ("RAM que ocupa el modelo cargado (MB)", [m["ram_aislada"]["incremento_modelo_mb"] for m in modelos], "{:.1f} MB"),
        ("Uso medio de CPU en inferencia continua (%)", [m["estabilidad"]["cpu_percent"]["media"] for m in modelos], "{:.1f}%"),
        ("Inferencias por segundo sostenidas", [m["estabilidad"]["inferencias_por_segundo"] for m in modelos], "{:.1f}"),
        ("Temperatura máxima (°C)", [m["estabilidad"]["temperatura_c"]["max"] for m in modelos], "{:.1f} °C"),
        ("Tiempo de carga del modelo (ms)", [m["carga_ms"] for m in modelos], "{:.0f} ms"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11, 7.4))
    fig.subplots_adjust(top=0.86, hspace=0.75, wspace=0.62, left=0.17, right=0.94, bottom=0.07)
    for ax, (nombre, valores, fmt) in zip(axes.flat, paneles):
        barras_h(ax, modelos, valores, [fmt.format(v) for v in valores])
        ax.set_title(nombre, fontsize=10, loc="left", color=TINTA)
    titulo(fig, f"Consumo de recursos en la {DISPOSITIVO}", contexto(d), y=0.985)
    guardar(fig, "10_recursos.png")


def fig_hilos(d1, d4):
    modelos = d1["modelos"]
    por_clave = {m["clave"]: m for m in d4["modelos"]}
    h4 = d4["parametros"]["num_threads"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    fig.subplots_adjust(top=0.76, left=0.25, right=0.93)
    y, alto = np.arange(len(modelos)), 0.3
    sep = max(m["latencia_ms"]["media"] for m in modelos) * 0.012
    for i, m in enumerate(modelos):
        a, b = m["latencia_ms"]["media"], por_clave[m["clave"]]["latencia_ms"]["media"]
        ax.barh(i - alto / 2 - 0.02, a, height=alto, color=COLOR[m["clave"]])
        ax.barh(i + alto / 2 + 0.02, b, height=alto, color=COLOR[m["clave"]], alpha=0.45)
        ax.text(a + sep, i - alto / 2 - 0.02, f"1 hilo: {a:.2f} ms", va="center", fontsize=9.5)
        ax.text(b + sep, i + alto / 2 + 0.02, f"{h4} hilos: {b:.2f} ms  (×{a / b:.2f} más rápido)", va="center", fontsize=9.5)
    ax.set_yticks(y, [m["nombre"] for m in modelos], fontsize=10.5)
    ax.set_ylim(len(modelos) - 0.45, -0.55)
    ax.set_xlim(0, max(m["latencia_ms"]["media"] for m in modelos) * 1.35)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.set_xlabel("Latencia media por imagen (ms)  —  menos es mejor")
    titulo(fig, f"Efecto del número de hilos: 1 hilo frente a {h4} hilos",
           f"{DISPOSITIVO} · {d1['imagenes']['total']:,} imágenes de prueba · {d1['runtime']}\n"
           f"Barra sólida: 1 hilo (configuración por defecto). Barra clara: {h4} hilos.")
    guardar(fig, "11_latencia_1_vs_4_hilos.png")


def _miniatura(ax, ruta):
    with Image.open(ruta) as img:
        img = img.convert("RGB")
        lado = min(img.size)
        izq, sup = (img.width - lado) // 2, (img.height - lado) // 2
        ax.imshow(img.crop((izq, sup, izq + lado, sup + lado)).resize((224, 224), Image.BILINEAR))
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True), s.set_color(EJE)


def _pie_predicciones(ax, d, idx, real):
    ax.text(0, -0.05, f"Real: {real}", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", color=TINTA)
    for k, m in enumerate(d["modelos"]):
        pred, conf = m["por_imagen"]["predicha"][idx], m["por_imagen"]["confianza"][idx]
        y = -0.17 - 0.115 * k
        ax.plot(0.03, y - 0.032, "s", color=COLOR[m["clave"]], markersize=6, transform=ax.transAxes, clip_on=False)
        ax.text(0.09, y, f"{'✓' if pred == real else '✗'} {pred} ({conf * 100:.0f}%)", transform=ax.transAxes, fontsize=8.5,
                va="top", color=TINTA if pred == real else CRITICO, fontweight="normal" if pred == real else "bold")


def fig_mosaico(d, dir_test, indices, nombre, tit, sub, columnas=5):
    filas = math.ceil(len(indices) / columnas)
    fig, axes = plt.subplots(filas, columnas, figsize=(2.55 * columnas, 3.75 * filas + 1.3))
    fig.subplots_adjust(top=1 - 1.25 / (3.75 * filas + 1.3), hspace=0.5, wspace=0.12, left=0.03, right=0.97, bottom=0.11 / filas + 0.04)
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for ax, idx in zip(axes.flat, indices):
        ax.axis("on")
        _miniatura(ax, dir_test / d["imagenes"]["ruta"][idx])
        _pie_predicciones(ax, d, idx, d["imagenes"]["real"][idx])
    leyenda_modelos(fig, d["modelos"], y=1 - 0.78 / (3.75 * filas + 1.3))
    titulo(fig, tit, sub, y=1 - 0.1 / (3.75 * filas + 1.3))
    guardar(fig, nombre)


def figs_mosaicos(d, dir_test, seleccion):
    """Mosaicos de imágenes. Si hay selección manual (ruta relativa de cada imagen) se usa; si no, se eligen al azar."""
    reales = d["imagenes"]["real"]
    preds = [m["por_imagen"]["predicha"] for m in d["modelos"]]
    n = len(reales)
    pos = {r: i for i, r in enumerate(d["imagenes"]["ruta"])}
    rng = np.random.default_rng(7)

    if seleccion.get("ejemplos"):
        muestra = [pos[r] for r in seleccion["ejemplos"]]
        sub = "Imágenes seleccionadas del conjunto de prueba. Bajo cada imagen: clase real y predicción de cada modelo con su confianza."
    else:  # 2 imágenes al azar por clase (semilla fija para que sea reproducible)
        muestra = []
        for c in d["clases"]:
            candidatos = [i for i in range(n) if reales[i] == c]
            muestra += sorted(rng.choice(candidatos, size=2, replace=False).tolist())
        sub = "Dos imágenes al azar por clase. Bajo cada imagen: clase real y predicción de cada modelo con su confianza."
    fig_mosaico(d, dir_test, muestra, "12_ejemplos_predicciones.png", f"Ejemplos de predicciones hechas en la {DISPOSITIVO}", sub)

    # Casos difíciles: imágenes en las que fallaron los tres modelos
    fallan_todos = [i for i in range(n) if all(p[i] != reales[i] for p in preds)]
    if seleccion.get("casos_dificiles"):
        elegidos = [pos[r] for r in seleccion["casos_dificiles"]]
        if any(i not in fallan_todos for i in elegidos):
            raise SystemExit("La selección de casos difíciles incluye imágenes que no fallaron los tres modelos")
        sub = f"{len(fallan_todos)} de {n:,} imágenes fueron mal clasificadas por los tres modelos; se muestran {len(elegidos)}."
    elif fallan_todos:
        elegidos = sorted(rng.choice(fallan_todos, size=min(10, len(fallan_todos)), replace=False).tolist())
        sub = f"{len(fallan_todos)} de {n:,} imágenes fueron mal clasificadas por los tres modelos; se muestran {len(elegidos)} al azar."
    else:
        return
    fig_mosaico(d, dir_test, elegidos, "13_errores_comunes.png",
                "Casos difíciles: imágenes que los tres modelos clasificaron mal", sub)


def fig_coincidencia_errores(d):
    """Cuántas imágenes falla cada modelo en exclusiva y cuántas fallan varios a la vez."""
    reales = d["imagenes"]["real"]
    modelos = d["modelos"]
    fallo = np.array([[p != r for p, r in zip(m["por_imagen"]["predicha"], reales)] for m in modelos])
    cuantos = fallo.sum(axis=0)
    etiquetas = ["Solo este modelo", "Este y otro modelo", "Los tres modelos"]
    tonos = [1.0, 0.6, 0.3]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    fig.subplots_adjust(top=0.7, left=0.25, right=0.95)
    for i, m in enumerate(modelos):
        izq = 0
        for k, (e, a) in enumerate(zip(etiquetas, tonos), start=1):
            v = int((fallo[i] & (cuantos == k)).sum())
            ax.barh(i, v, left=izq, height=0.5, color=COLOR[m["clave"]], alpha=a, edgecolor=SUPERFICIE, linewidth=2)
            if v >= 12:
                ax.text(izq + v / 2, i, str(v), ha="center", va="center", fontsize=9.5,
                        color="white" if a == 1.0 else TINTA, fontweight="bold")
            izq += v
        ax.text(izq + 3, i, f"{izq} errores", va="center", fontsize=10)
    ax.set_yticks(range(len(modelos)), [m["nombre"] for m in modelos], fontsize=10.5)
    ax.set_ylim(len(modelos) - 0.4, -0.6)
    ax.set_xlim(0, fallo.sum(axis=1).max() * 1.18)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.set_xlabel("Imágenes mal clasificadas")
    manijas = [plt.Rectangle((0, 0), 1, 1, color=TINTA_2, alpha=a, label=e) for e, a in zip(etiquetas, tonos)]
    fig.legend(handles=manijas, loc="upper left", bbox_to_anchor=(0.015, 0.83), ncols=3, handlelength=1.2, columnspacing=1.6)
    titulo(fig, "¿Los modelos se equivocan en las mismas imágenes?",
           contexto(d) + "\nCada barra reparte los errores de un modelo según cuántos modelos fallaron en esa misma imagen.")
    guardar(fig, "14_coincidencia_errores.png")


# --------------------------------------------------------------------------
# Capturas de pantalla (SVG generados en la Raspberry -> PNG)
# --------------------------------------------------------------------------
_TEXTO = re.compile(r'(<text class="[^"]*" x=")([\d.]+)(" y="[\d.]+" textLength=")([\d.]+)("[^>]*>)((?:&#160;)*)(.*?)((?:&#160;)*)(</text>)')
ANCHO_CARACTER = 12.2  # ancho de celda que usa rich al exportar a SVG


def _arreglar_espacios(m):
    """rsvg-convert descarta los espacios al inicio/fin de cada <text>; se compensa moviendo el texto."""
    ini, fin = m.group(6).count("&#160;"), m.group(8).count("&#160;")
    if not m.group(7):
        return ""
    x = float(m.group(2)) + ini * ANCHO_CARACTER
    largo = float(m.group(4)) - (ini + fin) * ANCHO_CARACTER
    return f"{m.group(1)}{x:.1f}{m.group(3)}{largo:.1f}{m.group(5)}{m.group(7)}{m.group(9)}"


def convertir_capturas(dir_svg):
    if not shutil.which("rsvg-convert"):
        print("  (se omiten las capturas: falta el comando rsvg-convert)")
        return
    destino = DIR_FIGURAS / "capturas"
    destino.mkdir(parents=True, exist_ok=True)
    for svg in sorted(dir_svg.glob("*.svg")):
        texto = svg.read_text(encoding="utf-8")
        texto = texto.replace('"Fira Code"', '"JetBrains Mono NL", "DejaVu Sans Mono"')
        texto = texto.replace("Fira Code, monospace", "'JetBrains Mono NL', 'DejaVu Sans Mono', monospace")
        texto = _TEXTO.sub(_arreglar_espacios, texto)
        salida = destino / f"captura_{svg.stem}.png"
        subprocess.run(["rsvg-convert", "-z", "1.5", "-o", str(salida)], input=texto.encode("utf-8"), check=True)
        print(f"  {mostrar_ruta(salida)}")


def main():
    ap = argparse.ArgumentParser(description="Gráficas de evidencia del benchmark en Raspberry Pi")
    ap.add_argument("--resultados", type=Path, default=DIR_DATOS / "resultados_rpi_1hilo.json")
    ap.add_argument("--resultados-hilos", type=Path, default=DIR_DATOS / "resultados_rpi_4hilos.json")
    ap.add_argument("--capturas", type=Path, default=DIR_DATOS / "capturas_svg")
    ap.add_argument("--test-dir", type=Path, default=DIR_TEST, help="Set de prueba local, para los mosaicos de imágenes")
    ap.add_argument("--salida", type=Path, default=DIR_FIGURAS)
    ap.add_argument("--seleccion", type=Path, default=DIR_DATOS / "seleccion_mosaicos.json",
                    help="JSON con las imágenes elegidas a mano para los mosaicos: {\"ejemplos\": [...], \"casos_dificiles\": [...]}")
    args = ap.parse_args()

    globals()["DIR_FIGURAS"] = args.salida
    args.salida.mkdir(parents=True, exist_ok=True)
    d = json.loads(args.resultados.read_text(encoding="utf-8"))
    placa = re.search(r"Raspberry Pi \d+", d["dispositivo"].get("placa", ""))
    globals()["DISPOSITIVO"] = placa.group(0) if placa else d["dispositivo"].get("placa", DISPOSITIVO)
    print(f"Resultados: {args.resultados}  ({d['dispositivo'].get('placa', '')}, {d['fecha_inicio']})")

    print("Gráficas:")
    fig_accuracy(d)
    fig_accuracy_por_clase(d)
    fig_matrices(d)
    fig_latencia(d)
    fig_latencia_h1(d)
    fig_distribucion_latencia(d)
    fig_latencia_secuencia(d)
    fig_estabilidad(d)
    fig_compromiso(d)
    fig_recursos(d)
    if args.resultados_hilos.exists():
        fig_hilos(d, json.loads(args.resultados_hilos.read_text(encoding="utf-8")))
    fig_coincidencia_errores(d)
    if args.test_dir.is_dir():
        seleccion = json.loads(args.seleccion.read_text(encoding="utf-8")) if args.seleccion.exists() else {}
        figs_mosaicos(d, args.test_dir, seleccion)
    else:
        print(f"  (se omiten los mosaicos: no existe {args.test_dir})")

    if args.capturas.is_dir():
        print("Capturas de pantalla:")
        convertir_capturas(args.capturas)


if __name__ == "__main__":
    main()
