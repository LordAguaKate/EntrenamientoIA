"""
Convierte docs/reporte_benchmark_modelos.md en docs/reporte_benchmark_modelos.pdf,
con portada, índice, encabezados, pies de página y figuras numeradas.

Usa WeasyPrint y Markdown, que NO están en requirements.txt; se corre en un
entorno aislado con uv para no tocar el venv del proyecto (desde la raíz del repo):

    uv run --no-project --python 3.12 --with weasyprint --with markdown \
        tests/generar_pdf_reporte.py

Vuelve a correrlo cada vez que edites el .md. Para otro reporte (por ejemplo, el
de la Raspberry Pi 4) se pasan --md, --salida y --datos; la placa y la fecha se
toman del JSON.
"""
import argparse
import json
import re
from pathlib import Path

import markdown
from weasyprint import HTML

REPO = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(description="Convierte un reporte de benchmark en Markdown a PDF")
ap.add_argument("--md", type=Path, default=REPO / "docs" / "reporte_benchmark_modelos.md")
ap.add_argument("--salida", type=Path, default=REPO / "docs" / "reporte_benchmark_modelos.pdf")
ap.add_argument("--datos", type=Path, default=REPO / "reports" / "benchmark_rpi" / "resultados_rpi_1hilo.json")
ap.add_argument("--subtitulo", default="Comparación de MobileNetV3-Large, EfficientNetB0 y ShuffleNetV2 (ImageNet) "
                "en la laptop de desarrollo y en la Raspberry Pi 5")
ap.add_argument("--recomendado", default="mobilenet", help="Clave del modelo que lleva el sello de recomendado")
args = ap.parse_args()
MD, SALIDA = args.md, args.salida
DATOS = json.loads(args.datos.read_text())
_placa = re.search(r"Raspberry Pi \d+", DATOS["dispositivo"]["placa"])
DISPOSITIVO = _placa.group(0) if _placa else DATOS["dispositivo"]["placa"]
_MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
_a, _m, _d = (int(x) for x in DATOS["fecha_inicio"][:10].split("-"))
FECHA = f"{_d} de {_MESES[_m - 1]} de {_a}"

texto = MD.read_text(encoding="utf-8")
titulo_md, cuerpo = texto.split("\n", 1)
titulo = titulo_md.lstrip("# ").strip()

# La introducción (antes de la sección 1) se vuelve una sección sin número
intro, resto = cuerpo.split("\n## 1.", 1)
intro = re.sub(r"^\s*#.*$", "", intro, flags=re.M).replace("\n---\n", "\n")
cuerpo = "## Introducción {: .sin-num }\n" + intro + "\n## 1." + resto
cuerpo = cuerpo.replace("\n---\n", "\n")

html = markdown.markdown(cuerpo, extensions=["tables", "fenced_code", "sane_lists", "toc", "attr_list"],
                         extension_configs={"toc": {"slugify": lambda v, s: re.sub(r"[^a-z0-9]+", "-", v.lower()).strip("-")}})

# Imágenes sueltas -> figuras con pie numerado
html = re.sub(r'<p><img alt="([^"]*)" src="([^"]*)" ?/?></p>',
              r'<figure><img src="\2" alt="\1"/><figcaption>\1</figcaption></figure>', html)
# Columnas numéricas alineadas a la derecha
def celda(m):
    contenido = re.sub(r"<[^>]+>", "", m.group(1)).strip()
    es_num = bool(re.match(r"^[\-−+×~]?\s?\d[\d.,]*\s?(%|ms|MB|GB|°C|s|pp|err\.)?", contenido)) and len(contenido) < 40
    if not es_num:
        return m.group(0)
    return f'<td class="num{" corto" if len(contenido) <= 14 else ""}">{m.group(1)}</td>'
html = re.sub(r"<td>(.*?)</td>", celda, html)
# Tablas anchas (8+ columnas) con letra más pequeña
html = re.sub(r"<table>(\s*<thead>\s*<tr>(?:\s*<th[^>]*>.*?</th>){8,})", r'<table class="ancha">\1', html, flags=re.S)
# Los títulos de las tablas del md llevan negritas al inicio de un párrafo: se mantienen

# Índice
entradas = re.findall(r'<h2 [^>]*id="([^"]+)"[^>]*>(.*?)</h2>', html)
indice = "".join(f'<li><a href="#{i}">{re.sub("<[^>]+>", "", t)}</a></li>' for i, t in entradas)

# Portada con las cifras clave
COLOR = {"mobilenet": "#2a78d6", "efficientnet": "#eb6834", "shufflenet": "#1baf7a"}
tarjetas = ""
for m in DATOS["modelos"]:
    c = m["clasificacion"]
    rec = ' <span class="sello">Recomendado</span>' if m["clave"] == args.recomendado else ""
    tarjetas += f'''
    <div class="tarjeta" style="border-top-color:{COLOR[m["clave"]]}">
      <div class="t-nombre">{m["nombre"]}{rec}</div>
      <div class="t-fila"><span>Accuracy</span><b>{c["accuracy"] * 100:.2f}%</b></div>
      <div class="t-fila"><span>Latencia media</span><b>{m["latencia_ms"]["media"]:.2f} ms</b></div>
      <div class="t-fila"><span>Cuadros/s</span><b>{m["cuadros_por_segundo"]:.1f}</b></div>
      <div class="t-fila"><span>Archivo .tflite</span><b>{m["tam_tflite_mb"]:.2f} MB</b></div>
    </div>'''

portada = f'''
<section class="portada">
  <div class="p-sup">Proyecto EcoSort · Clasificación de residuos</div>
  <h1 class="p-titulo">Reporte de pruebas de modelos ligeros</h1>
  <div class="p-sub">{args.subtitulo}</div>
  <div class="p-linea"></div>
  <div class="p-meta">
    <div><span>Dispositivo de despliegue</span>{DATOS["dispositivo"]["placa"]}</div>
    <div><span>Conjunto de prueba</span>{DATOS["imagenes"]["total"]:,} imágenes · 5 clases</div>
    <div><span>Fecha de las pruebas</span>{FECHA}</div>
  </div>
  <div class="p-tarjetas">{tarjetas}</div>
  <div class="p-nota">Cifras medidas en la {DISPOSITIVO} con modelos .tflite cuantizados, 1 hilo de inferencia (runtime {DATOS["runtime"]}).</div>
</section>
<section class="indice">
  <h2 class="sin-num">Contenido</h2>
  <ol>{indice}</ol>
</section>
'''

CSS = '''
@page {
  size: letter; margin: 22mm 18mm 20mm 18mm;
  @top-left { content: "EcoSort · Reporte de pruebas de modelos"; font: 8pt "Noto Sans"; color: #898781; }
  @top-right { content: string(seccion); font: 8pt "Noto Sans"; color: #898781; }
  @bottom-right { content: counter(page) " / " counter(pages); font: 8pt "Noto Sans"; color: #898781; }
  @bottom-left { content: "DISPOSITIVO · MobileNetV3-Large · EfficientNetB0 · ShuffleNetV2"; font: 8pt "Noto Sans"; color: #b8b7b0; }
}
@page :first { margin: 0; @top-left { content: none } @top-right { content: none } @bottom-right { content: none } @bottom-left { content: none } }
html { font-family: "Noto Sans", "DejaVu Sans", sans-serif; font-size: 9.6pt; line-height: 1.5; color: #1d1d1b; }
body { margin: 0; }
a { color: #1c5cab; text-decoration: none; }
strong { color: #0b0b0b; }

/* Portada */
.portada { page: portada; height: 279.4mm; box-sizing: border-box; padding: 34mm 22mm 18mm 22mm;
  background: linear-gradient(160deg, #0d366b 0%, #184f95 55%, #2a78d6 100%); color: #fff; position: relative; }
.p-sup { font-size: 10pt; letter-spacing: .14em; text-transform: uppercase; color: #b7d3f6; }
.p-titulo { font-size: 30pt; line-height: 1.15; margin: 10mm 0 4mm; font-weight: 700; color: #fff; }
.p-sub { font-size: 13pt; color: #dbe9fb; max-width: 150mm; }
.p-linea { width: 40mm; height: 3px; background: #86b6ef; margin: 12mm 0 10mm; }
.p-meta { display: flex; gap: 10mm; font-size: 10pt; }
.p-meta div { display: flex; flex-direction: column; }
.p-meta span { font-size: 8pt; text-transform: uppercase; letter-spacing: .08em; color: #9ec5f4; margin-bottom: 1mm; }
.p-tarjetas { display: flex; gap: 5mm; margin-top: 22mm; }
.tarjeta { flex: 1; background: #fcfcfb; color: #1d1d1b; border-radius: 3mm; border-top: 2.2mm solid; padding: 5mm 5mm 4mm; }
.t-nombre { font-weight: 700; font-size: 10.5pt; margin-bottom: 3mm; min-height: 11mm; }
.t-fila { display: flex; justify-content: space-between; font-size: 9pt; padding: 1.3mm 0; border-bottom: 1px solid #ecebe6; }
.t-fila:last-child { border-bottom: none; }
.t-fila span { color: #52514e; }
.sello { display: inline-block; font-size: 6.8pt; font-weight: 700; background: #e6f4e6; color: #006300; border-radius: 2mm;
  padding: .3mm 2mm; margin-left: 1mm; vertical-align: 1px; letter-spacing: .03em; }
.p-nota { position: absolute; bottom: 18mm; left: 22mm; right: 22mm; font-size: 8.5pt; color: #b7d3f6; }

/* Índice */
.indice { padding-top: 6mm; margin-bottom: 10mm; }
.indice ol { list-style: none; padding: 0; margin: 6mm 0 0; counter-reset: none; }
.indice li { font-size: 11pt; padding: 2.4mm 0; border-bottom: 1px solid #e1e0d9; }
.indice li a { color: #1d1d1b; }
.indice li a::after { content: target-counter(attr(href url), page); float: right; color: #52514e; }

/* Títulos */
h2 { string-set: seccion content(); break-before: page; font-size: 17pt; color: #0d366b; margin: 0 0 5mm;
  padding-bottom: 2.5mm; border-bottom: 2px solid #2a78d6; }
h2.sin-num { break-before: auto; }
h3 { font-size: 12pt; color: #184f95; margin: 7mm 0 2.5mm; break-after: avoid; }
p { margin: 0 0 2.6mm; text-align: justify; hyphens: auto; }
p:has(+ table), p:has(+ pre) { break-after: avoid; }
ul, ol { margin: 0 0 3mm; padding-left: 5.5mm; }
li { margin-bottom: 1.4mm; }

/* Tablas */
table { width: 100%; border-collapse: collapse; margin: 2mm 0 5mm; font-size: 8.6pt; break-inside: auto; }
table.ancha { font-size: 7.6pt; }
thead { display: table-header-group; }
th { background: #0d366b; color: #fff; font-weight: 600; text-align: left; padding: 1.8mm 2mm; vertical-align: bottom; }
td { padding: 1.5mm 2mm; border-bottom: 1px solid #e1e0d9; vertical-align: top; }
tr { break-inside: avoid; }
tbody tr:nth-child(even) td { background: #f5f8fc; }
td.num { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
table.ancha td.num { white-space: normal; }
table.ancha td.num.corto { white-space: nowrap; }
th code { background: rgba(255,255,255,.16); color: #fff; }
table.ancha th { padding: 1.6mm 1.5mm; } table.ancha td { padding: 1.4mm 1.5mm; }
td code, p code, li code { font-size: 8pt; }

/* Código y citas */
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.4pt; background: #f0efec; padding: .2mm 1mm; border-radius: 1mm; color: #3a3936; }
pre { background: #f5f5f2; border: 1px solid #e1e0d9; border-left: 3px solid #2a78d6; border-radius: 1.5mm; padding: 3mm 4mm;
  font-size: 7.8pt; line-height: 1.45; break-inside: avoid; white-space: pre-wrap; }
pre code { background: none; padding: 0; font-size: inherit; }
blockquote { margin: 3mm 0 4mm; padding: 3.5mm 5mm; background: #eef5fd; border-left: 4px solid #2a78d6; border-radius: 0 2mm 2mm 0;
  color: #0d366b; break-inside: avoid; }
blockquote p { margin: 0; }

/* Figuras */
body { counter-reset: figura; }
figure { margin: 3mm 0 6mm; break-inside: avoid; text-align: center; counter-increment: figura; }
figure img { max-width: 100%; max-height: 150mm; }
figcaption { font-size: 8.4pt; color: #52514e; margin-top: 1.5mm; }
figcaption::before { content: "Figura " counter(figura) ". "; font-weight: 700; color: #184f95; }
'''

documento = f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><title>{titulo}</title>
<style>{CSS.replace("DISPOSITIVO", DISPOSITIVO)}</style></head><body>{portada}{html}</body></html>'''
HTML(string=documento, base_url=str(MD.parent) + "/").write_pdf(SALIDA)
print(SALIDA, f"{SALIDA.stat().st_size / 2**20:.1f} MB")
