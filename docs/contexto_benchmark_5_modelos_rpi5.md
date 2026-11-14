# Contexto para continuar: benchmark de 5 modelos en la Raspberry Pi 5

Documento de traspaso entre sesiones (escrito el 17/10/2026). Resume lo que ya se hizo, lo que se descubrió al revisar los dos modelos nuevos y los pasos pendientes para correr el benchmark de **5 modelos** en la Raspberry Pi 5 y generar su reporte.

---

## 1. Objetivo

1. Correr en la **Raspberry Pi 5** el benchmark de inferencia de los **5 modelos**, todos juntos y en las mismas condiciones:
   - MobileNetV3-Large
   - EfficientNetB0
   - ShuffleNetV2 (ImageNet)
   - **SqueezeNet v1.1 (ImageNet)**, nuevo
   - **YOLO11s-cls**, nuevo
2. Generar un reporte nuevo (`.md` + `.pdf`) con gráficas y capturas, con el mismo estilo que `docs/reporte_benchmark_modelos.md`.
3. **El reporte nuevo solo incluye resultados medidos en la Raspberry.** Nada de la laptop: ni `.keras` en GPU, ni `.tflite` en x86, ni tablas "laptop frente a Raspberry". Las cifras de la laptop de la sección 4 de este documento sirven únicamente para verificar que todo funciona, no para el reporte.

---

## 2. Lo que ya existe

| Elemento | Ubicación | Notas |
|---|---|---|
| Reporte de 3 modelos en Pi 5 (13/10) | `docs/reporte_benchmark_modelos.md` / `.pdf` | Incluye resultados de la laptop. No modificarlo; el nuevo es un documento aparte. |
| Datos crudos Pi 5, 3 modelos | `reports/benchmark_rpi/` | `resultados_rpi_1hilo.json`, `resultados_rpi_4hilos.json`, logs, `capturas_svg/`, `seleccion_mosaicos.json` |
| Gráficas Pi 5, 3 modelos | `reports/figures/raspberry/` | 14 gráficas, `capturas/` y `escritorio/` |
| Script del benchmark | `pruebas_modelos/raspberry/benchmark_visual.py` | Corre en la Raspberry. Hoy solo soporta modelos NHWC con entrada 0–255 (ver sección 5). |
| Script de gráficas | `pruebas_modelos/reportes/graficar_benchmark_rpi.py` | Corre en la laptop. Ya toma la placa del JSON. Pensado para 3 modelos. |
| Script del PDF | `pruebas_modelos/reportes/generar_pdf_reporte.py` | Acepta `--md`, `--salida`, `--datos`, `--subtitulo` y `--recomendado`. |

**Entornos en la laptop:**

- `venv/` del proyecto (Python 3.12, TensorFlow, matplotlib, rich, psutil) para `graficar_benchmark_rpi.py`.
- El `python3` del sistema **no tiene numpy ni matplotlib**: usar `venv/bin/python`.
- El PDF se genera con `uv run --no-project --python 3.12 --with weasyprint --with markdown pruebas_modelos/reportes/generar_pdf_reporte.py ...`.
- `rsvg-convert` está instalado; se usa para convertir las capturas SVG a PNG.

---

## 3. Los 5 modelos

| Clave sugerida | Modelo | `.tflite` | Tamaño | Entrada | Pixeles | Preprocesamiento de entrenamiento |
|---|---|---|---|---|---|---|
| `mobilenet` | MobileNetV3-Large | `models/MobileNetV3/modelo_residuos_rpi_V3_05.tflite` | 3.24 MB | `[1,224,224,3]` NHWC float32 | 0–255 (normaliza dentro) | RGB, estirar a 224×224 con *nearest* |
| `efficientnet` | EfficientNetB0 | `models/EfficientNetB0/modelo_residuos_EfficientNetB0_v1.tflite` | 4.50 MB | igual | 0–255 | igual |
| `shufflenet` | ShuffleNetV2 (ImageNet) | `models/ShuffleNetV2/modelo_residuos_shufflenetV2_imagenet.tflite` | 1.50 MB | igual | 0–255 | igual |
| `squeezenet` | SqueezeNet v1.1 (ImageNet) | `models/SqueezeNet/modelo_residuos_SqueezeNet_imagenet.tflite` | 0.81 MB | igual | 0–255 (capa `Normalization` con media de ImageNet) | igual (`flow_from_directory`) |
| `yolo` | YOLO11s-cls | `models/YOLO/modelo_residuos_YOLO_v1.tflite` | 5.39 MB | **`[1,3,224,224]` NCHW** float32 | **0–1** (dividir entre 255) | **Ultralytics:** RGB, lado corto a 224 con interpolación **bilineal** y **recorte central** de 224×224 |

Las 5 salidas son `[1,5]` float32 con softmax, en el orden `aluminio, carton, otros, plastico, vidrio`. Todos los modelos usan cuantización de rango dinámico (pesos int8, activaciones float32) y solo operaciones nativas de TFLite. Los 5 cargan en `ai-edge-litert` 2.2.0, el runtime de la Raspberry.

**Detalles de los modelos nuevos:**

- **SqueezeNet** (autor: jesus, commits `0564eef` y `f0313e1`): notebooks en `notebooks/SqueezeNet/`. Mismo encabezado y las mismas 2 fases (15 + 20 épocas) que los otros modelos Keras. Su autor reportó 91.55% con el `.keras`.
- **YOLO** (autor: yisus050820, commit `18f4115`): notebooks en `notebooks/YOLO/`, base `yolo11s-cls.pt`, 2 fases (15 épocas con `freeze=10` + 20 con `freeze=7`). Detalles:
  - Se exportó `last.pt`, no `best.pt`, con `format="litert", quantize="w8a32"`.
  - Su autor reportó 97.31% con el `.pt` en GPU, sobre 4,535 imágenes porque incluye `.webp`.
  - `modelo_residuos_YOLO_v1_w8a32.tflite` es **idéntico byte por byte** a `modelo_residuos_YOLO_v1.tflite` (mismo sha256): usar solo uno.
  - Su *data augmentation* es distinto (el de Ultralytics: recortes aleatorios, RandAugment, sin rotación). Hay que mencionarlo en el reporte.
  - YOLO11s-cls hace unas 4 veces más operaciones que EfficientNetB0 (~1.5 GFLOPs a 224 px).
- **Validación con `test/`:** todos los modelos, no solo YOLO, usaron `test/` como validación durante el entrenamiento. Ninguno usa *early stopping* ni guarda el mejor modelo según `test/`, así que la evaluación sigue siendo válida. Conviene mencionarlo en una línea en el reporte.

---

## 4. Verificación previa en la laptop (solo para comprobar; NO va en el reporte)

Se evaluaron los `.tflite` con `ai-edge-litert` 2.2.0 en x86, con 1 hilo y las mismas 4,534 imágenes que usa el benchmark (sin `.gif`, `.HEIC` ni `.webp`):

| Modelo / preprocesamiento | Accuracy | Errores | Latencia laptop (1 hilo) |
|---|---|---|---|
| SqueezeNet (estirar *nearest*, 0–255) | 91.80% | 372 | 6.52 ms |
| MobileNetV3-Large (referencia) | 96.27% | 169 | 7.22 ms |
| EfficientNetB0 (referencia) | 96.56% | 156 | 15.69 ms |
| YOLO **sin adaptar** (0–255, *nearest*) | 22.45% | 3,516 | — |
| YOLO estirado *nearest*, /255 | 95.54% | 202 | 14.03 ms |
| **YOLO preprocesamiento Ultralytics, /255** | **97.46%** | **115** | 14.08 ms |

**Para qué sirven estas cifras:** al correr en la Pi 5, la accuracy de cada modelo debería quedar a unas décimas de estos valores. Por la diferencia de kernels ARM/x86, en la corrida anterior hubo hasta ±0.2 puntos. Si YOLO da ~22% o ~95.5%, el preprocesamiento está mal.

**Latencia estimada de YOLO:** EfficientNetB0 fue 2.96 veces más lento en la Pi 5 que en la laptop. Si YOLO escala igual, tardaría unos 30–42 ms en la Pi 5. Es una estimación; hay que medirla.

**Código del preprocesamiento de YOLO** (validado: da 97.46%):

```python
def recorte_central_bilineal(img, lado=224):
    """Igual que classify_transforms de Ultralytics: lado corto a 224 (bilineal) + recorte central."""
    img = img.convert("RGB")
    w, h = img.size
    e = lado / min(w, h)
    img = img.resize((max(lado, round(w * e)), max(lado, round(h * e))), Image.BILINEAR)
    w, h = img.size
    l, t = (w - lado) // 2, (h - lado) // 2
    return np.asarray(img.crop((l, t, l + lado, t + lado)), dtype=np.uint8)

# Al inferir: x = arr.astype(np.float32) / 255.0 ; x = x.transpose(2, 0, 1)[None]   # -> [1,3,224,224]
```

---

## 5. Cambios pendientes en los scripts

### 5.1 `pruebas_modelos/raspberry/benchmark_visual.py`

1. **Configuración por modelo** en la lista `MODELOS`. Agregar a cada entrada:
   - `layout`: `"NHWC"` o `"NCHW"`.
   - `escala`: `1.0` o `1/255`.
   - `preprocesamiento`: `"estirar_nearest"` o `"recorte_central_bilineal"`.

   Agregar SqueezeNet (`clave: squeezenet`) y YOLO (`clave: yolo`) con colores nuevos de `rich`. El orden de ejecución sugerido es: MobileNetV3, EfficientNetB0, ShuffleNetV2, SqueezeNet, YOLO.
2. **Tamaño de entrada:** hoy se calcula con `shape[1], shape[2]` y el script se detiene si los modelos no coinciden. Hay que leerlo según el `layout`: en NCHW son `shape[2], shape[3]`.
3. **Imágenes en memoria:** hoy se cargan una sola vez para todos los modelos. Hay que cargar **un juego por cada preprocesamiento distinto**: dos juegos de ~680 MB cada uno en uint8, lo cual cabe sin problema en 16 GB. Reportar el tiempo de preprocesamiento de cada juego por separado.
4. **`inferir()`:** la conversión de tipo, la escala y la transposición deben quedar **fuera del cronómetro**, igual que hoy la conversión de tipo. Se sigue cronometrando solo `set_tensor + invoke + get_tensor`.
5. **Rutas de los modelos:** desde la reorganización del repo están en `models/<Arquitectura>/`. El valor por defecto `aqui.parent / "models"` ya no los encuentra. Opciones:
   - poner la subcarpeta en cada entrada de `MODELOS`;
   - buscar el archivo con `rglob` dentro de `--models-dir`.

   En la Raspberry no importa, porque ahí se copian planos a `modelos/`.
6. **Textos fijos:** buscar "tres modelos" u otros que asuman 3 modelos. La ficha de "Preprocesamiento" de la etapa 1 debe mostrar el preprocesamiento de cada modelo; puede ir como columna en la tabla "Modelos a evaluar".
7. Guardar en el JSON, por modelo: `layout`, `escala` y `preprocesamiento`. Además: el tiempo de carga de cada juego de imágenes y el **SHA-256 de cada `.tflite`**, para dejar constancia de qué archivo exacto se midió.
8. Antes de llevarlo a la Pi, probarlo en la laptop con pocas imágenes, por ejemplo `--max-imagenes 100 --duracion-estabilidad 5 --enfriamiento 0`. Comprobar que YOLO acierta igual que los demás y no da 22%.

### 5.2 `pruebas_modelos/reportes/graficar_benchmark_rpi.py`

1. `COLOR`: agregar `squeezenet` y `yolo`. Mantener la paleta validada para daltonismo (ver el skill `dataviz`), sin repetir tonos.
2. `fig_accuracy`: el eje está fijo en `set_xlim(92, 98.5)` y SqueezeNet (~91.8%) quedaría fuera. Calcular los límites a partir de los datos.
3. `fig_coincidencia_errores`:
   - las etiquetas "Solo este modelo / Este y otro modelo / Los tres modelos" asumen 3 modelos;
   - hay que generalizar a k = 1..5, o agrupar en "solo este", "2–4 modelos" y "los 5".
4. Mosaicos (`_pie_predicciones`, `fig_mosaico`): cada imagen lista una línea por modelo. Con 5 modelos hace falta más alto por fila o letra más chica.
5. `seleccion_mosaicos.json`: los "casos difíciles" deben ser imágenes que fallen **los 5 modelos**. Si alguna de la selección anterior no falla en los 5, el script se detiene con `SystemExit`. Generar una selección nueva para esta corrida.
6. Revisar las demás figuras con 5 barras (alturas y leyendas en una fila) y que `fig_hilos` funcione con 5 modelos.
7. Mirar todas las PNG generadas antes de darlas por buenas: con latencias más altas, las etiquetas de la gráfica de H1 pueden encimarse con la línea del límite (ya hay una corrección que las mete dentro de la barra; verificar que funcione con 5 modelos).

### 5.3 `pruebas_modelos/reportes/generar_pdf_reporte.py`

- La portada genera una tarjeta por modelo. Revisar que **5 tarjetas** quepan en una fila; si no, hacer 2 filas o reducir el tamaño.
- `--recomendado` recibe la clave del modelo que lleva el sello. Decidirlo con los resultados.

---

## 6. Procedimiento en la Raspberry Pi 5

**Acceso:** `ssh rasp` (en `~/.ssh/config`: `HostName 192.168.1.73`, `User renova`). **El 17/10 no respondía** ("No route to host"): verificar que esté encendida y en la red. Si cambió de IP, pedírsela al usuario.

**Carpeta de trabajo de la corrida anterior (13/10):** `~/ecosort_benchmark/`, con `benchmark_visual.py`, `modelos/` y `test_set/`. Ya tenía instalados `ai-edge-litert` 2.2.0, numpy, pillow, psutil y rich. Verificar qué sigue ahí antes de copiar:

```bash
ssh rasp 'ls ~/ecosort_benchmark ~/ecosort_benchmark/modelos; python3 -c "import ai_edge_litert, rich, psutil; print(\"ok\")"'
```

**Pasos:**

1. **Copiar el script y los modelos.** Los modelos van planos en `modelos/`. El `test_set/` ya debería estar; si no, copiar `data/raw/waste_classification/test/`, unos 920 MB.
   ```bash
   rsync -a pruebas_modelos/raspberry/benchmark_visual.py rasp:ecosort_benchmark/
   rsync -a models/MobileNetV3/modelo_residuos_rpi_V3_05.tflite models/EfficientNetB0/modelo_residuos_EfficientNetB0_v1.tflite \
         models/ShuffleNetV2/modelo_residuos_shufflenetV2_imagenet.tflite models/SqueezeNet/modelo_residuos_SqueezeNet_imagenet.tflite \
         models/YOLO/modelo_residuos_YOLO_v1.tflite rasp:ecosort_benchmark/modelos/
   ```
2. **Prueba de humo**, alrededor de 1 min:
   ```bash
   python3 benchmark_visual.py --max-imagenes 40 --duracion-estabilidad 5 --enfriamiento 0 --sin-capturas --salida-dir prueba_humo
   ```
   Revisar que los 5 modelos tengan una accuracy razonable. Después borrar `prueba_humo/`.
3. **Corridas oficiales,** en segundo plano y encadenadas para que sobrevivan a un corte de SSH. Lanzar con `setsid nohup ./correr_todo.sh > /dev/null 2>&1 < /dev/null &`. El script escribe un archivo de estado al terminar.
   ```bash
   #!/bin/bash
   cd ~/ecosort_benchmark
   export COLUMNS=140
   python3 benchmark_visual.py --salida-dir resultados_5modelos_1hilo > log_5modelos_1hilo.txt 2>&1
   echo "exit 1hilo: $?" > estado.txt
   python3 benchmark_visual.py --num-threads 4 --duracion-estabilidad 120 --salida-dir resultados_5modelos_4hilos > log_5modelos_4hilos.txt 2>&1
   echo "exit 4hilos: $?" >> estado.txt
   echo TERMINADO >> estado.txt
   ```
   - **Corrida de 1 hilo:** parámetros por defecto (todas las imágenes, 10 de calentamiento, 300 s de estabilidad, 60 s de enfriamiento). Estimado: **~40–45 min** (5 × 5 min de estabilidad + 5 min de enfriamiento + latencia + 2 juegos de imágenes).
   - **Corrida de 4 hilos:** estimado **~20–25 min**.
   - Si en la Pi 5 se instaló `ai-edge-litert` en un venv, usar `venv/bin/python` en lugar de `python3`.
4. **Esperar** con un bucle en segundo plano que consulte `estado.txt` por SSH cada 60 s, en lugar de polls manuales.
5. **Demostración en el escritorio** para las capturas reales. La Pi 5 tiene escritorio Wayland: usar `grim`, `lxterminal` y `XDG_RUNTIME_DIR=/run/user/$(id -u) WAYLAND_DISPLAY=wayland-0`.
   - Hacerla **después** de las corridas oficiales, porque altera la latencia.
   - Parámetros: `--max-imagenes 1500 --duracion-estabilidad 30 --enfriamiento 5`.
   - Tomar una captura cada 4 s **mientras exista el proceso de Python**, buscándolo con `pgrep -f "^python3 benchmark_visual.py"`. **No** buscar por el texto del comando de `lxterminal`, porque ese proceso sigue vivo y el bucle no termina.
   - Elegir 5 capturas: inicio, latencia de MobileNetV3, estabilidad, latencia de YOLO o de otro modelo, y la comparación final.
6. **Traer los resultados a la laptop:**
   - `reports/benchmark_rpi5_5modelos/`: JSON de 1 y 4 hilos, logs y `capturas_svg/`.
   - `reports/figures/raspberry5_5modelos/`: gráficas, `capturas/` y `escritorio/`.
7. **Generar las gráficas** con `graficar_benchmark_rpi.py --resultados ... --resultados-hilos ... --capturas ... --seleccion ... --salida ...`. Después, revisar visualmente las PNG.
8. **Al final, limpiar la Pi** solo si el usuario lo pide, y dar el comando en lugar de ejecutarlo. Borrar únicamente lo que se creó en esta sesión.

---

## 7. Reporte nuevo

- **Archivo:** `docs/reporte_benchmark_5_modelos_rpi5.md` y su `.pdf`.
- **Solo resultados de la Raspberry Pi 5.** Sin secciones ni columnas de la laptop.
- **Estructura sugerida,** basada en `docs/reporte_benchmark_modelos.md` pero sin la sección "Laptop frente a Raspberry":
  1. Resumen: tabla de los 5 modelos, recomendación y H1.
  2. Entorno de pruebas: solo la Pi 5, con el preprocesamiento de cada modelo.
  3. Precisión:
     - accuracy con IC 95% de Wilson;
     - McNemar exacto entre pares: con 5 modelos son 10 comparaciones, así que aplicar corrección de **Holm**;
     - accuracy por clase, matrices de confusión y coincidencia de errores;
     - casos difíciles y ejemplos.
  4. Latencia: tabla con media, desviación, mínimo, P50, P95, P99 y máximo, cuadros/s y margen frente a H1 (200 ms). Incluir el costo del preprocesamiento de cada juego de imágenes y el peor caso más el preprocesamiento.
  5. Estabilidad: 5 minutos por modelo.
  6. Consumo de recursos.
  7. Compromiso entre precisión y velocidad.
  8. Prueba con 4 hilos.
  9. Conclusiones y recomendación.
  10. Evidencias.
  11. Cómo reproducir.
- **Notas metodológicas obligatorias:**
  - cada modelo recibe el preprocesamiento con el que se entrenó (YOLO con recorte central y /255; los demás estirados y 0–255);
  - el *data augmentation* de YOLO fue distinto;
  - todos los modelos usaron `test/` como validación durante el entrenamiento, sin selección del mejor modelo;
  - los hilos de inferencia son el valor por defecto del runtime (1 hilo, deducido del 25% de CPU).
- **Referencia al reporte anterior:** se puede mencionar que los 3 modelos originales ya se midieron el 13/10 y que esta corrida los repite junto con los nuevos. Comparar con esa corrida solo si aporta algo (repetibilidad): ambas son de la misma Pi, no de la laptop.

---

## 8. Lecciones de las sesiones anteriores (evitar repetir errores)

- **No usar `pkill -f <texto>` por SSH** si el texto aparece en el propio comando: mata la sesión SSH (exit 255). Matar por PID o usar un patrón anclado (`^python3 ...`).
- **Textos fijos en el script:** `benchmark_visual.py` tenía "Raspberry Pi 5" fijo. Ya se corrigió con `DISPOSITIVO = nombre_dispositivo()`. Revisar que no queden textos parecidos, como "tres modelos".
- **Log legible:** al correr con `nohup`, `rich` detecta que no hay terminal y solo imprime el estado final de cada pantalla, lo cual está bien para el log. Exportar `COLUMNS=140` para que el log tenga ancho legible.
- **Preprocesamiento:** en la Pi 5 costó 8.2 ms por imagen con un solo juego. Con dos juegos se duplica el tiempo de carga.
- **Diferencias entre procesadores:** el mismo `.tflite` da predicciones ligeramente distintas según el CPU, por los kernels de XNNPACK y las instrucciones `SDOT`/FP16 del Cortex-A76. Por eso las cifras oficiales son las medidas en la Pi.
- **Commits:** los datos y figuras que no se suben a git se pueden perder. Proponer al usuario un commit de `reports/benchmark_rpi5_5modelos/` y `reports/figures/raspberry5_5modelos/` al terminar. Hacerlo solo si lo pide.

---

## 9. Lista de pasos

- [ ] Verificar acceso a `ssh rasp` y el estado de `~/ecosort_benchmark/`.
- [ ] Adaptar `pruebas_modelos/raspberry/benchmark_visual.py` (sección 5.1) y probarlo en la laptop con pocas imágenes.
- [ ] Adaptar `pruebas_modelos/reportes/graficar_benchmark_rpi.py` (sección 5.2).
- [ ] Copiar el script y los 5 modelos a la Pi 5; hacer la prueba de humo.
- [ ] Correr la corrida oficial de 1 hilo y la de 4 hilos.
- [ ] Hacer la demostración en el escritorio con capturas `grim`.
- [ ] Traer los resultados; generar las gráficas y revisarlas.
- [ ] Escribir `docs/reporte_benchmark_5_modelos_rpi5.md` (solo Raspberry) y generar el PDF.
- [ ] Ofrecer al usuario el commit y el comando de limpieza de la Pi.
