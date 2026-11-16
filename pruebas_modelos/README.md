# Pruebas de los modelos

Scripts con los que se probaron los modelos entrenados, separados según dónde se ejecutan.
Todos se corren desde la raíz del repositorio.

## `modelos/` — copia de los modelos probados

Los scripts toman los modelos de esta carpeta, no de `models/`. El conjunto de prueba sí se lee de
`data/raw/waste_classification/test`.

| Modelo | Archivos |
|---|---|
| MobileNetV3-Large | `MobileNetV3/modelo_residuos_rpi_V3_05.tflite` y `.keras` |
| EfficientNetB0 | `EfficientNetB0/modelo_residuos_EfficientNetB0_v1.tflite` y `.keras` |
| ShuffleNetV2 (ImageNet) | `ShuffleNetV2/modelo_residuos_shufflenetV2_imagenet.tflite` y `.keras` |
| SqueezeNet v1.1 (ImageNet) | `SqueezeNet/modelo_residuos_SqueezeNet_imagenet.tflite` y `.keras` |
| YOLO11s-cls | `YOLO/modelo_residuos_YOLO_v1.tflite` y `.pt` |
| MobileNetV2 (cámara) | `MobileNetV2/modelo_residuos_rpi.tflite` |

## `raspberry/` — se ejecutan en la Raspberry Pi

| Script | Qué hace |
|---|---|
| `benchmark_visual.py` | Benchmark principal de los 5 modelos `.tflite` (MobileNetV3-Large, EfficientNetB0, ShuffleNetV2, SqueezeNet v1.1, YOLO11s-cls): latencia, precisión, estabilidad (CPU, RAM, temperatura) y comparación final. |
| `Benchmark_rpi.py` | Versión anterior del benchmark: latencia, CPU/RAM y estabilidad de un solo `.tflite`. |
| `test_camara_realtime.py` | Inferencia en tiempo real con la cámara. |

En la Raspberry se copia el script junto a una carpeta `modelos/` y otra `test_set/`; si no existen, el
script usa `models/` y `data/raw/waste_classification/test` del repositorio. También funcionan en la
laptop (caen a `tensorflow.lite`).

## `local/` — se ejecutan en la computadora

| Script | Qué hace |
|---|---|
| `comparar_modelos_inferencia.py` | Carga los `.keras` y evalúa el test set completo (4534 imágenes). |
| `Evaluate_tflite.py` | Evalúa el `.tflite` cuantizado para comparar precisión antes/después de cuantizar. |
| `comparar_modelos.py` | Arma la tabla comparativa con los resultados guardados en los notebooks de evaluación. |

## `reportes/` — procesan los resultados (en la computadora)

| Script | Qué hace |
|---|---|
| `graficar_benchmark_rpi.py` | Genera las gráficas a partir del JSON medido en la Raspberry. |
| `generar_pdf_reporte.py` | Convierte un reporte `.md` de `docs/` a PDF. |
