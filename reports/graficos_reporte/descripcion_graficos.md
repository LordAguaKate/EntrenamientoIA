# Descripción de los gráficos del reporte

Gráficos e imágenes de evidencia del reporte `docs/reporte_benchmark_modelos.pdf`. **Todos provienen de las pruebas hechas en la Raspberry Pi 5** (13/10/2026) con los modelos `.tflite` cuantizados de MobileNetV3-Large, EfficientNetB0 y ShuffleNetV2 (ImageNet), sobre las 4,534 imágenes del conjunto de prueba.

En todas las gráficas cada modelo conserva su color: **azul** = MobileNetV3-Large, **naranja** = EfficientNetB0, **verde** = ShuffleNetV2 (ImageNet).

---

## Figuras del PDF

El número de cada archivo coincide con el número de figura en el PDF.

### Figura 1 — `Figura_01_accuracy_modelos.png`
**Precisión de clasificación de cada modelo.** El punto es la accuracy medida y la barra horizontal, su intervalo de confianza del 95% (método de Wilson). A la derecha se indica el número de errores. Los intervalos de MobileNetV3-Large (96.47%) y EfficientNetB0 (96.67%) se traslapan, así que su precisión es equivalente. ShuffleNetV2 (94.18%) queda claramente por debajo. El eje no inicia en cero, para que se distingan las diferencias.

### Figura 2 — `Figura_02_accuracy_por_clase.png`
**Accuracy de cada modelo en cada una de las 5 clases.** Cada clase tiene tres puntos, uno por modelo, con su porcentaje, y junto al nombre de la clase se indica cuántas imágenes tiene (n). Muestra que `vidrio` es la clase más difícil para los tres modelos y que ShuffleNetV2 es el más bajo en casi todas.

### Figura 3 — `Figura_03_matrices_confusion.png`
**Matrices de confusión de los tres modelos, lado a lado.** Las filas son la clase real y las columnas la clase predicha. Cada celda tiene el número de imágenes; la diagonal son los aciertos. El tono de azul indica la proporción dentro de la clase real, con una escala realzada para que se noten los errores pequeños. Se aprecia que la confusión más frecuente es `vidrio` predicho como `plastico`.

### Figura 4 — `Figura_04_coincidencia_errores.png`
**¿Los modelos se equivocan en las mismas imágenes?** Una barra por modelo con su total de errores, dividida en tres tramos: errores que solo comete ese modelo (tono sólido), errores compartidos con otro modelo (tono medio) y errores que cometen los tres (tono claro). Hay 69 imágenes que fallan los tres modelos, lo que apunta a imágenes ambiguas del conjunto de datos más que a fallas de un modelo en particular.

### Figura 5 — `Figura_05_errores_comunes.png`
**Casos difíciles: 10 imágenes que los tres modelos clasificaron mal** (elegidas a mano entre las 69). Debajo de cada imagen aparece su clase real y la predicción de cada modelo con su confianza; en rojo, porque todas son errores. Son sobre todo envolturas metalizadas, bolsas arrugadas y objetos pequeños sobre fondos con textura.

### Figura 6 — `Figura_06_ejemplos_predicciones.png`
**Ejemplos de predicciones hechas en la Raspberry Pi** (10 imágenes elegidas a mano). Debajo de cada imagen aparece su clase real y la predicción de cada modelo con su confianza: ✓ cuando acierta y ✗ en rojo cuando falla. En el empaque de jugo (cartón), ShuffleNetV2 se equivoca y predice aluminio.

### Figura 7 — `Figura_07_latencia_media.png`
**Latencia media por imagen y cuadros por segundo de cada modelo** (1 hilo, media de 4,524 inferencias). ShuffleNetV2: 9.35 ms (107 cuadros/s); MobileNetV3-Large: 15.50 ms (65 cuadros/s); EfficientNetB0: 45.40 ms (22 cuadros/s). EfficientNetB0 es 2.9 veces más lento que MobileNetV3.

### Figura 8 — `Figura_08_latencia_vs_limite_H1.png`
**Peor latencia observada de cada modelo frente al límite de la hipótesis H1 (200 ms)**, marcado con una línea roja vertical. Junto a cada barra se indica cuántas veces cabe esa latencia dentro del límite (margen ×11.2, ×4.4 y ×20.8). Los tres modelos cumplen H1 con amplio margen.

### Figura 9 — `Figura_09_distribucion_latencia.png`
**Histograma de la latencia de cada inferencia**, un panel por modelo, cada uno con su propia escala. En el título de cada panel van la media, la desviación estándar y los percentiles P50, P95, P99 y el máximo. La línea vertical marca la media. Muestra que la latencia es muy estable: la desviación estándar no pasa de 0.06 ms.

### Figura 10 — `Figura_10_latencia_por_imagen.png`
**Latencia de cada una de las 4,534 inferencias, en el orden en que se ejecutaron.** Una línea por modelo; la leyenda incluye la media de cada uno. Sirve para ver que la latencia se mantiene plana durante toda la prueba, sin tendencias ni degradación.

### Figura 11 — `Figura_11_estabilidad_series.png`
**Prueba de estabilidad: 5 minutos de inferencia continua por modelo**, con una muestra por segundo. Tiene cuatro paneles con el tiempo en el eje horizontal:
1. **Latencia media por segundo:** se mantiene plana.
2. **Uso total de CPU:** ~25%, es decir, un núcleo de cuatro.
3. **Temperatura del SoC:** máximo 57.9 °C, lejos de los 80 °C; las diferencias entre modelos se deben al ventilador.
4. **Variación de RAM respecto al inicio:** se mantiene en 0, sin fugas de memoria.

### Figura 12 — `Figura_12_recursos.png`
**Consumo de recursos en seis paneles de barras:**
- **Tamaño del archivo `.tflite`:** 3.24, 4.50 y 1.50 MB.
- **RAM que ocupa el modelo cargado:** 17.1, 34.2 y 12.2 MB.
- **Uso medio de CPU:** ~25% en los tres.
- **Inferencias por segundo sostenidas.**
- **Temperatura máxima.**
- **Tiempo de carga del modelo.**

### Figura 13 — `Figura_13_precision_vs_latencia.png`
**Compromiso entre precisión y velocidad.** Gráfica de dispersión con la latencia media en el eje horizontal (menos es mejor) y la accuracy en el vertical (más es mejor); el tamaño del círculo es proporcional al tamaño del `.tflite`. La mejor zona es arriba a la izquierda. MobileNetV3-Large tiene casi la misma precisión que EfficientNetB0 con un tercio de su latencia.

### Figura 14 — `Figura_14_latencia_1_vs_4_hilos.png`
**Efecto de usar 4 hilos de inferencia en lugar de 1.** Por modelo, la barra sólida es la latencia con 1 hilo y la clara, con 4 hilos, junto con el factor de aceleración. Con cuatro veces más núcleos la latencia solo mejora entre 1.34 y 1.57 veces, a cambio de ocupar casi el 100% de la CPU. Por eso se recomienda 1 hilo.

### Figura 15 — `Figura_15_escritorio_2_latencia_mobilenet.png`
**Captura de pantalla real del escritorio de la Raspberry Pi (1920×1080)** mientras el script `benchmark_visual.py` clasifica imágenes con MobileNetV3-Large en una terminal. Se ven la tabla en vivo (imagen, clase real, clase predicha, confianza y latencia de cada inferencia) y el panel de estadísticas acumuladas. Es de una ejecución de demostración, así que su latencia es un poco mayor que la oficial.

---

## Capturas adicionales (`capturas_adicionales/`)

El reporte menciona estas imágenes, pero no las inserta como figuras.

### `capturas/` — pantallas del benchmark tal como se vieron en la terminal de la Raspberry

El script las guardó automáticamente durante la corrida oficial.

| Archivo | Qué muestra |
|---|---|
| `captura_0a_sistema_y_modelos.png` | **Etapa 1:** ficha del dispositivo (placa, sistema, kernel, Python, RAM, núcleos, runtime, temperatura inicial, preprocesamiento) y tabla de los tres modelos (archivo, tamaño, tensores de entrada/salida y RAM que ocupan) |
| `captura_0b_imagenes_de_prueba.png` | **Etapa 2:** número de imágenes de prueba por clase, con barras |
| `captura_1a_latencia_mobilenet.png` | Prueba de latencia y precisión de **MobileNetV3-Large** al terminar: últimas imágenes clasificadas y panel de estadísticas (media, desviación, mínimo, máximo, P95, cuadros/s y precisión acumulada) |
| `captura_1b_estabilidad_mobilenet.png` | Monitor de estabilidad de **MobileNetV3-Large** al terminar los 5 minutos: inferencias totales, CPU total y por núcleo, RAM, temperatura, frecuencia, estado de energía y latencia por segundo |
| `captura_1c_resumen_mobilenet.png` | Tabla de resumen de **MobileNetV3-Large** (métricas contra sus criterios, con ✔ CUMPLE) y su matriz de confusión |
| `captura_2a_latencia_efficientnet.png` | Igual que 1a, para **EfficientNetB0** |
| `captura_2b_estabilidad_efficientnet.png` | Igual que 1b, para **EfficientNetB0** |
| `captura_2c_resumen_efficientnet.png` | Igual que 1c, para **EfficientNetB0** |
| `captura_3a_latencia_shufflenet.png` | Igual que 1a, para **ShuffleNetV2 (ImageNet)** |
| `captura_3b_estabilidad_shufflenet.png` | Igual que 1b, para **ShuffleNetV2 (ImageNet)** |
| `captura_3c_resumen_shufflenet.png` | Igual que 1c, para **ShuffleNetV2 (ImageNet)** |
| `captura_9a_comparacion_final.png` | **Etapa final:** tablas comparativas de precisión (accuracy, errores, precisión/recall/F1 macro) y de rendimiento (latencia, P95, cuadros/s, CPU, RAM, temperatura, tamaño, cumplimiento de H1), con barras de latencia y accuracy. Dos encabezados salen recortados ("Temp. m…", "H1 (< 200 m…") |
| `captura_9b_accuracy_por_clase.png` | Tabla de accuracy por clase de los tres modelos; el mejor de cada clase va resaltado en verde |

### `escritorio/` — capturas de pantalla reales del escritorio de la Raspberry Pi

Son de la ejecución de demostración, con 1,500 imágenes y 30 s de estabilidad.

| Archivo | Qué muestra |
|---|---|
| `escritorio_1_inicio_sistema_y_modelos.png` | Inicio del benchmark: encabezado, ficha del entorno, tabla de modelos y barra de carga de imágenes |
| `escritorio_3_estabilidad_mobilenet.png` | Fin de la prueba de latencia de MobileNetV3-Large e inicio de su monitor de estabilidad (CPU por núcleo, RAM, temperatura, frecuencia y estado de energía) |
| `escritorio_4_latencia_shufflenet.png` | ShuffleNetV2 (ImageNet) clasificando imágenes en vivo, con aciertos (✔) y errores (✘) |
| `escritorio_5_comparacion_final.png` | Tablas comparativas finales y resumen (mayor precisión, menor latencia, modelo más ligero) al terminar la demostración |

> **Nota:** las capturas del escritorio y algunas pantallas de la terminal muestran nombres de archivo de imágenes del conjunto de datos, y algunos incluyen nombres de personas. Conviene revisarlas antes de publicarlas.
