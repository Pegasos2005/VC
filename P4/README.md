# Práctica 4: Detección y seguimiento de vehículos, personas y matrículas

Prototipo que procesa el vídeo de test de la práctica y hace lo siguiente:

- detecta y sigue a personas y vehículos;
- localiza las matrículas con un detector YOLO entrenado por nosotros;
- cuenta los objetos de cada clase;
- guarda un vídeo con los resultados, un vídeo anonimizado y un CSV con todas las detecciones.

Todo el código está en el cuaderno [`VC_P4.ipynb`](VC_P4.ipynb).

## Contenido de la carpeta

| Ruta | Descripción |
|---|---|
| `VC_P4.ipynb` | Cuaderno con todo el proceso: dataset, entrenamiento, prototipo y extras |
| `dataset_matriculas/` | Dataset de matrículas en formato YOLO (train / val / test) y `matriculas.yaml` |
| `runs/detect/matriculas_yolo11n/` | Resultados del entrenamiento: pesos `best.pt`, curvas y métricas |
| `resultados/resultado.mp4` | Vídeo de test con detecciones, IDs de seguimiento, matrículas y conteo |
| `resultados/anonimizado.mp4` | Vídeo de test con personas y matrículas difuminadas (extra) |
| `resultados/detecciones.csv` | Detección y seguimiento por fotograma |
| `resultados/flujo.csv` | Dirección y borde de salida de cada objeto seguido (extra) |

> Los vídeos de `resultados/` se han reescalado de 1920×1080 a 1280×720 para no superar el límite de 100 MB por archivo de GitHub.
> El vídeo de entrada (`C0142.MP4`, 200 MB) no se incluye en el repositorio; es el vídeo de test proporcionado en el enunciado.

## 1. Dataset de matrículas

- **2387 imágenes** anotadas con **5235 matrículas** y una única clase, `matricula`.
- Las anotaciones originales (un `.json` por imagen con `rectangle = [xc, yc, w, h]` normalizado) se convirtieron al formato YOLO.
  Se descartaron las cajas degeneradas y se recortaron las que salían de la imagen.
- Reparto aleatorio con semilla fija: **64 % train / 16 % val / 20 % test**.

| Split | Imágenes | Matrículas |
|---|---|---|
| train | 1528 | 3362 |
| val | 382 | 828 |
| test | 477 | 1045 |

- Las imágenes se guardaron reescaladas a 1280 px de lado mayor. Las originales ocupaban unos 7 GB, y reducirlas acelera mucho el entrenamiento en CPU.

**Acceso al dataset:** la versión ya preparada está en [`dataset_matriculas/`](dataset_matriculas/) en este repositorio.
<!-- TODO: indicar aquí el origen de las imágenes originales (dataset_raw) -->

## 2. Entrenamiento del detector de matrículas

| Parámetro | Valor |
|---|---|
| Modelo base | `yolo11n.pt` (preentrenado en COCO) |
| Épocas | 40 (`patience=10`) |
| `imgsz` / `batch` | 640 / 16 |
| Optimizador | AdamW, `lr0=0.002` |
| Dispositivo | CPU (AMD Ryzen 7 5700G), unas 2,6 h |

![Curvas de entrenamiento](runs/detect/matriculas_yolo11n/results.png)

**Métricas**

| Conjunto | Precisión | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Validación (mejor época, la 40) | 0.864 | 0.644 | 0.658 | 0.356 |
| **Test** | **0.889** | **0.600** | **0.619** | **0.334** |

**Análisis**

- Las pérdidas de caja, clasificación y DFL bajan de forma continua, tanto en entrenamiento como en validación, y no se observa sobreajuste.
- La mejor época es la última, así que el modelo **todavía no había convergido**. Con más épocas, o con GPU, el rendimiento probablemente mejoraría.
- La precisión es alta (≈0.89), pero el *recall* se queda en ≈0.60: cuando el modelo detecta algo casi siempre es una matrícula, pero se le escapan muchas.
  Suelen ser matrículas muy pequeñas o lejanas, que con un modelo *nano* a 640 px quedan con muy pocos píxeles.
- Los resultados de test son parecidos a los de validación, así que el modelo generaliza a imágenes que no ha visto.

![Curva PR](runs/detect/matriculas_yolo11n/BoxPR_curve.png)

## 3. Prototipo sobre el vídeo de test

- **Personas y vehículos:** se usa `yolo11n.pt` en modo seguimiento (BoT-SORT) con las clases *persona, bicicleta, coche, moto, autobús y camión*.
- **Matrículas:** se recorta cada vehículo con motor seguido, con un margen del 5 %, y se pasa el recorte al detector entrenado.
  Se queda la matrícula de mayor confianza. Así cada matrícula queda asociada a su vehículo y ocupa más píxeles en la entrada del detector.
- **Conteo:** número de IDs de seguimiento distintos por clase. Solo cuentan los tracks que aparecen en al menos 5 fotogramas, para descartar IDs espurios.
- **OCR:** la columna `texto_matricula` queda vacía, porque el reconocimiento de texto corresponde a la segunda parte de la práctica (P4b).

![Fotograma del resultado](imagenes/resultado_f1400.jpg)

### Formato del CSV (`resultados/detecciones.csv`)

```
fotograma, tipo_objeto, confianza, identificador_tracking, x1, y1, x2, y2,
matricula_en_su_caso, confianza_matricula, mx1, my1, mx2, my2, texto_matricula
```

`matricula_en_su_caso` vale `si` o `no` en los vehículos con motor y queda vacío en personas y bicicletas.
Las coordenadas están en píxeles del vídeo original (1920×1080).

### Resultados

- **2832 fotogramas** procesados y **21 013 detecciones**.
- Tracks totales: 274, de los cuales **214 son válidos** (al menos 5 fotogramas).

| Clase | Objetos distintos |
|---|---|
| coche | 180 |
| persona | 24 |
| moto | 6 |
| camión | 2 |
| bicicleta | 2 |

- Matrícula localizada en el **10,8 %** de las detecciones de vehículos (1926 de 17 806).
  Por vehículo, se encuentra la matrícula al menos una vez en **57 de los 188** vehículos válidos.
- La relación de aspecto media de las cajas de matrícula es de ~3,5:1, coherente con una matrícula real. La confianza mediana es de 0,67.

### Limitaciones

- **El conteo de coches está sobreestimado.** Al fondo de la escena hay muchos coches aparcados, pequeños y que se tapan entre sí.
  El tracker pierde su ID con frecuencia y les asigna uno nuevo, así que un mismo coche puede contar varias veces.
  Esto se ve en el análisis de flujo: 153 coches salen como "parados" y la mayoría de los tracks "desaparecen dentro" de la imagen sin salir por un borde.
- La clase de un track es la del último fotograma en que se detectó. Algunos autobuses acaban clasificados como coche o camión, y por eso no aparecen en el conteo final.
- Con el detector de matrículas se pierden sobre todo las matrículas lejanas, por el *recall* de ≈0.60.

## 4. Extras

### 4.1 Flujo por dirección (`resultados/flujo.csv`)

Para cada track válido se compara la posición media del centro en los primeros y en los últimos fotogramas.
El desplazamiento dominante da la dirección: izquierda, derecha, arriba o abajo. Si el desplazamiento es menor que el 5 % del ancho, el objeto se considera *parado*.
También se indica por qué borde sale de la imagen.

| Clase | Derecha | Izquierda | Parado | Total |
|---|---|---|---|---|
| bicicleta | 0 | 2 | 0 | 2 |
| camión | 0 | 1 | 1 | 2 |
| coche | 4 | 23 | 153 | 180 |
| moto | 3 | 3 | 0 | 6 |
| persona | 9 | 3 | 12 | 24 |
| **total** | **16** | **32** | **166** | **214** |

| Borde de salida | Objetos |
|---|---|
| izquierda | 16 (+1 por la esquina superior izquierda) |
| derecha | 7 |
| desaparece dentro de la imagen | 185 |
| sigue en imagen al final | 5 |

### 4.2 Anonimización (`resultados/anonimizado.mp4`)

En la misma pasada se genera un segundo vídeo con desenfoque gaussiano sobre la caja de cada persona y sobre cada matrícula detectada.
El tamaño del kernel se adapta al tamaño de la caja. Los vehículos se anonimizan difuminando su matrícula, que es el dato que identifica al propietario.

![Fotograma anonimizado](imagenes/anonimizado_f1400.jpg)

## Entorno

Python 3.11 con `ultralytics` 8.4, `opencv-python`, `pandas`, `matplotlib` y `lap` (necesario para el seguimiento). Todo se ejecuta en CPU.
