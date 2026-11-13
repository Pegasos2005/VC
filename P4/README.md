# Práctica 4: Detección y seguimiento de vehículos, personas y matrículas

Prototipo que procesa el vídeo de test de la práctica y hace lo siguiente:

- detecta y sigue a personas y vehículos;
- localiza las matrículas con un detector YOLO entrenado por nosotros;
- cuenta los objetos de cada clase;
- guarda un vídeo con los resultados, un vídeo anonimizado y un CSV con todas las detecciones.

Todo el código está en el cuaderno [`VC_P4.ipynb`](VC_P4.ipynb). El reconocimiento de caracteres (segunda parte) está en [`../P4B`](../P4B/README.md).

![Fragmento del vídeo resultado](imagenes/resultado.gif)

## Resumen de la entrega

| Requisito del enunciado | Dónde está |
|---|---|
| Detectar y seguir personas y vehículos | [Sección 3](#3-prototipo-sobre-el-vídeo-de-test), `resultados/resultado.mp4` |
| Detectar las matrículas (detector entrenado) | [Secciones 1 y 2](#2-entrenamiento-del-detector-de-matrículas): dataset, entrenamiento y métricas |
| Contar el total de cada clase | [Resultados](#resultados) |
| Vídeo con los resultados | `resultados/resultado.mp4` (con OCR: `../P4B/resultados/resultado_ocr.mp4`) |
| CSV con detección y seguimiento | `resultados/detecciones.csv` (con el texto de la matrícula: `../P4B/resultados/detecciones.csv`) |
| Acceso al dataset y análisis del entrenamiento | [Sección 1](#1-dataset-de-matrículas) y [sección 2](#2-entrenamiento-del-detector-de-matrículas) |
| **P4b:** texto de la matrícula en el CSV y en el vídeo, comparativa OCR clásico vs VLM | [README de P4B](../P4B/README.md) |
| *Extra:* flujo por dirección | [4.1](#41-flujo-por-dirección-resultadosflujocsv) |
| *Extra:* anonimización | [4.2](#42-anonimización-resultadosanonimizadomp4) |
| *Extra:* detección de matrículas por contornos | [4.3](#43-detección-de-matrículas-mediante-contornos) |
| *Extra:* detección de matrículas con YOLO y con RT-DETR | [4.4](#44-detección-de-matrículas-yolo11n-frente-a-rt-detr-l) |

## Contenido de la carpeta

| Ruta | Descripción |
|---|---|
| `VC_P4.ipynb` | Cuaderno con todo el proceso: dataset, entrenamiento, prototipo y extras |
| `dataset_matriculas/` | Dataset de matrículas en formato YOLO (train / val / test) y `matriculas.yaml` |
| `VC_P4_RTDETR.ipynb` | Extra: comparación YOLO11n vs RT-DETR-L (evaluación en local) |
| `VC_P4_RTDETR_colab.ipynb` | Extra: entrenamiento de RT-DETR-L en Google Colab con GPU |
| `VC_P4_contornos.ipynb` | Extra: detección de matrículas mediante contornos y comparación con YOLO |
| `runs/detect/matriculas_yolo11n/` | Resultados del entrenamiento: pesos `best.pt`, curvas y métricas |
| `resultados/resultado.mp4` | Vídeo de test con detecciones, IDs de seguimiento, matrículas y conteo |
| `resultados/anonimizado.mp4` | Vídeo de test con personas y matrículas difuminadas (extra) |
| `resultados/detecciones.csv` | Detección y seguimiento por fotograma |
| `resultados/flujo.csv` | Dirección y borde de salida de cada objeto seguido (extra) |

> Los vídeos de `resultados/` se han reescalado de 1920×1080 a 1280×720 para no superar el límite de 100 MB por archivo de GitHub.
> Después se recodificaron a H.264 para que se puedan reproducir en el navegador: `ffmpeg -i in.mp4 -c:v libopenh264 -b:v 3M -pix_fmt yuv420p out.mp4`.
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

**Acceso al dataset:**

- La versión ya preparada en formato YOLO (imágenes a 1280 px, etiquetas y reparto) está en [`dataset_matriculas/`](dataset_matriculas/), en este repositorio.
- El dataset original completo (imágenes a resolución original, ~7 GB, y anotaciones `.json`) está en
  [Google Drive](https://drive.google.com/drive/folders/1RFd6sKH_QMQcpE_w5rWmy8rKiYUY5_SI).
  Para regenerar `dataset_matriculas/`, se descarga en `P4/dataset_raw/annotations/` (los `.json` en esa carpeta y las imágenes en `images/` dentro de ella)
  y se ejecuta la primera celda de código del cuaderno. El reparto es el mismo, porque usa semilla fija.
Además de la caja, cada `.json` incluye el texto de la matrícula (`plate`), que se usa en la [P4b](../P4B/README.md) para evaluar el OCR.

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
- **OCR:** en esta carpeta la columna `texto_matricula` queda vacía. El reconocimiento de texto se hace en la [P4b](../P4B/README.md), que genera el CSV y el vídeo con las matrículas leídas.

Vídeo completo: [`resultados/resultado.mp4`](resultados/resultado.mp4). Al principio de este README hay un fragmento de 5 s.

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

![Fragmento del vídeo anonimizado](imagenes/anonimizado.gif)

*Fragmento recortado: las personas de la acera y la matrícula del coche aparecen difuminadas. Vídeo completo: [`resultados/anonimizado.mp4`](resultados/anonimizado.mp4).*

![Fotograma anonimizado](imagenes/anonimizado_f1400.jpg)

### 4.3 Detección de matrículas mediante contornos

Cuaderno: [`VC_P4_contornos.ipynb`](VC_P4_contornos.ipynb). Es una alternativa clásica, sin entrenamiento, al detector YOLO de matrículas.
Aprovecha que una matrícula es un rectángulo claro de proporción ~4,7:1 con caracteres oscuros que generan muchos bordes verticales juntos:

1. *Black-hat* (13×5) para realzar los caracteres oscuros sobre fondo claro.
2. Sobel en *x* para obtener los bordes verticales, y cierre morfológico horizontal (17×3) para unir los caracteres en un bloque.
3. Umbral de Otsu, limpieza morfológica y una máscara de zonas claras (el fondo de la matrícula).
4. `findContours` y filtrado del rectángulo envolvente: proporción entre 2 y 7, tamaño mínimo y máximo, y relleno ≥ 0,45.
5. Puntuación de cada candidato: densidad de bordes × relleno × cercanía de la proporción a 4,5. Se procesa a 3 escalas y se fusiona con NMS.

![Pasos del método de contornos](imagenes/contornos_pasos.jpg)

Se evalúa en dos escenarios:

- **imagen completa**;
- **dentro de vehículos**, como en el prototipo: se detectan los vehículos con `yolo11n`, cada uno se reescala a 480 px de ancho
  y la matrícula se busca solo en su 65 % inferior.

Los parámetros se ajustaron en *val*. Los resultados son sobre las **mismas 477 imágenes de test** y con el **mismo evaluador** para ambos métodos
(IoU ≥ 0,5, AP calculado a partir de la curva precisión-recall). El evaluador da para YOLO en imagen completa un AP50 de 0,618,
prácticamente igual que el mAP@0.5 de Ultralytics (0,619), lo que confirma que la comparación es coherente.

| Método | Escenario | Matrículas | AP50 | P | R | F1 | Recall máx. | ms/imagen |
|---|---|---|---|---|---|---|---|---|
| Contornos | imagen completa | 1045 | 0,135 | 0,385 | 0,201 | 0,264 | 0,268 | **9** |
| YOLO | imagen completa | 1045 | **0,618** | 0,878 | 0,614 | **0,723** | 0,661 | 22 |
| Contornos | dentro de vehículos | 942 | 0,259 | 0,503 | 0,357 | 0,417 | 0,412 | **7** |
| YOLO | dentro de vehículos | 942 | 0,590 | 0,717 | 0,658 | 0,686 | **0,734** | 26 |

*P, R y F1 en el umbral de F1 máximo. Los tiempos son en CPU (Ryzen AI 7 445) y no incluyen la detección de vehículos (25 ms/imagen).*

![Contornos frente a YOLO](imagenes/contornos_vs_yolo.png)

![Ejemplos de contornos y YOLO](imagenes/contornos_ejemplos.jpg)

**Conclusiones**

- **YOLO es muy superior**: en imagen completa su AP50 es 4,6 veces el de los contornos, y su F1 es 0,72 frente a 0,26.
  Los contornos solo compiten en tiempo: son unas 2–3 veces más rápidos y no necesitan entrenamiento ni datos anotados.
- **Buscar dentro del vehículo casi duplica el rendimiento de los contornos** (AP50 de 0,14 a 0,26). Se eliminan muchos falsos positivos
  (rejillas, carteles, señales, marcas viales) y la matrícula se reescala a un tamaño más uniforme.
  A YOLO no le aporta mucho porque ya distingue bien una matrícula de lo que no lo es.
- **El tamaño vuelve a ser decisivo**: el 32 % de las matrículas de test mide menos de 20 px. Ahí los contornos no encuentran ninguna,
  porque los caracteres no llegan a formar bordes. A partir de 40 px, YOLO supera el 94 % de recall y los contornos se quedan entre el 41 y el 60 %.
- Los fallos típicos de los contornos son matrículas inclinadas o en perspectiva (la proporción sale del rango), con poco contraste,
  o pegadas a otros elementos con muchos bordes (parachoques, rejillas), que el cierre morfológico une con la matrícula.

### 4.4 Detección de matrículas: YOLO11n frente a RT-DETR-L

Cuadernos: [`VC_P4_RTDETR_colab.ipynb`](VC_P4_RTDETR_colab.ipynb) (entrenamiento) y [`VC_P4_RTDETR.ipynb`](VC_P4_RTDETR.ipynb) (análisis y evaluación).

RT-DETR es un detector basado en *transformers*, de Baidu, sin NMS. Se ha usado su implementación en Ultralytics (`RTDETR('rtdetr-l.pt')`, preentrenado en COCO),
así que el dataset, el `.yaml`, las métricas y la evaluación son exactamente los mismos que con YOLO.

**Entrenamiento.** En CPU habría tardado días, así que se entrenó en **Google Colab con una GPU Tesla T4**. Tardó **77 minutos**.
Los ajustes son los mismos que en YOLO11n: `imgsz=640`, 40 épocas, `patience=10`, mismo reparto train/val/test y optimizador elegido automáticamente (AdamW).
El *batch* es de 8 en lugar de 16, por la memoria de la GPU. Los resultados están en `runs/detect/matriculas_rtdetr_l/`.

![Curvas de entrenamiento de RT-DETR](runs/detect/matriculas_rtdetr_l/results.png)

![Métricas de validación por época: YOLO11n frente a RT-DETR-L](imagenes/rtdetr_curvas.png)

- RT-DETR **aprende mucho más rápido**: en la época 1 ya tiene un mAP@0.5 de 0,58 en validación, que YOLO11n no alcanza hasta la época 13.
  Supera a YOLO en *recall* y mAP durante todo el entrenamiento.
- Es menos estable al principio. En la época 7 hay una caída puntual de todas las métricas, típica de los *transformers*, y se recupera en la siguiente.
- Como en YOLO, la mejor época es la última (mAP@0.5 de validación 0,762 frente a 0,658), así que **todavía no había convergido**.
- Las pérdidas de validación salen `nan` en 26 de las 40 épocas, por eso hay huecos en esas gráficas. Es un problema del cálculo de esa pérdida
  en Ultralytics con precisión mixta (AMP). No afecta a las métricas, que se calculan en todas las épocas.

**Resultados en test** (mismas 477 imágenes y 1045 matrículas; inferencia medida por Ultralytics en CPU, Ryzen AI 7 445):

| Modelo | Parámetros | Pesos | Precisión | Recall | mAP@0.5 | mAP@0.5:0.95 | Inferencia CPU |
|---|---|---|---|---|---|---|---|
| YOLO11n | **2,6 M** | **5,5 MB** | **0,89** | 0,60 | 0,619 | 0,333 | **19 ms/img** |
| RT-DETR-L | 32,8 M | 66 MB | 0,88 | **0,70** | **0,705** | **0,400** | 492 ms/img |

![Ejemplos en test: YOLO11n arriba, RT-DETR-L abajo](imagenes/rtdetr_ejemplos.jpg)

*Ejemplos de test (arriba YOLO11n, abajo RT-DETR-L). RT-DETR encuentra matrículas pequeñas y lejanas que YOLO no detecta, como las del aparcamiento o la de la moto.*

**En el vídeo de test** (1 de cada 10 fotogramas; vehículos detectados con `yolo11n` y la matrícula buscada en cada recorte con ambos modelos, a `imgsz=640`),
RT-DETR encuentra matrícula en el **22,6 %** de los recortes de vehículo y YOLO11n en el **5,4 %**. Tarda 569 ms por recorte, frente a 47 ms.
En los recortes en que no coinciden, RT-DETR detecta correctamente la matrícula del autobús, donde YOLO no ve nada:

![Recortes del vídeo en que los modelos no coinciden](imagenes/rtdetr_video.jpg)

**Conclusiones**

- **RT-DETR-L es más preciso.** El mAP@0.5 sube un 14 % y el mAP@0.5:0.95 un 20 %. Sobre todo **encuentra más matrículas**
  (*recall* de 0,60 a 0,70) sin perder precisión, que es justo el punto débil de YOLO11n: las matrículas pequeñas y lejanas.
- **Es mucho más caro.** Tiene 12,6 veces más parámetros, es unas 26 veces más lento en CPU y no se puede entrenar sin GPU.
  Para el prototipo en un portátil sin GPU, YOLO11n sigue siendo la opción práctica: procesar el vídeo con RT-DETR llevaría horas.
  Con GPU, RT-DETR-L sería la mejor elección.
- **Licencia:** el RT-DETR original de Baidu es Apache 2.0 y permite uso comercial sin publicar el código.
  La implementación de Ultralytics usada aquí sigue siendo AGPL-3.0, así que para un producto habría que usar la implementación original.

## Entorno

Python 3.11 con `ultralytics` 8.4, `opencv-python`, `pandas`, `matplotlib` y `lap` (necesario para el seguimiento). Todo se ejecuta en CPU.
