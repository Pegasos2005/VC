# Práctica de Reconocimiento de caracteres en matrículas

## Autores:

#### Grupo G43
Cristian Tomás Vega Appelqvist [[GitHub](https://github.com/Pegasos2005)]  
Helena González Báez [[GitHub](https://github.com/heleengb)]

Segunda parte de la [P4](../P4/README.md). Comparamos un OCR clásico (**EasyOCR**) con un modelo de visión y lenguaje (**SmolVLM-256M**) leyendo matrículas, y usamos el mejor para leer las matrículas del vídeo de test y meter el texto en el CSV y en el vídeo.

![Fragmento del vídeo con la matrícula leída](./imagenes/resultado_ocr.gif)

## Contenido

| Archivo | Descripción |
|---|---|
| `VC_P4b_OCR.ipynb` | Notebook de la práctica: comparativa de los dos OCR y lectura de las matrículas del vídeo |
| `VC_P4b.ipynb` | Notebook de ejemplo del enunciado |
| `resultados/comparativa_ocr.csv` | Lo que lee cada modelo en cada matrícula de la comparativa, con su tiempo |
| `resultados/detecciones.csv` | El CSV de la P4 con el texto de las matrículas relleno |
| `resultados/lecturas_video.csv` | Lecturas de los dos modelos sobre los recortes del vídeo |
| `resultados/resultado_ocr.mp4` | Vídeo de test con la matrícula leída encima de cada coche |
| `imagenes/` | Gráficas y capturas de este README |

## Requisitos

```
conda create -n VC_P4 python=3.11
conda activate VC_P4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics lap easyocr transformers accelerate num2words pandas matplotlib
```

Necesita la carpeta `../P4` con el dataset, el vídeo `C0142.MP4` y las anotaciones originales (`dataset_raw/`, ver el README de la P4). Ejecutar las celdas en orden. La comparativa tarda una media hora en CPU, así que se guarda en `comparativa_ocr.csv` y si ya existe no se repite.

Todas las pruebas están hechas en el mismo portátil, **sin GPU**:

| Equipo | |
| :--- | :--- |
| CPU | AMD Ryzen AI 7 445 w/ Radeon 840M (12 hilos) |
| RAM | 30 GB |
| GPU | ninguna, todo en CPU |
| Sistema | Fedora 44 (Linux 7.2), Python 3.11, torch 2.15, transformers 5.23 |

---

## Tarea 1: Comparativa EasyOCR vs SmolVLM

### Datos de prueba
Las anotaciones originales del dataset de la P4 traen, además de la caja, el texto de cada matrícula, así que no hemos tenido que etiquetar nada a mano. Usamos solo el split de **test**, para no evaluar con imágenes que el detector ya ha visto.

- Quitamos las matrículas anotadas como `None`, que son las que no se podían leer.
- Quedan **629 matrículas** de 439 imágenes: 594 con el formato actual (`1234ABC`), 32 con el provincial antiguo (`GC1234AB`) y 3 con otros.
- Cada matrícula se recorta de la imagen con un poco de margen (5 % a los lados y 15 % arriba y abajo). Son bastante pequeñas: la mediana mide 71 px de ancho y el 42 % no llega a 60 px.

### Modelos
- **EasyOCR**: OCR clásico (detector CRAFT + reconocedor CRNN). Limitamos los caracteres a `0-9A-Z` con `allowlist` y juntamos los trozos que detecta de izquierda a derecha.
- **SmolVLM-256M-Instruct**: modelo de visión y lenguaje pequeño de HuggingFace (256 M parámetros). Le pasamos el recorte con el prompt *"Read the text on this license plate. Answer only with the characters."*

Al principio SmolVLM tardaba unos 10 s por matrícula. Mirándolo vimos que por defecto parte cada imagen en 9 trozos de 512 px (*image splitting*), cosa que para un recorte tan pequeño no sirve de nada. Desactivándolo (`do_image_splitting=False`) lee igual y baja a unos 2 s.

### Método
Las condiciones son las mismas para los dos:

1. **Mismos recortes**: las 629 matrículas.
2. **Mismo preprocesado** (`preprocesa`): si el recorte mide menos de 64 px de alto se amplía hasta 64 px.
3. **Misma limpieza de la salida** (`normaliza`): mayúsculas, solo letras y números, y si sale un patrón de matrícula española nos quedamos solo con eso (así no penaliza leer la `E` de la banda azul).
4. **Mismo equipo**: CPU, de una en una y con los modelos ya cargados. Antes de medir se hacen 3 lecturas de calentamiento que no cuentan.

Para medir el acierto usamos tres métricas: matrícula exacta, matrícula con como mucho un carácter mal, y precisión por carácter (1 − CER, con la distancia de Levenshtein).

### Resultados

![Comparativa](./imagenes/comparativa_ocr.png)

| Modelo | Matrícula exacta | Con ≤1 fallo | Precisión por carácter | Tiempo medio | Mediana | p95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EasyOCR | 27.2 % | 39.9 % | 52.3 % | 131 ms | 105 ms | 248 ms |
| SmolVLM-256M | 53.6 % | 65.0 % | 62.2 % | 2202 ms | 2020 ms | 3081 ms |

Acierto según lo ancha que es la matrícula en la imagen:

| Ancho (px) | Matrículas | EasyOCR | SmolVLM |
| :--- | :--- | :--- | :--- |
| < 60 | 264 | 1.9 % | 13.6 % |
| 60-100 | 139 | 21.6 % | 71.2 % |
| 100-150 | 95 | 56.8 % | 93.7 % |
| 150-250 | 92 | 56.5 % | 85.9 % |
| > 250 | 39 | 76.9 % | 87.2 % |

Por formato, EasyOCR acierta el 27.8 % de las actuales y el 18.8 % de las antiguas, y SmolVLM el 53.7 % y el 56.2 %. Mirando matrícula a matrícula, aciertan los dos en el 24.8 %, solo SmolVLM en el 28.8 %, solo EasyOCR en el 2.4 % y ninguno en el 44.0 %.

![Ejemplos](./imagenes/ejemplos_ocr.png)

### Errores más comunes
- **EasyOCR**: en las matrículas pequeñas muchas veces no devuelve nada, porque su detector de texto ni siquiera encuentra la zona. Cuando lee, confunde letras parecidas (`G` por `0`, `P` por `F`, `M` por `I`) o se inventa un carácter de más con el borde o la banda azul.
- **SmolVLM**: cuando la matrícula se ve bien suele darla entera y con su formato, fallando uno o dos caracteres (por ejemplo `6118 MF8` en vez de `6118NFB`). Pero cuando no la ve bien **se la inventa**: devuelve cosas como `111111111111111` o `12345`. Por eso hay que comprobar siempre que la salida tenga formato de matrícula.

### Conclusiones
- SmolVLM acierta el doble que EasyOCR (53.6 % frente a 27.2 %) y le gana en todos los tamaños y formatos. EasyOCR casi nunca acierta una que SmolVLM falle (2.4 %), así que combinarlos no aportaría mucho.
- EasyOCR es unas 19 veces más rápido (105 ms frente a 2 s de mediana). En nuestro portátil EasyOCR podría ir casi en tiempo real y SmolVLM no.
- Lo que más influye es el tamaño, más que el modelo. Por debajo de 60 px de ancho ninguno pasa del 14 %, y a partir de 100 px SmolVLM está entre el 86 % y el 94 %.

---

## Tarea 2: Matrículas en el vídeo de test

Partimos de las detecciones y el seguimiento de la P4 (`../P4/resultados/detecciones.csv`), así que los tracks y el conteo son los mismos, y volvemos a recorrer el vídeo.

### El problema
Las matrículas del vídeo son mucho peores que las del dataset. Lejos miden unos 42×12 px de mediana, y cuando el coche pasa cerca y la matrícula se ve grande, sale movida. Probando, EasyOCR no sacaba casi nada ni en las que se leen a simple vista. Leer las ~1900 matrículas detectadas con SmolVLM tampoco tenía sentido: tardaría más de una hora y la mayoría son ilegibles.

### Método
1. **Nitidez** (`nitidez`): se recortan todas las matrículas detectadas y se mide lo nítida que es cada una (varianza del laplaciano).
2. **Selección**: para cada vehículo cogemos los 5 recortes más nítidos que midan al menos 30 px de ancho y los leemos con SmolVLM, que es el que mejor salió en la comparativa. Así son 140 lecturas en vez de ~1900. Los mismos recortes los leemos también con EasyOCR para compararlos.
3. **Lectura por fotograma**: si lo que lee SmolVLM tiene formato de matrícula, se guarda en `texto_matricula` de ese fotograma.
4. **Votación** (`vota`): si un vehículo tiene al menos 2 lecturas válidas, se elige letra a letra la más repetida. Así se corrigen lecturas con algún fallo (`1965K8P` + `1965KBP` + ... da `1965KBP`). El resultado va a una columna nueva, `texto_matricula_track`, y es lo que sale en el vídeo encima del coche. En rojo, junto a la matrícula, sale lo que se leyó en cada fotograma.

Con los mismos 140 recortes, SmolVLM da una matrícula con formato válido en el 20.0 % de los casos y EasyOCR solo en el 2.9 %. En el vídeo la diferencia es todavía mayor que en el dataset.

![Fotograma del vídeo](./imagenes/resultado_ocr_f2100.jpg)

Vídeo completo: [`resultados/resultado_ocr.mp4`](./resultados/resultado_ocr.mp4)

### Resultados
De los 60 vehículos con matrícula detectada, solo 28 tienen algún recorte de al menos 30 px, y 8 acaban con matrícula leída. Como el vídeo no tiene el texto anotado, las hemos revisado a ojo con el recorte más nítido de cada uno:

![Matrículas leídas en el vídeo](./imagenes/matriculas_video.png)

| Track | Leída | Lo que pone | Lecturas válidas (iguales) |
| :--- | :--- | :--- | :--- |
| 647 | 1965KBP | bien | 4 (4) |
| 540 | 1770JYG | bien | 4 (3) |
| 96 | 3713LKC | bien | 3 (3) |
| 469 | 8536LKJ | bien | 3 (3) |
| 338 | 0053JFM | la última letra no se distingue (M o W) | 3 (3) |
| 277 | 2914FGM | falla una letra (es FGN) | 2 (1) |
| 416 | 3902FWD | falla una letra (es FXD) | 2 (1) |
| 611 | 0092JUM | mal, está muy borrosa | 3 (0) |

Cuando las lecturas coinciden entre sí, la matrícula sale bien; cuando no, hay fallos. Así que el número de lecturas que coinciden sirve para saber cuánto fiarse.

### CSV (`resultados/detecciones.csv`)
Es el formato de la P4 con una columna más:

```
fotograma, tipo_objeto, confianza, identificador_tracking, x1, y1, x2, y2,
matricula_en_su_caso, confianza_matricula, mx1, my1, mx2, my2, texto_matricula, texto_matricula_track
```

- `texto_matricula`: lo que lee SmolVLM en ese fotograma (solo en los recortes que se leen y si tiene formato válido).
- `texto_matricula_track`: la matrícula final del vehículo, repetida en todas sus filas.

### Problemas observados
- La mayoría de las matrículas del vídeo no se leen porque son demasiado pequeñas o salen movidas. Muchas no las podemos leer ni nosotros, así que ahí no hay OCR que valga.
- SmolVLM se inventa texto cuando no ve la matrícula. Lo controlamos filtrando por formato y pidiendo al menos 2 lecturas, pero si las lecturas no coinciden (como en el 611) la votación puede mezclar letras de unas y otras.

### Posibles mejoras
- Probar un OCR entrenado específicamente para matrículas, o un VLM más grande si hay GPU.
- Enderezar la matrícula (corregir la perspectiva) antes de leerla.
- Grabar el vídeo con más resolución o con menos tiempo de exposición para que no salgan movidas.
