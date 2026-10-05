# Práctica de Procesamiento de Imagen y Visión Artificial con OpenCV

Este repositorio contiene las soluciones a varias tareas de procesamiento de imágenes y vídeo en tiempo real utilizando Python, `OpenCV` y `matplotlib`. El objetivo principal es la comprensión y aplicación de filtros de detección de bordes y técnicas de sustracción de fondo.

## Requisitos y Dependencias

El entorno de ejecución requiere las siguientes librerías:
- `numpy`
- `opencv-python` (`cv2`)
- `matplotlib`
- `Pillow`

## Contenido y Tareas Resueltas

### Tarea 1: Detección de Bordes con Canny y Análisis por Filas
**Objetivo:** Obtener los contornos de la imagen `mandril.jpg` mediante el detector multicapa de Canny y realizar un análisis cuantitativo de la densidad de bordes por filas.

**Implementación:**
1. Se pasa la imagen original a escala de grises.
2. Se aplica `cv2.Canny()` para obtener el mapa de bordes binario.
3. Mediante `cv2.reduce()`, se cuentan los píxeles blancos (valor 255) proyectados sobre el eje vertical (filas).
4. Se calcula el valor máximo de píxeles blancos en una fila (`maxfil`) y se localizan todas aquellas filas que posean un conteo $\ge 0.90 \times maxfil$.
5. Se dibujan líneas horizontales sobre estas coordenadas directamente en una copia en color del resultado de Canny para su rápida visualización.

### Tarea 2: Análisis mediante Operador Sobel y Comparativa
**Objetivo:** Replicar el análisis de proyecciones (esta vez por filas y columnas simultáneamente) usando el operador de Sobel, superponiendo los resultados en la imagen original y comparando con Canny.

**Implementación:**
1. Se suaviza la imagen en grises con un filtro Gaussiano (`cv2.GaussianBlur`) para atenuar ruido de alta frecuencia.
2. Se calculan los gradientes horizontales y verticales (`cv2.Sobel`) y se combinan usando el valor absoluto.
3. Se convierte la imagen a 8 bits (`cv2.convertScaleAbs`) y se le aplica un umbral binario (`cv2.threshold`) para extraer los bordes dominantes.
4. Se reducen filas y columnas para encontrar los máximos y las coordenadas que superan el 90% de sus respectivos picos.
5. Se dibujan estas proyecciones sobre la imagen en color original.

**Conclusiones (Canny vs Sobel):**
Mientras que Sobel (tras un umbralizado) produce contornos anchos y muy sensibles al nivel de textura/ruido de la imagen, Canny logra perfilar contornos de un píxel de grosor mucho más limpios gracias a su algoritmo de supresión de no-máximos e histéresis. Sobel extrae la intensidad cruda del gradiente; Canny realiza una interpretación más topológica del borde.

### Tarea 3: Demostrador WebCam - "My Little Piece of Privacy" (Virtual)
**Objetivo:** Desarrollar un sistema de visión artificial en tiempo real basado en una instalación artística interactiva. En este caso se ha adaptado la obra *[My little piece of privacy]* de Niklas Roy.

**Implementación:**
En la instalación original, una cámara detecta la posición de una persona y un motor mueve una cortina física a lo largo de un riel para bloquear el campo de visión, "otorgando privacidad" al sujeto en la ventana.

Nuestro sistema lo replica de manera completamente virtual:
1. Captura vídeo en tiempo real y aplica `cv2.createBackgroundSubtractorMOG2` para separar el fondo del objeto en movimiento (primer plano).
2. Se limpia la máscara de movimiento usando algoritmos morfológicos (`cv2.morphologyEx`) de apertura y dilatación.
3. Se detectan los contornos, aislando el de mayor tamaño para ignorar ruidos menores.
4. Se extrae el `Bounding Box` (caja delimitadora) de ese contorno y el software dibuja un polígono negro ("cortina virtual") que abarca toda la altura de la pantalla en esa coordenada X, acompañando continuamente al usuario e impidiendo que la cámara renderice su cuerpo.

## Ejecución
Para visualizar los gráficos estáticos y probar el demostrador de la webcam, simplemente ejecuta:

```bash
python tareas_vision.py