# Práctica de Procesamiento de Imagen y Visión Artificial con OpenCV

## GitHub Autores:
Tomás: https://github.com/Pegasos2005
Helen: https://github.com/heleengb

Este repositorio contiene las soluciones a varias tareas de procesamiento de imágenes y vídeo en tiempo real utilizando Python, `OpenCV` y `matplotlib`. El objetivo principal es la comprensión y aplicación de filtros de detección de bordes y técnicas de sustracción de fondo.

## Requisitos y Dependencias

El entorno de ejecución requiere las siguientes librerías:
- `numpy`
- `opencv-python` (`cv2`)
- `matplotlib`
- `Pillow`
- `pygame`

## Contenido y Tareas Resueltas

### Tarea 1: Detección de Bordes con Canny y Análisis por Filas
**Objetivo:** Obtener los contornos de la imagen `mandril.jpg` mediante el detector multicapa de Canny y realizar un análisis de la densidad de bordes por filas.

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

### Tarea 3: Demostrador WebCam - "Censura Selectiva con Feedback Acústico"
**Objetivo:** Desarrollar un sistema de visión en tiempo real que localiza objetos un color amarillo, pone una cortina encima y reproduce un sonido.

**Implementación:**
1. Se utiliza el espacio de color `HSV` para independizar el matiz de la luminosidad y generar una máscara binaria estable.
2. Mediante transformaciones morfológicas (`cv2.morphologyEx`) se eliminan falsos positivos y ruido ambiental.
3. Al detectar un área superior al umbral establecido, el algoritmo dibuja una cortina de censura guiada por el `Bounding Box` del contorno dominante.
4. **Integración Asíncrona:** Utilizando la librería `pygame`, el sistema dispara un evento de audio local. Para mantener los FPS estables y evitar el solapamiento del búfer de audio, se implementa un control de estados (`pygame.mixer.get_busy()`) que verifica la disponibilidad del canal antes de lanzar el hilo de reproducción.
