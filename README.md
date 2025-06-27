# VisionArtificial_ManoArduino

Contacto: labtec@umce.cl

Este código de Python se utiliza para detectar los dedos de una mano en tiempo real a través de una cámara web. Los gestos de la mano son utilizados 
para controlar una placa de desarrollo Arduino. 

El tutorial completo lo pueden descargar en *.pdf desde este repositorio (Tutorial mano Python.pdf)

Para operar con este código (Dedos_Leds.py), es necesario importar las siguientes bibliotecas en Python:
1) cv2: Para capturar y procesar imágenes de la cámara.
2) mediapipe: Para la detección de manos y puntos clave.
3) numpy: Para operaciones matemáticas y de matrices.
4) math: Para cálculos matemáticos.
5) pyfirmata: Para interactuar con una placa Arduino.

A continuación se da una descripcion general del código: 
1.	La Función palm_centroid calcula el centroide de la palma de la mano a partir de una lista de coordenadas.

2.	Se inicializan los módulos de dibujo y detección de manos de MediaPipe y OpenCV. Luego se configura la captura de video desde la cámara web y se conecta a una placa Arduino en el puerto COM respectivo.

3.	Se define los puntos clave para el pulgar, la palma y las puntas y bases de los restantes dedos.

4.	En el Bucle principal se captura fotogramas de la cámara en un bucle y se procesa cada fotograma para detectar manos y sus puntos clave. Luego se calcula el ángulo del pulgar y se determina si está extendido. Posteriormente se calcula las distancias entre el centroide de la palma y las puntas y bases de los dedos para determinar si los restantes dedos están extendidos. Finalmente se identifica los dedos extendidos y se controla los respectivos pines digitales de la placa Arduino.

5.	Visualización y control de Arduino:
  •	Dibuja los puntos clave y las conexiones de la mano en el fotograma.
  •	Controla los pines digitales de la placa Arduino para encender o apagar LEDs según los dedos detectados.



