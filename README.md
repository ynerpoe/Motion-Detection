Este código realiza detección y seguimiento de objetos en movimiento en un video utilizando el algoritmo Mixture of Gaussians (MoG2) de OpenCV. A continuación se describe su funcionamiento paso a paso:

Detecta movimiento en cada frame del video.
Asigna y sigue objetos en movimiento usando centroides y un algoritmo de asignación óptima.
Dibuja trayectorias, IDs y bounding boxes en tiempo real.
Guarda las trayectorias de todos los objetos detectados en un archivo CSV para análisis posterior.
