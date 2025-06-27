Este código realiza detección y seguimiento de objetos en movimiento en un video utilizando el algoritmo Mixture of Gaussians (MoG2) de OpenCV. A continuación se describe su funcionamiento paso a paso:

1) Detecta movimiento en cada frame del video.
2) Asigna y sigue objetos en movimiento usando centroides y un algoritmo de asignación óptima.
3) Dibuja trayectorias, IDs y bounding boxes en tiempo real.
4) Guarda las trayectorias de todos los objetos detectados en un archivo CSV para análisis posterior.
