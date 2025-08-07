# Este programa utiliza el algoritmo Mixture of Gaussians (MoG2)
# para detectar movimiento en un video, trazar la trayectoria del objeto detectado
# y guardar coordenadas en archivo csv filtrando IDs con registros mayores que min_frames_to_CSV.
# se genera un registro de video de las trayectorias para contrastar los datos del archivo CSV
# by YnerPoe  # ynerpoe@gmail.com  # labtec@umce.cl
# julio-2025    
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 YnerPoe

import cv2
import numpy as np
import random
import csv
import time

print("INICIO DEL PROCESO")
start_time = time.time()  # Antes del procesamiento principal

# Inicializar el algoritmo Mixture of Gaussians (MOG2)
LEARNING_RATE = -1  # Tasa de aprendizaje automática
motiondetection = cv2.createBackgroundSubtractorMOG2()

# Archivo de salida para el video con trayectorias
video_tray_out = 'videos/video_tray_test.mp4'  
csv_out = 'tray_test.csv'  # Archivo CSV para guardar las trayectorias

# Cargar video desde archivo
cap = cv2.VideoCapture(.../ruta/video)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) # Número total de frames en el video
fps = cap.get(cv2.CAP_PROP_FPS) # Frames por segundo del video
duracion_segundos = total_frames / fps # Duración del video en segundos

# Tiempo minimo o Numero de frames que permanece un ID en el video para registralo luego en el CSV.
min_frames_to_CSV = round(fps) # 1 fps equivale a 1 segundo.
# Número máximo de frames que un objeto puede desaparecer antes de ser eliminado
max_disappeared = round(fps*3) # tiempo en segundos de inactividad

print(f"Tiempo mínimo para registrar un ID: {min_frames_to_CSV/fps} seg.")
print(f"Tiempo máximo de inactividad: {max_disappeared/fps} seg.")

# Ajustes para el algoritmo de detección de movimiento
ajuste_area = 600 # área mínima del contorno para ser considerado un objeto
ajuste_distancia = 100 # distancia mínima entre el centroide del objeto y el nuevo centroide detectado

timeframe = 1 # Tiempo de espera entre frames en milisegundos

# Validar si el video se cargó correctamente
if not cap.isOpened():
    print("Error: No se pudo cargar el video.")
    exit()

if total_frames == 0 or fps == 0:
    print("Error: No se pudo obtener información del video.")
    exit()  

objects = {} # Diccionario para almacenar las trayectorias de los objetos
next_object_id = 0  # Identificador único para cada objeto

colors = {}  # Diccionario para almacenar el color de cada ID

def get_color(object_id):
    if object_id not in colors:
        # Genera un color aleatorio en formato BGR
        colors[object_id] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    return colors[object_id]

# Función para calcular la distancia euclidiana
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

trayectorias_csv = []  # Lista para almacenar [ID, frame_idx, x, y]
frame_idx = 0

# Obtener dimensiones del frame para el VideoWriter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec para el VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para el VideoWriter
out = cv2.VideoWriter(video_tray_out, fourcc, fps, (frame_width, frame_height))
#out = cv2.VideoWriter('video_trayectorias.avi', fourcc, fps, (frame_width, frame_height))

# Imprimir información del video
print(f"Video cargado: {path}")
print(f"FPS del video: {fps}")
print(f"Número total de frames: {total_frames}")
print(f"Duración estimada del video: {duracion_segundos:.2f} segundos")
print(f"Dimensiones del video: {frame_width} x {frame_height} píxeles")

# Validar si el VideoWriter se creó correctamente
try:
    while True:
        # Leer el frame actual
        ret, frame = cap.read()
        if not ret:
            print("Fin del proceso de  video")
            break

        # Aplicar el algoritmo Mixture of Gaussians
        motion_mask = motiondetection.apply(frame, LEARNING_RATE)

        # Detectar contornos en la máscara de movimiento
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lista para almacenar los centroides detectados en este frame
        current_centroids = []

        for contour in contours:
            if cv2.contourArea(contour) > ajuste_area:  # Filtrar contornos pequeños
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])  # Coordenada X del centroide
                    cy = int(M["m01"] / M["m00"])  # Coordenada Y del centroide
                    current_centroids.append((cx, cy))

        # Asociar los centroides actuales con los objetos existentes
        updated_objects = {}
        for object_id, data in objects.items():
            trajectory, disappeared = data
            if current_centroids:
                # Encontrar el centroide más cercano
                distances = [euclidean_distance(trajectory[-1], c) for c in current_centroids]
                min_distance_index = int(np.argmin(distances))
                if distances[min_distance_index] < ajuste_distancia:
                    new_point = current_centroids[min_distance_index]
                    trajectory.append(new_point)
                    updated_objects[object_id] = (trajectory, 0)
                    trayectorias_csv.append([object_id, frame_idx, new_point[0], new_point[1]])
                    current_centroids.pop(min_distance_index)
                else:
                    updated_objects[object_id] = (trajectory, disappeared + 1)
            else:
                updated_objects[object_id] = (trajectory, disappeared + 1)

        # Eliminar objetos que han desaparecido por demasiado tiempo
        objects = {object_id: data for object_id, data in updated_objects.items() if data[1] < max_disappeared}

        # Crear nuevos objetos para los centroides restantes
        for centroid in current_centroids:
            objects[next_object_id] = ([centroid], 0)
            trayectorias_csv.append([next_object_id, frame_idx, centroid[0], centroid[1]])
            next_object_id += 1

        # Precalcula bounding boxes y centroides de los contornos válidos
        contornos_info = [
            {'centroid': (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])), 'bbox': cv2.boundingRect(contour)}
            for contour in contours
            if cv2.contourArea(contour) > ajuste_area and (M := cv2.moments(contour))["m00"] > 0
        ]

        # Dibuja trayectorias, etiquetas y bounding box de los objetos
        for object_id, data in objects.items():
            trajectory, _ = data
            color = get_color(object_id)
            # Solo los últimos 30 puntos para mayor velocidad visual
            start_idx = max(1, len(trajectory) - 30)
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1], trajectory[i], color, 2)
            # Dibuja el identificador del objeto
            cv2.putText(frame, f"ID {object_id}", trajectory[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Bounding box más cercano al último punto de la trayectoria
            last_point = trajectory[-1]
            min_dist = float('inf')
            best_bbox = None
            for info in contornos_info:
                dist = euclidean_distance(info['centroid'], last_point)
                if dist < min_dist:
                    min_dist = dist
                    best_bbox = info['bbox']
            if best_bbox:
                x, y, w, h = best_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        frame_idx += 1 # Incrementar el índice del frame

     # Salir si se presiona 'q'
        if cv2.waitKey(timeframe) == ord('q'):
            break
        out.write(frame)  # Guarda el frame procesado
finally:
    from collections import Counter
    # IDs identificados antes de filtrar
    ids_identificados = set(row[0] for row in trayectorias_csv)
    print("Número total de IDs identificados antes de filtrar:", len(ids_identificados))
    print("IDs identificados antes de filtrar:", sorted(ids_identificados))

    id_counts = Counter([row[0] for row in trayectorias_csv])
    # Filtrar solo los IDs con más de min_frames_to_CSV
    trayectorias_filtradas = [row for row in trayectorias_csv if id_counts[row[0]] > min_frames_to_CSV]
    # Obtener el conjunto de IDs filtrados
    ids_filtrados = set(row[0] for row in trayectorias_filtradas)
    print("Número total de IDs filtrados en el archivo CSV:", len(ids_filtrados))
    print("IDs registrados en el archivo de salida CSV:", sorted(ids_filtrados))

    # Guardar las trayectorias filtradas en un archivo CSV
    with open(csv_out, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'Frame', 'X', 'Y'])
        writer.writerows(trayectorias_filtradas)
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Tiempo total de ejecución: {total_time:.2f} segundos")

    print(f"FIN DEL PROCESO")


