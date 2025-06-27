# Este programa utiliza el algoritmo Mixture of Gaussians (MoG2)
# para detectar movimiento en un video, trazar la trayectoria del objeto detectado
# y guardar coordenadas en archivo csv filtrando IDs con registros mayores que min_frames_to_CSV.
# by YnerPoe  # ynerpoe@gmail.com  # labtec@umce.cl
# junio-2025    
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 YnerPoe

import cv2
import numpy as np
import random
from scipy.optimize import linear_sum_assignment
import csv

# Inicializar el algoritmo Mixture of Gaussians (MOG2)
LEARNING_RATE = -1
motiondetection = cv2.createBackgroundSubtractorMOG2()

# Cargar video desde archivo
cap = cv2.VideoCapture('path/video.mp4')  # Reemplaza con la ruta de tu video

# Parámetros de ajuste
ajuste_tamaño = 100      # Área mínima para considerar un contorno como objeto
ajuste_distacia = 75     # Distancia máxima para asociar un centroide a un objeto existente

# Validar si el video se cargó correctamente
if not cap.isOpened():
    print("Error: No se pudo cargar el video.")
    exit()

# Obtener los FPS del video y calcular el tiempo de espera entre frames
fps = int(cap.get(cv2.CAP_PROP_FPS))
timeframe = max(1, int(1000 / fps))

# Diccionario para almacenar las trayectorias de los objetos
objects = {}
next_object_id = 0
max_disappeared = fps # Número máximo de frames antes de eliminar un objeto

colors = {}

# Lista global para trayectorias: [ID, frame_idx, x, y]
trayectorias_csv = []

def get_color(object_id):
    """Devuelve un color único para cada ID."""
    if object_id not in colors:
        colors[object_id] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    return colors[object_id]

def euclidean_distance(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

frame_idx = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o error al leer el frame.")
            break

        # Reducción de tamaño para mayor velocidad
        frame = cv2.resize(frame, (640, 480))

        motion_mask = motiondetection.apply(frame, LEARNING_RATE)
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_centroids = [
            (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            for contour in contours if cv2.contourArea(contour) > ajuste_tamaño
            if (M := cv2.moments(contour))["m00"] > 0
        ]

        updated_objects = {}
        used_centroids = set()
        object_ids = list(objects.keys())
        object_last_points = [objects[oid][0][-1] for oid in object_ids]

        if object_last_points and current_centroids:
            # Matriz de costos (distancias)
            cost_matrix = np.zeros((len(object_last_points), len(current_centroids)), dtype=np.float32)
            for i, obj_pt in enumerate(object_last_points):
                for j, cen in enumerate(current_centroids):
                    cost_matrix[i, j] = euclidean_distance(obj_pt, cen)

            rows, cols = linear_sum_assignment(cost_matrix)
            assigned_rows = set()
            for row, col in zip(rows, cols):
                object_id = object_ids[row]
                trajectory, disappeared = objects[object_id]
                if cost_matrix[row, col] < ajuste_distacia:
                    trajectory.append(current_centroids[col])
                    updated_objects[object_id] = (trajectory, 0)
                    assigned_rows.add(row)
                    used_centroids.add(col)
                    # Guardar para CSV
                    trayectorias_csv.append([object_id, frame_idx, current_centroids[col][0], current_centroids[col][1]])
                else:
                    updated_objects[object_id] = (trajectory, disappeared + 1)
                    assigned_rows.add(row)
            # Marcar objetos no asignados como desaparecidos
            for idx, object_id in enumerate(object_ids):
                if idx not in assigned_rows:
                    trajectory, disappeared = objects[object_id]
                    updated_objects[object_id] = (trajectory, disappeared + 1)
        else:
            # Si no hay objetos existentes, solo incrementar desaparecidos
            for object_id, data in objects.items():
                trajectory, disappeared = data
                updated_objects[object_id] = (trajectory, disappeared + 1)

        # Eliminar objetos desaparecidos por demasiado tiempo
        objects = {object_id: data for object_id, data in updated_objects.items() if data[1] < max_disappeared}

        # Crear nuevos objetos para los centroides no asignados
        for idx, centroid in enumerate(current_centroids):
            if idx not in used_centroids:
                objects[next_object_id] = ([centroid], 0)
                trayectorias_csv.append([next_object_id, frame_idx, centroid[0], centroid[1]])
                next_object_id += 1

        # Precalcula centroides y bounding boxes de los contornos válidos
        contornos_info = []
        for contour in contours:
            if cv2.contourArea(contour) > ajuste_tamaño:
                x, y, w, h = cv2.boundingRect(contour)
                cx = x + w // 2
                cy = y + h // 2
                contornos_info.append({'centroid': (cx, cy), 'bbox': (x, y, w, h)})

        # Dibujar trayectorias, etiquetas y bounding box de los objetos
        for object_id, data in objects.items():
            trajectory, _ = data
            color = get_color(object_id)
            # Dibuja solo los últimos 30 puntos de la trayectoria para mayor velocidad visual
            for i in range(max(1, len(trajectory)-30), len(trajectory)):
                if i > 0:
                    cv2.line(frame, trajectory[i - 1], trajectory[i], color, 2)
            # Dibuja el texto del ID
            cv2.putText(frame, f"ID {object_id}", trajectory[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Buscar el bounding box más cercano al último punto de la trayectoria
            min_dist = float('inf')
            best_bbox = None
            last_point = trajectory[-1]
            for info in contornos_info:
                dist = euclidean_distance(info['centroid'], last_point)
                if dist < min_dist:
                    min_dist = dist
                    best_bbox = info['bbox']
            # Dibuja el bounding box si se encontró uno cercano
            if best_bbox:
                x, y, w, h = best_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        frame_idx += 1
        if cv2.waitKey(timeframe) == ord('q'):
            break

    # Guardar CSV solo una vez al final
    with open('trayectorias.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'Frame', 'X', 'Y'])  # Cabecera
        writer.writerows(trayectorias_csv)
finally:
    cap.release()
    cv2.destroyAllWindows()
