# Este programa controla algunos LEDs con pyfirmata mediante Vision Artificial
# Es necesario instalar StandardFirmata en Arduino 
# ruta: Examples->Firmata->StandardFirmata
# By YnerPoe-LabTec (nov-2024)
# ynerpoe@gmail.com
# python -m serial.tools.list_ports (para conocer puertos disponibles)

import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
from pyfirmata import Arduino

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
board = Arduino("COM3")

# Palma de la mano (centro)
def palm_centroid(coordinates_list):
     coordinates = np.array(coordinates_list)
     centroid = np.mean(coordinates, axis=0)
     centroid = int(centroid[0]), int(centroid[1])
     return centroid

# Pulgar, índice, medio, anular y meñique (dedos)
thumb_points = [1, 2, 4]
palm_points = [0, 1, 2, 5, 9, 13, 17]
fingertips_points = [8, 12, 16, 20]
finger_base_points =[6, 10, 14, 18]

with mp_hands.Hands(
     model_complexity=1,
     max_num_hands=1,
     min_detection_confidence=0.5,
     min_tracking_confidence=0.5) as hands:
   
     while cap.isOpened():
          ret, frame = cap.read()
          if not ret:
               print("Ignorar frame vacío")
               break
          
          frame = cv2.flip(frame, 1)
          height, width, _ = frame.shape
          results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
          thickness = [1,1,1,1,1]

          if results.multi_hand_landmarks:
               coordinates_thumb = []
               coordinates_palm = []
               coordinates_ft = []
               coordinates_fb = []
               for hand_landmarks in results.multi_hand_landmarks:
                    for index in thumb_points:
                         x = int(hand_landmarks.landmark[index].x * width)
                         y = int(hand_landmarks.landmark[index].y * height)
                         coordinates_thumb.append([x, y])
                    
                    for index in palm_points:
                         x = int(hand_landmarks.landmark[index].x * width)
                         y = int(hand_landmarks.landmark[index].y * height)
                         coordinates_palm.append([x, y])
                    
                    for index in fingertips_points:
                         x = int(hand_landmarks.landmark[index].x * width)
                         y = int(hand_landmarks.landmark[index].y * height)
                         coordinates_ft.append([x, y])
                    
                    for index in finger_base_points:
                         x = int(hand_landmarks.landmark[index].x * width)
                         y = int(hand_landmarks.landmark[index].y * height)
                         coordinates_fb.append([x, y])

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
               # Pulgar
                    p1 = np.array(coordinates_thumb[0])
                    p2 = np.array(coordinates_thumb[1])
                    p3 = np.array(coordinates_thumb[2])

                    l1 = np.linalg.norm(p2 - p3)
                    l2 = np.linalg.norm(p1 - p3)
                    l3 = np.linalg.norm(p1 - p2)

               # Calcular el ángulo de los dedos 
                    angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                    thumb_finger = np.array(False)
                    if angle > 160:
                         thumb_finger = np.array(True)
                    
               # Índice, medio, anular y meñique
                    nx, ny = palm_centroid(coordinates_palm)
                    cv2.circle(frame, (nx, ny), 3, (255, 0, 0), 2) #punto central mano
                    coordinates_centroid = np.array([nx, ny])
                    coordinates_ft = np.array(coordinates_ft)
                    coordinates_fb = np.array(coordinates_fb)

               # Distancias
                    d_centrid_ft = np.linalg.norm(coordinates_centroid - coordinates_ft, axis=1)
                    d_centrid_fb = np.linalg.norm(coordinates_centroid - coordinates_fb, axis=1)
                    dif = d_centrid_ft - d_centrid_fb
                    fingers = dif > 0
                    fingers = np.append(thumb_finger, fingers)
                    fingers_counter = str(np.count_nonzero(fingers==True))
                    
                    for (i, finger) in enumerate(fingers):
                         if finger == True:
                              thickness[i] = 0
             
               # Control de Arduino
                    if thickness[0]== 0:    # Pulgar
                         board.digital[8].write(1)
                    else:
                         board.digital[8].write(0)

                    if thickness[1] == 0:  # Indice
                         board.digital[9].write(1)
                    else:
                         board.digital[9].write(0)

                    if thickness[2] == 0:  # Medio
                         board.digital[10].write(1)
                    else:
                         board.digital[10].write(0)

                    if thickness[3] == 0:  # Anular
                         board.digital[11].write(1)
                    else:
                         board.digital[11].write(0)

                    if thickness[4] == 0:  # Menique
                         board.digital[12].write(1)
                    else:
                         board.digital[12].write(0)

                    if thickness == [1,1,1,1,1]:
                         board.digital[8].write(0)
                         board.digital[9].write(0)
                         board.digital[10].write(0)
                         board.digital[11].write(0)
                         board.digital[12].write(0)                         
          else: # Apaga todo 
               board.digital[8].write(0)
               board.digital[9].write(0)
               board.digital[10].write(0)
               board.digital[11].write(0)
               board.digital[12].write(0)  
          
          cv2.imshow('Hand Tracking', frame)
          if cv2.waitKey(10) & 0xFF == ord('q'):
               break
cap.release()
cv2.destroyAllWindows()        
board.exit()
