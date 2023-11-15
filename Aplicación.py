import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import math

# Lista de videos para elegir
videos = {
    "1": {
        "path": r'C:\Users\Acer\Desktop\proyectometodos\Files\pelota1.mp4',
        "hsvVals": {'hmin': 9, 'smin': 83, 'vmin': 88, 'hmax': 180, 'smax': 127, 'vmax': 255},
        "canastaX1": 90,
        "canastaX2": 200,
        "canastaY": 350,  # Añadir la coordenada Y de la canasta
        "frameThreshold": 1  # Añadir frameThreshold específico
    },
    "2": {
        "path": r'C:\Users\Acer\Desktop\proyectometodos\Files\Videos\vid (1).mp4',
        "hsvVals": {'hmin': 8, 'smin': 96, 'vmin': 115, 'hmax': 14, 'smax': 255, 'vmax': 255},
        "canastaX1": 300,
        "canastaX2": 405,
        "canastaY": 600,  # Añadir la coordenada Y de la canasta
        "frameThreshold": 1  # Añadir frameThreshold específico
    },
    "3": {
        "path": r'C:\Users\Acer\Desktop\proyectometodos\Files\WhatsApp Video 1.mp4',
        "hsvVals": {'hmin': 0, 'smin': 71, 'vmin': 69, 'hmax': 180, 'smax': 255, 'vmax': 255} ,
        "canastaX1": 90,
        "canastaX2": 200,
        "canastaY": 350,  # Añadir la coordenada Y de la canasta
        "frameThreshold": 2  # Añadir frameThreshold específico
    },
    "4": {
        "path": r'C:\Users\Acer\Desktop\proyectometodos\Files\objeto1.mp4',
        "hsvVals": {'hmin': 22, 'smin': 0, 'vmin': 0, 'hmax': 180, 'smax': 255, 'vmax': 255} ,
        "canastaX1": 90,
        "canastaX2": 200,
        "canastaY": 375,  # Añadir la coordenada Y de la canasta
        "frameThreshold": 2  # Añadir frameThreshold específico
    },
    "5": {
        "path": r'C:\Users\Acer\Desktop\proyectometodos\Files\objeto2.mp4',
        "hsvVals": {'hmin': 22, 'smin': 0, 'vmin': 0, 'hmax': 180, 'smax': 255, 'vmax': 255},
        "canastaX1": 90,
        "canastaX2": 200,
        "canastaY": 350,  # Añadir la coordenada Y de la canasta
        "frameThreshold": 2  # Añadir frameThreshold específico
    },
    # Agrega tantas configuraciones de video como necesites aquí...
}

# Pedir al usuario que elija el video
choice = input("Elige el video que quieres analizar (1, 2, 3, 4, 5): ")

# Configurar el video y los valores de HSV basados en la elección del usuario
videoConfig = videos.get(choice)
if not videoConfig:
    print("Elección no válida.")
    exit()

# Inicializar la captura de video
cap = cv2.VideoCapture(videoConfig["path"])

# Crear el objeto para encontrar el color
myColorFinder = ColorFinder(False)
hsvVals = videoConfig["hsvVals"] # Valores HSV para el video elegido

# Variables para los límites de la canasta
canastaX1, canastaX2 = videoConfig["canastaX1"], videoConfig["canastaX2"]
canastaY = videoConfig["canastaY"]  # Coordenada Y para la canasta

# Variables
posListX, posListY = [], []
prediction = False
frameCounter = 0
frameThreshold = videoConfig["frameThreshold"]  # Usar frameThreshold del video elegido
paused = False  # Variable para controlar la pausa después de 10 puntos

while True:
    if not paused:
        success, img = cap.read()
        if not success:
            print("Failed to read the video or video has ended.")
            break

    if img is None:
        continue

    imgColor, mask = myColorFinder.update(img, hsvVals)
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)
    
    if contours and frameCounter % frameThreshold == 0:
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])
    frameCounter += 1

    for i, (posX, posY) in enumerate(zip(posListX, posListY)):
        cv2.circle(imgContours, (posX, posY), 10, (0, 255, 0), cv2.FILLED)
        if i != 0:
            cv2.line(imgContours, (posX, posY), (posListX[i - 1], posListY[i - 1]), (0, 255, 0), 5)

    if len(posListX) >= 10 and not paused:
        A, B, C = np.polyfit(posListX, posListY, 2)
        for x in range(0, 1300):
            y = int(A * x ** 2 + B * x + C)
            if 0 <= y < img.shape[0]:
                cv2.circle(imgContours, (x, y), 2, (0, 0, 255), cv2.FILLED)

        discriminant = B ** 2 - (4 * A * (C - canastaY))
        if discriminant >= 0:
            x1 = int((-B - math.sqrt(discriminant)) / (2 * A))
            x2 = int((-B + math.sqrt(discriminant)) / (2 * A))
            prediction = canastaX1 < x1 < canastaX2 or canastaX1 < x2 < canastaX2
        else:
            prediction = False

        if prediction:
            cvzone.putTextRect(imgContours, "Basket", (50, 150), scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
        else:
            cvzone.putTextRect(imgContours, "No Basket", (50, 80), scale=5, thickness=5, colorR=(0, 0, 200), offset=20)

        paused = True  # Pausa el video después de alcanzar 10 puntos

    # Dibuja los límites de la canasta
    cv2.circle(imgContours, (canastaX1, canastaY), 10, (255, 0, 0), cv2.FILLED)
    cv2.circle(imgContours, (canastaX2, canastaY), 10, (255, 0, 0), cv2.FILLED)

    cv2.imshow("ImageColor", imgContours)

    if paused:
        # Si el video está pausado, esperamos hasta que el usuario presione una tecla
        key = cv2.waitKey(0)
        if key == ord('q'):  # Si se presiona 'q', se cierra el programa.
            break
        elif key == ord('c'):  # Si se presiona 'c', continuamos con el siguiente frame.
            paused = False
            continue
    else:
        # Si no hemos llegado a 10 puntos o no estamos en pausa, continuamos mostrando el video.
        key = cv2.waitKey(50)
        if key == ord('q'):  # Si se presiona 'q', se cierra el programa.
            break

# Liberar la captura de video y cerrar todas las ventanas al terminar
cap.release()
cv2.destroyAllWindows()
