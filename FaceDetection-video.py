import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#Obtener imagen a procesar a traves de video
#cap = cv2.VideoCapture(0)


#Obtener imagen a procesar a traves de imagen
# CV_LOAD_IMAGE_UNCHANGED (<0) loads the image as is (including the alpha channel if present)
# CV_LOAD_IMAGE_GRAYSCALE ( 0) loads the image as an intensity one
# CV_LOAD_IMAGE_COLOR (>0) loads the image in the BGR format

img = cv2.imread('person-test-detection.png')

#while 1:
# determina si el frame se lee corectamente
# ret, img = cap.read()

# Cambia los colores de la imagen
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# algoritmo de ML para detectar caras
faces = face_cascade.detectMultiScale(gray)

# Dibuja rectangulos en los objetos detectados
for (x, y, w, h) in faces:
    # image = cv2.rectangle(image, start_point, end_point, color, thickness)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    # algoritmo de ML para detectar ojos
    eyes = eye_cascade.detectMultiScale(roi_gray)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv2.imshow('Process Image', img)

# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break

cv2.waitKey(0)

#cap.release()
cv2.destroyAllWindows()