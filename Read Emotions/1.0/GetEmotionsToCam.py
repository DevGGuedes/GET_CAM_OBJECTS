#faz leitura das emoções pela camera
import numpy as np
import pandas as pd
import zipfile
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
import cv2
from PIL import Image 
import PIL 

video = cv2.VideoCapture(0)
expressoes = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]

#Carregamento dos modelos
cascade_faces = "../Testing Faces/haarcascade_frontalface_default.xml"
#caminho da rede neural pré treinada
#caminho_modelo = "../Testing Faces/Models/modelo_01_expressoes.h5"
caminho_modelo = "../Testing Faces/Models/modelo_02_expressoes.h5"
#objeto especifico para deteccao de faces
face_detection = cv2.CascadeClassifier(cascade_faces)
classificador_emocoes = load_model(caminho_modelo, compile = False)


while(True):
    ret, frame = video.read()
    frame = cv2.flip(frame,1)
    faces = face_detection.detectMultiScale(frame, scaleFactor = 1.2, minNeighbors = 5, minSize = (20,20))
    #faces = face_detection.detectMultiScale(frame, scaleFactor = 1.5, minNeighbors = 7, minSize = (30,30))

    #transforma imagem original para escala de cinza
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #copia o frame original
    original = frame.copy()
    #print(faces, len(faces))
    
    for (x, y, w, h) in faces:

        roi = frame_cinza[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))

        roi = roi.astype('float') / 255
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis = 0)

        #Previsões, classifica as emoçoes
        preds = classificador_emocoes.predict(roi)[0]

        #Emoção detectada, pega maior numero da probabilidade
        emotion_probability = np.max(preds)

        #Apresenta Emoção detectada
        label = expressoes[preds.argmax()]
        print(f'Emoção encontrada: {label}')

        #Monta para apresentar o texto da emoção
        cv2.putText(original, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2, cv2.LINE_AA)
        cv2.rectangle(original, (x,y), (x + w, y + h), (255,0,0), 2 )

    #apresenta frame
    original = cv2.imshow('Camera', original)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()