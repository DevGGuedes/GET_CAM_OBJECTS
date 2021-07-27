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
cascade_faces = "../../Testing Faces/haarcascade_frontalface_default.xml"
#caminho da rede neural pré treinada
#caminho_modelo = "../Testing Faces/Models/modelo_01_expressoes.h5"
caminho_modelo = "../../Testing Faces/Models/modelo_02_expressoes.h5"
#objeto especifico para deteccao de faces
face_detection = cv2.CascadeClassifier(cascade_faces)
classificador_emocoes = load_model(caminho_modelo, compile = False)

#ret, frame = video.read()
largura_maxima = 600
redimensionar = True
proporcao = None
video_largura = None
video_altura = None

def TrataImg(frame):
    if (redimensionar and frame.shape[1] > largura_maxima):
    # precisamos deixar a largura e altura proporcionais (mantendo a proporção do vídeo original) para que a imagem não fique com aparência esticada
        proporcao = frame.shape[1] / frame.shape[0]
        # para isso devemos calcular a proporção (largura/altura) e usaremos esse valor para calcular a altura (com base na largura que definimos acima) 
        video_largura = largura_maxima
        video_altura = int(video_largura / proporcao)
    else:
        video_largura = video.shape[1]
        video_altura = video.shape[0]
    
    return frame

Qtdfaces = 0

while(True):
    ret, frame = video.read()
    frame = cv2.flip(frame,1)
    faces = face_detection.detectMultiScale(frame, scaleFactor = 1.2, minNeighbors = 5, minSize = (30,30))
    #faces = face_detection.detectMultiScale(frame, scaleFactor = 1.5, minNeighbors = 7, minSize = (30,30))

    if (redimensionar and frame.shape[1] > largura_maxima):
    # precisamos deixar a largura e altura proporcionais (mantendo a proporção do vídeo original) para que a imagem não fique com aparência esticada
        proporcao = frame.shape[1] / frame.shape[0]
        # para isso devemos calcular a proporção (largura/altura) e usaremos esse valor para calcular a altura (com base na largura que definimos acima) 
        video_largura = largura_maxima
        video_altura = int(video_largura / proporcao)
    else:
        video_largura = video.shape[1]
        video_altura = video.shape[0]

    #if redimensionar: # se redimensionar = True então redimensiona o frame para os novos tamanhos
    #    TrataImg(frame)
    #    frame = cv2.resize(frame, (video_largura, video_altura))

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

        if len(faces) == 1:
            #Qtdfaces = 1
            for (index, (emotion, prob)) in enumerate(zip(expressoes, preds)):
                # nomes das emoções
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                barra = int(prob * 150)  # calcula do tamanho da barra, com base na probabilidade
                espaco_esquerda = 7      # é a coordenada x onde inicia a barra. define quantos pixels tem de espaçamento à esquerda das barras, pra não ficar muito no canto. 
                if barra <= espaco_esquerda:
                    barra = espaco_esquerda + 1
                #mostra quadro com porcentagens de emoções
                cv2.rectangle(original, (espaco_esquerda, (index * 18) + 7), (barra, (index * 18) + 18), (200, 250, 20), -1)
                cv2.putText(original, text, (15, (index * 18) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1, cv2.LINE_AA)
                #table = cv2.rectangle(frame, (espaco_esquerda, (index * 18) + 7), (barra, (index * 18) + 18), (200, 250, 20), -1)
                #cv2.putText(frame, text, (15, (index * 18) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1, cv2.LINE_AA)
                #table = cv2.imshow('table', table)

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