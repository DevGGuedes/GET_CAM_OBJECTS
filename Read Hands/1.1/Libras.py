import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("Models")

import extrator_POSICAO as posicao
import extrator_ALTURA as altura
import extrator_PROXIMIDADE as proximidade
import alfabeto

arquivo_proto = "Models/pose_deploy.prototxt"
arquivo_pesos = "Models/pose_iter_102000.caffemodel"
numero_pontos = 22
pares_poses = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], 
              [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
              [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
letras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q',
          'R', 'S', 'T', 'U', 'V', 'W']

modelo = cv2.dnn.readNetFromCaffe(arquivo_proto, arquivo_pesos)

imagem = cv2.imread("/content/imagens/hand/Libras/A.JPG")
cv2.imshow('MediaPipe Hands', imagem)