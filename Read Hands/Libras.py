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

#mostra iamgem original
imagem = cv2.imread("Images/D.JPG")
#cv2.imshow('Foto', imagem)
#cv2.waitKey()

imagem_copia = np.copy(imagem)

imagem_largura = imagem.shape[1]
imagem_altura = imagem.shape[0]
proporcao = imagem_largura / imagem_altura

cor_pontoA, cor_pontoB, cor_linha = (14, 201, 255), (255, 0, 128), (192, 192, 192)
cor_txtponto = (10, 216, 245)

tamanho_fonte, tamanho_linha, tamanho_circulo, espessura = 5, 1, 4, 2

fonte = cv2.FONT_HERSHEY_SIMPLEX

entrada_altura = 256
entrada_largura = int(((proporcao * entrada_altura) * 8) // 8)

entrada_blob = cv2.dnn.blobFromImage(imagem, 1.0 / 255, (entrada_largura, entrada_altura), 
                                    (0, 0, 0), swapRB=False, crop=False)

modelo.setInput(entrada_blob)
saida = modelo.forward()
print(saida.shape)
#print(type(saida.shape))

pontos = []
limite = 0.1
for i in range(numero_pontos):
    mapa_confianca = saida[0, i, :, :]
    #print(mapa_confianca)
    mapa_confianca = cv2.resize(mapa_confianca, (imagem_largura, imagem_altura))

    _, confianca, _, ponto = cv2.minMaxLoc(mapa_confianca)

    if confianca > limite:
        cv2.circle(imagem_copia, (int(ponto[0]), int(ponto[1])), 5, cor_pontoA, 
                   thickness=espessura, lineType=cv2.FILLED)
        cv2.putText(imagem_copia, ' ' + (str(int(ponto[0]))) + ',' + 
                    str(int(ponto[1])), (int(ponto[0]), int(ponto[1])),
                    fonte, 0.3, cor_txtponto, 0, lineType=cv2.LINE_AA)

        cv2.circle(imagem, (int(ponto[0]), int(ponto[1])), tamanho_circulo,
                   cor_pontoA,
                   thickness=espessura, lineType=cv2.FILLED)
        cv2.putText(imagem, ' ' + "{}".format(i), (int(ponto[0]), 
                                                  int(ponto[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    cor_txtponto,
                    0, lineType=cv2.LINE_AA)

        pontos.append((int(ponto[0]), int(ponto[1])))

    else:
        pontos.append((0, 0))


for par in pares_poses:
    parteA = par[0]
    parteB = par[1]

    if pontos[parteA] != (0, 0) and pontos[parteB] != (0, 0):
        cv2.line(imagem_copia, pontos[parteA], pontos[parteB], cor_linha, 
                 tamanho_linha, lineType=cv2.LINE_AA)
        cv2.line(imagem, pontos[parteA], pontos[parteB], cor_linha, tamanho_linha, 
                 lineType=cv2.LINE_AA)

#Mostra imagem com as ligações dos pontos da mão e a posição
#cv2.imshow('Foto', imagem)
#cv2.waitKey()

posicao.posicoes = []
# Dedo polegar
posicao.verificar_posicao_DEDOS(pontos[1:5], 'polegar', altura.verificar_altura_MAO(pontos))
# Dedo indicador
posicao.verificar_posicao_DEDOS(pontos[5:9], 'indicador', altura.verificar_altura_MAO(pontos))
# Dedo médio
posicao.verificar_posicao_DEDOS(pontos[9:13], 'medio', altura.verificar_altura_MAO(pontos))
# Dedo médio
posicao.verificar_posicao_DEDOS(pontos[9:13], 'medio', altura.verificar_altura_MAO(pontos))
# Dedo anelar
posicao.verificar_posicao_DEDOS(pontos[13:17], 'anelar', altura.verificar_altura_MAO(pontos))
# Dedo mínimo
posicao.verificar_posicao_DEDOS(pontos[17:21], 'minimo', altura.verificar_altura_MAO(pontos))
print(posicao.posicoes)

posicao_mao = proximidade.verificar_proximidade_DEDOS(pontos)

for i, a in enumerate(alfabeto.letras):
  if proximidade.verificar_proximidade_DEDOS(pontos) == alfabeto.letras[i]:
    cv2.putText(imagem, ' ' + letras[i], (50,50), fonte, 1, cor_txtponto,
                tamanho_fonte, lineType=cv2.LINE_AA)

    print(f'Letra detectada {letras[i]}')

plt.figure(figsize= [14,10])
plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
cv2.imshow('Foto', imagem)
cv2.waitKey()