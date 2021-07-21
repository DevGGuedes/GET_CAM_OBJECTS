## Reconhece rosto pela camera
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
import cv2
from PIL import Image 
import PIL
from win10toast import ToastNotifier
import tkinter as tk
from tkinter import simpledialog


ROOT = tk.Tk()
ROOT.withdraw()

#pedir o nome para salvar a foto do rosto reconhecido
#
#USER_INP = simpledialog.askstring(title="Savar reconhecimento facial",
#                                  prompt="Informe um nome para salvar sua foto: (Ex: rosto_gabriel.png)")


#notificação do windows
toast = ToastNotifier()
#toast.show_toast("Notificação","Aguarde Ativando Camera. Pressione Q para sair",duration=5,icon_path="icon.ico")
#toast.show_toast("Notificação","Aguarde Ativando Camera. qApós inicialização Pressione Q para sair",duration=5,)

#abrindo a camera
video = cv2.VideoCapture(0)
#video.set(cv2.CAP_PROP_FPS, 60)
#video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#video.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

count = 0

while(True):
    count += 1
    ret, frame = video.read()
    frame = cv2.flip(frame,1)
    #frame = cv2.imshow('Camera', frame)
    #casificação para o frame
    classifier = cv2.CascadeClassifier(f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml')
    bboxes = classifier.detectMultiScale(frame)
    # salva frame por fram detectado
    #cv2.imwrite(f'../Images Geteds/image{count}.png',frame)
    
    print(bboxes, len(bboxes))
    # bboxes é uma tupla, pode vir 0 = nenhuma face detectada, 1 = somente uma....
    
    for box in bboxes:
        x, y, width, height = box
        x2, y2 = x + width, y + height

        # draw a rectangle over the pixels
        rect = rectangle(frame, (x, y), (x2, y2), (0,0,255), 1)

        # salva o print da camera com o retangulo vermelho e recorta a img
        roi = frame[y:y+width, x:x+height]
        #cv2.imwrite(f'../Images Geteds/{USER_INP}}', roi)
        
        #mostra a img contada no retangulo vermelho
        #imshow('Rosto detectado', roi)

        # salva o print da camera com o retangulo vermelho
        #roi = frame[y:y+y2, x:x+x2]
        #cv2.imwrite(f'../Images Geteds/{USER_INP}', roi)

    #altera o tamenho do display da img
    #frame = cv2.resize(frame, (1336,726))
    frame = cv2.imshow('Camera', frame)
    
    # mostra video na tela, se precionar Q do teclado desliga a camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #abre uma nova tela mostrando a img que pegou
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imshow('Rosto detectado', roi)
        USER_INP = simpledialog.askstring(title="Savar reconhecimento facial",prompt="Informe um nome para salvar sua foto: (Ex: rosto_gabriel.png)")
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

# After the loop release the cap object
video.release()
# Destroy all the windows
cv2.destroyAllWindows()