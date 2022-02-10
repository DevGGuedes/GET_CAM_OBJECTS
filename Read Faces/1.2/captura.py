import cv2

camera = cv2.VideoCapture(0)

caminho_model = "../../Testing Faces/haarcascade_frontalface_default.xml"
classificador = cv2.CascadeClassifier(caminho_model)

while(True):
    conectado, imagem = camera.read()
    imagem = cv2.flip(imagem,1)
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificador.detectMultiScale(imagem_cinza, 
                        scaleFactor = 1.5, minSize = (100,100))

    #L e A largura e altura
    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x,y), (x + l, y + a), (255,0,0), 2 )

    cv2.imshow('Camera', imagem)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyWindow()