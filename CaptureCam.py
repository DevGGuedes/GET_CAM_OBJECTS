import cv2

# define a video capture object
video = cv2.VideoCapture(0)
  
while(True):
      
    #Pega o quadro do vídeo
    ret, frame = video.read()
    #arruma a posicao da camera
    frame = cv2.flip(frame,1)
    #Exibir o quadro resultante
    frame = cv2.imshow('Camera', frame)

    # mostra video na tela
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
video.release()
# Destroy all the windows
cv2.destroyAllWindows()