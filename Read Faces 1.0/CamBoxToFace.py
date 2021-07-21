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

video = cv2.VideoCapture(0)
count = 0
while(True):
    count += 1
    ret, frame = video.read()
    frame = cv2.flip(frame,1)
    #frame = cv2.imshow('Camera', frame)
    classifier = cv2.CascadeClassifier(f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml')
    bboxes = classifier.detectMultiScale(frame)
    cv2.imwrite(f'../Images Geteds/image{count}.png',frame)

    for box in bboxes:
        x, y, width, height = box
        x2, y2 = x + width, y + height
        # draw a rectangle over the pixels
        rect = rectangle(frame, (x, y), (x2, y2), (0,0,255), 1)

    #frame = cv2.resize(frame, (1336,726))
    frame = cv2.imshow('Camera', frame)
    
    # mostra video na tela, se precionar Q do teclado desliga a camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
video.release()
# Destroy all the windows
cv2.destroyAllWindows()