import cv2
import mediapipe as mp
import time

video = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while(True):
    ret, frame = video.read()
    frame = cv2.flip(frame,1)
    #original = frame.copy()

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    #mostra array de maos na tela
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = frame.shape
                cx = int(lm.x*w)
                cy = int(lm.y*h)
                #print(id, cx, cy)
                #id vai de 0 a 20, determina cada ponto da mao
                #if id == 0:
                #    cv2.circle(frame, (cx, cy), 25, (255,0,255), cv2.FILLED)
                
                #faz circulos em todos os pontos da mao
                #cv2.circle(frame, (cx, cy), 10, (255,0,255), cv2.FILLED)

            #mostra bolinhas vermelhas em pontos das maos, com o HAND_CONNECTIONS faz as ligações dos pontos
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    '''cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = pTime

    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)'''

    #apresenta frame
    original = cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()