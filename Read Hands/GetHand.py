import cv2 


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

hand_cascade = cv2.CascadeClassifier("Models/palm.xml")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    hands = hand_cascade.detectMultiScale(frame_gray, 1.1, 5)
    print(hands)

    for (x, y, w, h) in hands:
        rect = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 1)

    #apresenta frame
    original = cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()