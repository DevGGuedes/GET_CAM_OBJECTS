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
import math
import argparse

cascade_faces = "../../Testing Faces/haarcascade_frontalface_default.xml"
face_detection = cv2.CascadeClassifier(cascade_faces)

#carrega modelos para detecção
faceProto    = "../Models/opencv_face_detector.pbtxt"
faceModel    = "../Models/opencv_face_detector_uint8.pb"
ageProto     = "../Models/age_deploy.prototxt"
ageModel     = "../Models/age_net.caffemodel"
genderProto  = "../Models/gender_deploy.prototxt"
genderModel  = "../Models/gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
faceNet = cv2.dnn.readNet(faceModel,faceProto)
genderNet = cv2.dnn.readNet(genderModel,genderProto)

genderList = ['Masculino','Feminino']

# define a video capture object
video = cv2.VideoCapture(0)

def highlightFace(net, frame, conf_threshold=0.7):
	#frame = frame.copy()
	h = frame.shape[0]
	w = frame.shape[1]

	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)

	net.setInput(blob)
	detections=net.forward()
	faceBoxes=[]
	for i in range(detections.shape[2]):
		confidence=detections[0,0,i,2]
		if confidence>conf_threshold:
			x1 = int(detections[0,0,i,3] * w)
			y1 = int(detections[0,0,i,4] * h)
			x2 = int(detections[0,0,i,5] * w)
			y2 = int(detections[0,0,i,6] * h)
			faceBoxes.append([x1,y1,x2,y2])
			#cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), int(round(frame/150)), 8)
	
	return frame, faceBoxes

padding = 20

while(True):
	ret, frame = video.read()
	frame = cv2.flip(frame,1)
	faces = face_detection.detectMultiScale(frame, scaleFactor = 1.2, minNeighbors = 5, minSize = (30,30))
	
	original = frame.copy()
	#frame, faceBoxes=highlightFace(faceNet,frame)
	for (x, y, w, h) in faces:
		roi = frame[y:y + h, x:x + w]
		roi = cv2.resize(roi, (48, 48))
		#cv2.imshow('roi', roi)

		roi = roi.astype('float') / 255
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis = 0)
		
		original, faceBoxes = highlightFace(faceNet, frame)

		for faceBox in faceBoxes:
			face=frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
			
		blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        
        #faz previsões para definir o genero
		genderNet.setInput(blob)
		
		genderPreds = genderNet.forward()
        #print(f'genderPreds {genderPreds}')
		gender = genderList[genderPreds[0].argmax()]

		cv2.putText(original, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2, cv2.LINE_AA)
		cv2.rectangle(original, (x,y), (x + w, y + h), (255,0,0), 2 )

	cv2.imshow('Camera', original)
    
    
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
  
# After the loop release the cap object
video.release()
# Destroy all the windows
cv2.destroyAllWindows()