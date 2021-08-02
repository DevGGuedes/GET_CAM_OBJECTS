#faz leitura das emoções pela foto
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

parser = argparse.ArgumentParser()
parser.add_argument('--image')

args = parser.parse_args()

#carrega modelos para detecção
faceProto    = "../Models/opencv_face_detector.pbtxt"
faceModel    = "../Models/opencv_face_detector_uint8.pb"
ageProto     = "../Models/age_deploy.prototxt"
ageModel     = "../Models/age_net.caffemodel"
genderProto  = "../Models/gender_deploy.prototxt"
genderModel  = "../Models/gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
genderList=['Masculino','Feminino']

genderNet=cv2.dnn.readNet(genderModel,genderProto)
faceNet=cv2.dnn.readNet(faceModel,faceProto)

#carrega a imagem
video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20

while True:
	ret, frame = video.read()
	frame = cv2.flip(frame,1)
	faces = face_detection.detectMultiScale(frame, scaleFactor = 1.2, minNeighbors = 5, minSize = (30,30))

	#frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#original = frame.copy()

	for (x, y, w, h) in faces:

		roi = frame[y:y + h, x:x + w]
		roi = cv2.resize(roi, (48, 48))
		cv2.imshow('Camera', roi)

		roi = roi.astype('float') / 255
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis = 0)


		#print(x, y, w, h)

'''def highlightFace(net, frame, conf_threshold=0.7):
	faceBoxes = None
	
	original = frame.copy()
	frameHeight = original.shape[0]
	frameWidth = original.shape[1]

	frame_cinza = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

	for (x, y, w, h) in faces:
		roi = frame_cinza[y:y + h, x:x + w]
		roi = cv2.resize(roi, (48, 48))
		
		roi = roi.astype('float') / 255
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis = 0)
		cv2.imshow("Img", roi)


	return original, faceBoxes

while cv2.waitKey(1) < 0:
	hasFrame, frame = video.read()
	faces = face_detection.detectMultiScale(frame, scaleFactor = 1.2, minNeighbors = 5, minSize = (30,30))

	frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	for (x, y, w, h) in faces:
		roi = frame_cinza[y:y + h, x:x + w]
		roi = cv2.resize(roi, (48, 48))
		
		roi = roi.astype('float') / 255
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis = 0)
	
	cv2.imshow("Img", roi)'''

	#resultImg, faceBoxes = highlightFace(faceNet, faces)