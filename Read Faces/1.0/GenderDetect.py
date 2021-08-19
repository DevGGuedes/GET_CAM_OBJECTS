#arquivo para detectar sexo por imagem 

import cv2
import math
import argparse

# função para colocar retangulo na face da pessoa
def highlightFace(net, frame, conf_threshold=0.7):
	frameOpencvDnn = frame.copy()
	frameHeight = frameOpencvDnn.shape[0]
	frameWidth = frameOpencvDnn.shape[1]
	
	blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
	
	net.setInput(blob)
	detections = net.forward()
	#print(f'detections {detections} - len {len(detections)}')
	
	faceBoxes = []

	#frameOpencvDnn = cv2.cvtColor(frameOpencvDnn, cv2.COLOR_BGR2GRAY)
	#roi = frameOpencvDnn[48:48 + frameHeight, 48:48 + frameWidth]
	#cv2.imshow("func", roi)

	#for i in range(detections.shape[len(detections)]):
	for i in range(detections.shape[1]):
		confidence = detections[0,0,i,2]
		
		if confidence > conf_threshold:

			x1=int(detections[0,0,i,3]*frameWidth)
			y1=int(detections[0,0,i,4]*frameHeight)
			x2=int(detections[0,0,i,5]*frameWidth)
			y2=int(detections[0,0,i,6]*frameHeight)

			#retorna os pontos de encontro da imagem
			faceBoxes.append([x1,y1,x2,y2]) 
			print(f'faceBoxes {faceBoxes}')

			f = cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
			
	
	return frameOpencvDnn,faceBoxes

parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

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

while cv2.waitKey(1) < 0:
	hasFrame, frame = video.read()
	frame = cv2.flip(frame,1)

	if not hasFrame:
		cv2.waitKey()
		break
	
	#highlightFace retorna faces com o quadrado na face e o resultado da imagem e pontos aonde encontrou a face
	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	resultImg, faceBoxes = highlightFace(faceNet, frame)

	#verifica se achou faces
	if not faceBoxes:
		print("Nenhuma face detectada")

	for faceBox in faceBoxes:
		face = frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

		blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
		
		#faz previsões para definir o genero
		genderNet.setInput(blob)
		genderPreds = genderNet.forward()
		gender = genderList[genderPreds[0].argmax()]
		
		print(f'Genero: {gender}')

	cv2.putText(resultImg, f'{gender}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
	cv2.imshow("Detecao de genero", resultImg)