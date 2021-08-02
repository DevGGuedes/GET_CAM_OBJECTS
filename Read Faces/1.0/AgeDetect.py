# arquivo para detecção de idades em uma unica imagem
# modo para chamada via cmd "python AgeDetect.py --image C:\Users\gabri\Desktop\Documentos\IBM\Estudos\GET_CAM_OBJECTS\Images\Fotoperfil.jpg --face face_detector --age age_detector"

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-f", "--face", required=True,
	help="path to face detector model directory")
ap.add_argument("-a", "--age", required=True,
	help="path to age detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# define the list of age buckets our age detector will predict
# definir a lista de intervalos de idade que nosso detector de idade irá prever
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)","(38-43)", "(48-53)", "(60-100)"]

# load our serialized face detector model from disk
# carrega nosso modelo de detector facial serializado do disco
print("[INFO] loading face detector model...")

#prototxtPath = os.path.sep.join("deploy.prototxt")
#weightsPath = os.path.sep.join("res10_300x300_ssd_iter_140000.caffemodel")

prototxtPath = "../Models/deploy.prototxt"
weightsPath  = "../Models/res10_300x300_ssd_iter_140000.caffemodel"

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load our serialized age detector model from disk
# carrega nosso modelo de detector de idade serializado do disco

print("[INFO] loading age detector model...")
#prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
#weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])

prototxtPath = "../Models/age_deploy.prototxt"
weightsPath  = "../Models/age_net.caffemodel"

ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the input image and construct an input blob for the image
# carrega a imagem de entrada e construa um blob de entrada para a imagem
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))
# pass the blob through the network and obtain the face detections
print("[INFO] computing face detections...")
faceNet.setInput(blob)
detections = faceNet.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
	
	# extract the confidence (i.e., probability) associated with the prediction
	# extrai a confiança (ou seja, probabilidade) associada aa predição
	confidence = detections[0, 0, i, 2]
	# filter out weak detections by ensuring the confidence is
	# filtra as detecções fracas, garantindo que a confiança é maior
	
	# greater than the minimum confidence
	# maior do que a confiança mínima
	if confidence > args["confidence"]:
		
		# compute the (x, y) -coordinates of the bounding box for the object
		# calcula as coordenadas (x, y) da caixa delimitadora para o objeto
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		
		# extract the ROI of the face and then construct a blob from *only* the face ROI
		# extrai o ROI do rosto e, a seguir, constrói um blob a partir de *apenas* o ROI de face
		face = image[startY:endY, startX:endX]
		faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        
        # make predictions on the age and find the age bucket with the largest corresponding probability
		# faz previsões sobre a idade e encontrar a faixa etária com a maior probabilidade correspondente
		ageNet.setInput(faceBlob)
		preds = ageNet.forward()
		i = preds[0].argmax()
		age = AGE_BUCKETS[i]
		ageConfidence = preds[0][i]
		
		# display the predicted age to our terminal
		# mostrar a idade prevista para o nosso terminal
		text = "{}: {:.2f}%".format(age, ageConfidence * 100)
		print("[INFO] {}".format(text))

		# draw the bounding box of the face along with the associated predicted age
		# desenha a caixa delimitadora do rosto junto com a idade prevista associada
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# display the output image
print(f'preds {preds}')
cv2.imshow("Image", image)
cv2.waitKey(0)