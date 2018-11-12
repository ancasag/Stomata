import cv2 as cv
#import argparse
import sys
import numpy as np
import os.path
# import imutils
# import argparse
# import time
import matplotlib.pyplot as plt
# from datetime import datetime

#Iniciamos parametros

confThreshold = 0.5  #Confidence threshold
nmsThreshold  = 0.4  #Non-maximum suppression threshold
inpWidth      = 416  #Width of network's input image
inpHeight     = 416  #Height of network's input image

classesFile = "vocEstomas.names";
classes = None
with open(classesFile, 'rt') as f:
	classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg";
modelWeights       = "yolov3_250000.weights";
#configuración de la red
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Obtener los nombres de las capas de salida
def getOutputsNames(net):
	# Obtener los nombres de todas las capas de la red
	layersNames = net.getLayerNames()
	# Obtenga los nombres de las capas de salida, es decir, las capas con salidas no conectadas
	return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Dibuja el cuadro delimitador predicho
def drawPred(classId, conf, left, top, right, bottom, frame):
	# Dibuja un cuadro delimitador
	cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
	
	label = '%.2f' % conf
		
	# Obtener la etiqueta para el nombre de la clase y su confianza
	if classes:
		assert(classId < len(classes))
		label = '%s:%s' % (classes[classId], label)

	#Mostrar la etiqueta en la parte superior del cuadro delimitador
	labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
	top = max(top, labelSize[1])
	#cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
	#cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Elimine los cuadros delimitadores con poca confianza utilizando el non-maximun supression.
def postprocess(frame, outs):
	frameHeight = frame.shape[0]
	frameWidth  = frame.shape[1]

	classIds    = []
	confidences = []
	boxes       = []
	# Escanea todos los resultados de los cuadros delimitadores de la red y conserva aquellos que tengan puntuaciones de confianza altas. 
	# Asigna la etiqueta de clase del cuadro como la clase con la puntuación más alta.
	classIds    = []
	confidences = []
	boxes       = []
	for out in outs:
		for detection in out:
			scores     = detection[5:]
			classId    = np.argmax(scores)
			confidence = scores[classId]
			if confidence > confThreshold:
				center_x = int(detection[0] * frameWidth)
				center_y = int(detection[1] * frameHeight)
				width    = int(detection[2] * frameWidth)
				height   = int(detection[3] * frameHeight)
				left     = int(center_x - width / 2)
				top      = int(center_y - height / 2)
				classIds.append(classId)
				confidences.append(float(confidence))
				boxes.append([left, top, width, height])

	# Realizar el non maximum suppression  para eliminar las cajas superpuestas redundantes con coeficientes más bajos.
	indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
	for i in indices:
		i = i[0]
		box    = boxes[i]
		left   = box[0]
		top    = box[1]
		width  = box[2]
		height = box[3]
		drawPred(classIds[i], confidences[i], left, top, left + width, top + height, frame)

# A partir de una imagen genera otra nueva con la deteccion de los objetos
def predict(imagePath):
	#args = parser.parse_args()
	#comprobamos que el argumento sea una imagen jpg
	#if imagePath.split('.')[1] != "jpg":
		#print('Esta imagen no tiene formato .jpg')
		#sys.exit(1)#si no lo es salimos
	#else:
		#print('Todo correcto')
		#si es una imagen la cargamos
	frame = cv.imread(imagePath)
			
	#Crea un blob 4D desde una imagen
	blob  = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

	# Establece la entrada a la red.
	net.setInput(blob)

	# Ejecuta el paso hacia adelante para obtener la salida de las capas de salida.
	outs  = net.forward(getOutputsNames(net))

	# Retire los cuadros delimitadores con poca confianza.
	postprocess(frame, outs)

	# Poner información de eficiencia. La función getPerfProfile devuelve el tiempo total para la inferencia (t) y los tiempos para cada una de las capas (en layersTimes)
	t, _  = net.getPerfProfile()
	label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
	cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

	#Variable con el nombre de la imagen
	outputFile = imagePath.split('.')[0]+'_predict.jpg'
	# Escribe el marco con las cajas de detección.
	cv.imwrite(outputFile, frame.astype(np.uint8));

def showImage(image):
	if len(image.shape)==3:
		img2 = image[:,:,::-1]
		plt.imshow(img2)
		plt.show()
	else:
		img2 = image
		plt.imshow(img2,cmap='gray')
		plt.show()

#predict("1042_1_A1.jpg")

