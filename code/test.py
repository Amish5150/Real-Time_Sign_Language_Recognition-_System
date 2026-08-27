import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

offset = 20
imgSize = 300

folder = "Data/Ok "
counter = 0

label = ["A", "B", "C", "No", "Ok", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

while True:
    success,img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        #resizing and arranging it

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y - offset:y + h + offset,x - offset:x + w + offset] #to make the image size same as which it should be comaird to
       
        imgCropShape = imgCrop.shape

       

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((300-wCal)/2)

            imgWhite [:,wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction ,index)


        else:
             k = imgSize / w
             hCal = math.ceil(k*h)
             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
             imgResizeShape = imgResize.shape
             hGap = math.ceil((300-hCal)/2)

             imgWhite [hGap:hCal+hGap, :] = imgResize

             prediction, index = classifier.getPrediction(imgWhite)



        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImgWhite", imgWhite)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
   