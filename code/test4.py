import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tkinter as tk

# GUI text (Graphical user interface text)
def update_text():
    global sentence
    sentence += current_letter
    text_var.set(sentence)


def reset_sentence():
    global sentence
    sentence = ""
    text_var.set(sentence)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = ["A", "B", "C", "Ok", "No", "1", "2", "3", "4", "5", "_"]

# New text window1
root = tk.Tk()
root.title("ASL Recognizer")
text_var = tk.StringVar()
text_var.set("Start forming sentences!")
sentence_label = tk.Label(root, textvariable=text_var, font=("Helvetica", 16))
sentence_label.pack()

reset_button = tk.Button(root, text="Reset", command=reset_sentence)
reset_button.pack()

# Variables
sentence = ""
current_letter = ""

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspect_ratio = h / w

        if aspect_ratio > 1:
            k = imgSize / h
            w_cal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (w_cal, imgSize))
            w_gap = math.ceil((300 - w_cal) / 2)

            imgWhite[:, w_gap:w_cal + w_gap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            current_letter = labels[index]

        else:
            k = imgSize / w
            h_cal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, h_cal))
            h_gap = math.ceil((300 - h_cal) / 2)

            imgWhite[h_gap:h_cal + h_gap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            current_letter = labels[index]

        cv2.rectangle(imgOutput, (x - offset, y - offset - 90), (x - offset + 50, y - offset - 50 + 50), (0, 255, 255),
                      cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImgWhite", imgWhite)

    else:
        # No hand detected, reset imgWhite and display message
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        text_var.set("No hand detected")

    cv2.imshow("Image", imgOutput)

    # FPS
    update_text()
    root.update()
    time.sleep(1.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
