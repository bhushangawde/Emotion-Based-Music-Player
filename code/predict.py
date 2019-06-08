import cv2
import numpy as np
import argparse
import time
import glob
import os
import pandas as pd
import random

df = pd.read_excel("Songs.xlsx")
video_capture = cv2.VideoCapture(0)
faceDet = cv2.CascadeClassifier("OpenCV_FaceCascade/haarcascade_frontalface_default.xml")
lbphfr = cv2.face.LBPHFaceRecognizer_create()
emotions = ["angry", "happy", "sad"]

facedict = {}
actions = {}
df = pd.read_excel("EmotionLinks.xlsx")
actions["angry"] = [x for x in df.angry.dropna()]
actions["happy"] = [x for x in df.happy.dropna()]
actions["sad"] = [x for x in df.sad.dropna()]

try:
    lbphfr.read("model/trainedEmotionClassifier.xml")
except:
    print("Model not found.")
    
def send_face(gray, face): 
    for (x, y, width, height) in face:
        cropped_face = gray[y:y+height, x:x+width]
    facedict["face%s" %(len(facedict)+1)] = cropped_face
    return cropped_face

def get_webcamframe():
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    return clahe_image

def detect_face():
    clahe_image = get_webcamframe()
    face = faceDet.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face) == 1:
        faceslice = send_face(clahe_image, face)
        return faceslice
    else:
        print("Multiple faces detected!")

def run_recognizer():
    predictions = []
    confidence = []
    for x in facedict.keys():
        pred, conf = lbphfr.predict(facedict[x])
        cv2.imwrite("images\\%s.jpg" %x, facedict[x])
        predictions.append(pred)
        confidence.append(conf)
    print("I think you're %s" %emotions[max(set(predictions), key=predictions.count)])
    predicted_emotion = emotions[max(set(predictions), key=predictions.count)]
    #print(max(set(predictions), key=predictions.count))
    actionlist = [x for x in actions[predicted_emotion]]
    random.shuffle(actionlist) 
    os.startfile(actionlist[0])
    print("Playing the song that will suit your mood! :)")
    
while True:
    detect_face()
    if len(facedict) == 10:
        run_recognizer()
        break

