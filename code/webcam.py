import cv2
import numpy as np
video_capture = cv2.VideoCapture(0)
facecascade = cv2.CascadeClassifier("OpenCV_FaceCascade/haarcascade_frontalface_default.xml")

facedict = {}

def send_face(gray, face): #Crop the given face
    for (x, y, width, height) in face:
        cropped_face = gray[y:y+height, x:x+width]
    facedict["face%s" %(len(facedict)+1)] = cropped_face
    return cropped_face


while 1: 
    ret ,frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    face = facecascade.detectMultiScale(clahe_img , scaleFactor=1.1,minNeighbors=20, minSize=(10,10),flags=cv2.CASCADE_SCALE_IMAGE)
    
    #cv2.imshow("webcam" , frame)
    for (x,y,width,height) in face:
        cv2.rectangle(frame,(x,y) , (x+width , y+height),(0,0,255),2)
    
    if len(face) ==1:
        cropped_face = send_face(gray,face)
        cv2.imshow("detect" , cropped_face)
    cv2.imshow("webcam",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if len(facedict) == 10:
        break
    
    

    
     