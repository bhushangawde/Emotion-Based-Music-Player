import cv2
import glob
faceDet = cv2.CascadeClassifier("OpenCV_FaceCascade/haarcascade_frontalface_default.xml")
faceDet_2 = cv2.CascadeClassifier("OpenCV_FaceCascade/haarcascade_frontalface_alt2.xml")
faceDet_3 = cv2.CascadeClassifier("OpenCV_FaceCascade/haarcascade_frontalface_alt.xml")
faceDet_4 = cv2.CascadeClassifier("OpenCV_FaceCascade/haarcascade_frontalface_alt_tree.xml")

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
f_height = 350
f_width = 350

def detect_faces(emotion):
    files = glob.glob("sorted_set\\%s\\*" %(emotion))
    filenumber = 0;
    for file in files:
        img = cv2.imread(file) 
        #print(filenumber)
        #print(type(img))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        face = faceDet.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=10, minSize=(5,5),flags = cv2.CASCADE_SCALE_IMAGE)
        face_2 = faceDet_2.detectMultiScale(gray,scaleFactor = 1.1, minNeighbors = 10, minSize=(5,5),flags = cv2.CASCADE_SCALE_IMAGE)
        face_3 = faceDet_3.detectMultiScale(gray,scaleFactor = 1.1, minNeighbors = 10, minSize=(5,5),flags = cv2.CASCADE_SCALE_IMAGE)
        face_4 = faceDet_4.detectMultiScale(gray,scaleFactor = 1.1, minNeighbors = 10, minSize=(5,5),flags = cv2.CASCADE_SCALE_IMAGE)
        #print(filenumber)
        if len(face) == 1:
            facefeatures = face
        elif len(face_2) == 1:
            facefeatures = face_2
        elif len(face_3) == 1:
            facefeatures = face_3
        elif len(face_4) == 1:
            facefeatures = face_4
        else:
            facefeatures = ""
        
        for(x,y,width,height) in facefeatures:
            #print("Face in file %s" %file)
            image = gray[y:y+height , x:x+width]
            
            try:
                resized_image = cv2.resize(image,(f_height,f_width))
                print(resized_image.shape)
                cv2.imwrite("dataset\\%s\\%s.jpg" %(emotion,filenumber) , resized_image)
            except:
                pass
        #print(face)
        filenumber += 1
     
for emotion in emotions:
    detect_faces(emotion)