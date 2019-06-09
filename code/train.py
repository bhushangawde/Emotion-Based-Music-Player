import cv2
import glob
import numpy as np

lbphfr = cv2.face.LBPHFaceRecognizer_create()

data = {}
emotions = ["anger", "happy", "sadness"]
num_iterations = 20

def get_data():
    training_data = []
    training_labels = []

    for emotion in emotions:
        training = training = glob.glob("dataset\\%s\\*" %emotion)
        for item in training:
            image = cv2.imread(item) 
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            training_data.append(gray)
            training_labels.append(emotions.index(emotion))

    return training_data, training_labels

def run():
    training_data, training_labels = get_data()
    
    print("Training classifier")
    print("Size of training set: " + str(len(training_labels)) + " images")
    
    for i in range(num_iterations):
        lbphfr.train(training_data, np.asarray(training_labels))

def train():
    run()
    print("Saving trained model")
    lbphfr.save("model/trainedEmotionClassifier.xml")
    print("Model has been saved!")
    
lbphfr = cv2.face.LBPHFaceRecognizer_create()
result = train()