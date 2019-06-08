import cv2
import glob
import random
import numpy as np
#emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
#emotions = ["neutral", "anger", "disgust", "happy", "surprise"]
emotions = ["anger", "happy", "sadness"]

num_iterations = 20
finalscore = []

def get_emotion_files(emotion):
    files = glob.glob("dataset\\%s\\*" %emotion)
    #random.shuffle(files)
    train_files = files[:int(len(files)*0.85)]
    pred_files = files[-int(len(files)*0.15):]
    return train_files , pred_files


def make_train_test_set():
    training_data = [] 
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        train , pred = get_emotion_files(emotion)
        for file in train:
            img = cv2.imread(file,0)
            training_data.append(img)
            training_labels.append(emotions.index(emotion))
        
        for file in pred:
            img = cv2.imread(file,0)
            prediction_data.append(img)
            prediction_labels.append(emotions.index(emotion))
        
    return training_data , training_labels , prediction_data , prediction_labels
    
def run():
    training_data , training_labels , prediction_data , prediction_labels = make_train_test_set()
    #print(len(training_labels))
    #print(len(prediction_labels))
    print("Training begins")
    
    for i in range(num_iterations):
        lbphfr.train(training_data, np.asarray(training_labels))
   
    lbphfr.save("trainedEmotionClassifier.xml")
    print("Model has been saved!")
    
    print("Prediction Begins")
    count = 0;
    correct = 0;
    for img in prediction_data:
        pred, confidence = lbphfr.predict(img)
        #print(pred)
        if pred == prediction_labels[count]:
            correct += 1
        count +=1
        
    #print(correct)
    #print(count)
    return (100*correct)/count;

        
lbphfr = cv2.face.LBPHFaceRecognizer_create()
result = run()
print("Percentage correct are %d" %result)
    

#print("The final accuracy is %d percent." %np.mean(finalscore))