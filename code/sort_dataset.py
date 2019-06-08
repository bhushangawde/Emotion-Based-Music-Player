import glob
from shutil import copyfile

emotions = ["neutral","anger","contempt","disgust","fear","happy" , "sadness" , "surprise"]
participants = glob.glob("source_emotions\\*")
#images = glob.glob("source_images\\*")
#print(participants)

for participant in participants:
    list_part_number = "%s" %participant[-4:]
    #print(list_part_number)
    for number in glob.glob("%s\\*" %participant):
        #print(number)
        for files in glob.glob("%s\\*" %number):
            #print(files)
            current = files[20:-30]
            #print(current)
            file = open(files, 'r')
            emotion = int(float(file.readline()))
            #print(emotion)
            source_emotion = glob.glob("source_images\\%s\\%s\\*" %(list_part_number,current))[-1]
            source_neutral = glob.glob("source_images\\%s\\%s\\*" %(list_part_number,current))[0]
            destination_neutral = "sorted_set\\neutral\\%s" %(source_neutral[25:])
            destination_emotion = "sorted_set\\%s\\%s" %(emotions[emotion] ,source_emotion[25:])
            copyfile(source_neutral, destination_neutral)
            copyfile(source_emotion, destination_emotion)
            
print("Images have been sorted!")    
