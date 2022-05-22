import numpy as np
import os
import cv2
import random as ran
import pickle

#diractory where the images are
dataDir = 'C:/Users/jacka/Downloads/fer2013/train'

#infomation neede to create labeled data
categories = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
imgSize = 48

trainingData = []

#function to label and store the data
def createTrainingData():
    for category in categories:
        path = os.path.join(dataDir, category)
        classNum = categories.index(category)

        for img in os.listdir(path):

            try:
                imgArray = cv2.imread(os.path.join(path, img))
                newArray = cv2.resize(imgArray, (imgSize, imgSize))
                trainingData.append([newArray, classNum])
                print(newArray)
            except Exception as e:
                pass

#create the data
createTrainingData()

#shuffle the data so the model can not overfit first time
ran.shuffle(trainingData)

x = []
y = []
count = 0

#build data infomation arrays
for features, label in trainingData:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, imgSize, imgSize, 1)

#Store the data inside of .pickle files
pickleOut = open("xtrainPickleColour.pickle", "wb")
pickle.dump(x, pickleOut)
pickleOut.close()

pickleOut = open("ytrainPickleColour.pickle", "wb")
pickle.dump(y, pickleOut)
pickleOut.close()

#open and load the data to check its worked
pickleIn = open("xtrainPickleColour.pickle", "rb")
x = pickle.load(pickleIn)

