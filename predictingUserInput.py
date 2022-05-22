import cv2
import numpy as np
import tensorflow as tf
from tkinter import filedialog

categories = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]

model = tf.keras.models.load_model("emotionDetectionViaFace.h5")

correct = 0
incorrectArray = [[], [], [], [], [], []]
correctArray = [[], [], [], [], [], []]
imgSize = 48
trainingData = []
count = 0
label = ""

def createTrainingData(img):
    try:
        #convert the images to the correct size as well as grayscale
        imgArray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)/ 255.0
        newArray = cv2.resize(imgArray, (imgSize, imgSize))
        newArray = newArray.reshape(-1, imgSize, imgSize, 1)
        preds = model.predict(newArray)
        label = categories[preds.argmax()]
        print("This image shows a person with a " + label + " expression")

        #output the percentage of each epression showed in the image
        count = 0
        for expression in categories:
            print(expression + " = " + str(round((preds[0][count] * 100),1)) + "%")
            count = count + 1

    except Exception as e:
        pass

print("Please choose a image file, with a face")

#open window dialog to allow user to select their file
fileName = filedialog.askopenfilename()

createTrainingData(fileName)

input("Press Enter to exit...")





