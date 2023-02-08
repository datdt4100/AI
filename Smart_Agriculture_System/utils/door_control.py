import cv2
import numpy as np
from keras.models import load_model
import torch
import math
import keyboard

# Load the model
model = load_model('door_control_model.h5')

# Grab the labels from the labels.txt file. This will be used later.
labels = open('user_labels.txt', 'r').readlines()


def recognition(source):
    camera = cv2.VideoCapture(int(source))
    print("Detecting.....")
    for _ in range (100):
        # Grab the webcameras image.
        ret, image = camera.read()
        # Resize the raw image into (224-height,224-width) pixels.
        
        if(type(image) == type(None)):
            print("Can not get image!")
            pass
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        #imageShow = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)

        # Show the image in a window
        #cv2.imshow('Webcam Image', imageShow)
        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        # Normalize the image array
        image = (image / 127.5) - 1
        # Have the model predict what the current image is. Model.predict
        # returns an array of percentages. Example:[0.2,0.8] meaning its 20% sure
        # it is the first label and 80% sure its the second label.
        probabilities = model.predict(image)
        # Print what the highest value probabilitie label
        # print(labels[np.argmax(probabilities)])
        
        maxIdx = np.argmax(probabilities)
        maxVal = probabilities[0][maxIdx]
        maxVal = int(round(maxVal, 3)*1000)
        if maxVal > 995:
            count = count + 1
        else:
            print("Stranger")
            count = 0

        if count == 60:
            print("unlock!!")
            break

        # Listen to the keyboard for presses.
        
        if keyboard.is_pressed('o'):
            break
        
    camera.release() 
    
    #cv2.destroyAllWindows()