# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:21:58 2024

@author: h4tec
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
from keras.preprocessing import image

print("[INFO] loading crack detector model...")

#model_path = "C:/Rupali Shinde/Source code/Crack_Detection.model"
#model = load_model('Crack_Detection.model')
#image_path = "C:/Rupali Shinde/Source code/test1.jpg"
#predictions = model.predict('test1.jpg')

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, 
                default="Crack_Detection.model", 
                help="path to trained face mask detector model")
args = vars(ap.parse_args())

	# load the face mask detector model from disk
print("[INFO] loading crack detector model...")
model = load_model(args["model"])

#img = cv2.imread(("C:/Rupali Shinde/Source code/test2.jpg"))
#print(img.shape)
#img = img_to_array(img)
#3img =img.resize(224,224)
#mg = (np.expand_dims(img,0))



# predicting images
img = image.load_img('test2.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print (classes)