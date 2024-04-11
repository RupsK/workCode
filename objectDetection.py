# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 18:00:16 2024

@author: h4tec
"""

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO
from ultralytics.engine.results import Results

import cv2
import os


import cv2
import numpy as np
global image

blackHoles= 0
redHoles = 0


def detect_red(image):
    #image = cv2.imread('C:/Users/h4tec/Desktop/redHole.jpg')
   
    # Save the modified pixels as .png')
    image = cv2.resize(image, (400,800))


   # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

   # Define the range of red color in HSV
   # These ranges may need to be adjusted for your specific image
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

   # Threshold the HSV image to get only red colors
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

   # Apply edge detection to the red regions
    edges = cv2.Canny(red_mask, 100, 200)

   # Dilate the edges to make them more pronounced
    kernel = np.ones((3,3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

   # Use the dilated edges as a mask to find blobs
   # Invert the mask for blob detection (blobs should be white on black background)
    inverted_mask = cv2.bitwise_not(dilated_edges)

   # Set up the SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 300  # Adjust as needed
    #params.maxArea =500
    params.filterByCircularity = True
    params.minCircularity  = 0.5  # Since holes might not be perfectly circular
    params.filterByConvexity = False
    #params.minConvexity = 0.5
    params.filterByInertia = False
    #params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(inverted_mask)
    global redHoles
    readHoles = 0
    redHoles =  (len(keypoints))
    print("Number of red holes = ")
    print(redHoles)
    result = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
   
    
def detect_black(image):
    # Load the image 
    #image_path = 'C:/Users/h4tec/Desktop/redHOle.jpg'  # Make sure to provide the correct path
    #image = cv2.imread(image_path)
    #image = cv2.resize(image, (400,800))
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image to make black areas appear as white
    inverted_image = cv2.bitwise_not(gray_image)

    # Threshold the inverted image to further enhance the detection of black holes
    _, thresh_image = cv2.threshold(inverted_image, 220, 255, cv2.THRESH_BINARY)

    # Initialize the SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255  # Since the black areas are now white after inversion
    params.filterByArea = True
    params.minArea = 200  # Min area of blobs

    params.maxArea = 300
    params.filterByCircularity = True  # Depending on the shape of the holes
    params.minCircularity  = 0.5
    params.filterByConvexity = True
    params.minConvexity = 0.5
     
    params.filterByInertia =True
    params.minInertiaRatio = 0.2

    # Create a blob detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs in the thresholded image
    keypoints = detector.detect(thresh_image)
    global blackHoles
    blackHole = len(keypoints)
    print("number of black holes =")
    print(blackHole)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
import glob

for img in glob.glob("C:/Users/h4tec/Desktop/redHole.jpg"):
 
    image= Image.open(img)
    model = YOLO('best.pt')
    results = model.predict(image, save =True)
    result = results[0]
    print("printing lenght of boxes")
    print(len(result.boxes))
    if len(result.boxes) >= 1:
        
        box = result.boxes[0]
        
        
        print("Coordinates:",box.xyxy[0])
        
        cords = box.xyxy[0].tolist()
        
        print("Coordinates:after to list()", cords)
        cords = [round(x) for x in cords]
        print("Coordinates:", cords)
        crop_image = image.crop(cords)
        crop_image.save('crop_image.jpg')
        
        crop_image.show()
        
        
        org_image = np.array(crop_image)
        # Convert RGB to BGR
        org_image = org_image[:, :, ::-1].copy()
        
        
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(org_image, cv2.COLOR_BGR2HSV)
        #hsv = cv2.resize(hsv, (200,800))
        print("PIL to hsv format converted")
        
        detect_black(hsv)
        
        detect_red(hsv)
        
        if blackHoles== 0 and redHoles == 0:
            print("unknown pole detected")
        if blackHoles > redHoles:
            print("PC pole detected")
        if redHoles> blackHoles:
            print("RC pole detected")
        if blackHoles != 0 and redHoles !=0:
            if redHoles== blackHoles:
                print("RC pole")

       
    else:
        
        print("no pole detected")
        

    
