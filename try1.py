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

import matplotlib.pyplot as plt

# Load the model (Assuming there's a YOLOv8 model available)
# This will vary depending on how YOLOv8 models are made available.
# Replace this with the specific way to load a YOLOv8 model.

path = 'C:/Users/h4tec/Downloads/best.pt'
"""
model = torch.hub.load('ultralytics/yolov8','custom', path_or_model= 'best.pt')"""

model = YOLO('best.pt')

# Load an image
img = Image.open("C:/Users/h4tec/Desktop/15.jpg") #C:/Users/h4tec/Desktop/15.jpg
# Perform inference
results = model.predict(img, save =True)


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
    
    crop_image = img.crop(cords)
    crop_image.save('crop_image.jpg')
    
    crop_image.show()
    
    
    org_image = np.array(crop_image)
    # Convert RGB to BGR
    org_image = org_image[:, :, ::-1].copy()
    
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(org_image, cv2.COLOR_BGR2HSV)
    hsv = cv2.resize(hsv, (200,800))
    print("PIL to hsv format converted")
    # Define the range of red color in HSV
    # These ranges may need to be adjusted for your specific image
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    lower_orange = np.array([11, 120, 70])
    upper_orange = np.array([20, 255, 255])
    
    # Threshold the HSV image to get only red colors
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask, mask_orange)
    #red_mask = mask1 + mask2
    
    # Apply edge detection to the red regions
    edges = cv2.Canny(mask, 100, 200)
    
    # Dilate the edges to make them more pronounced
    kernel = np.ones((3,3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Use the dilated edges as a mask to find blobs
    # Invert the mask for blob detection (blobs should be white on black background)
    inverted_mask = cv2.bitwise_not(dilated_edges)
    
    # Set up the SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 500  # Adjust as needed
    
    params.filterByCircularity = True
    params.minCircularity = 0.6
    
      # Since holes might not be perfectly circular
    params.filterByConvexity = False
   #params.minConvexity = 0.5
    
    params.filterByInertia = False
    
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(inverted_mask)
    print(len(keypoints))
    #number_of_keypoints = len(keypoints)
    if keypoints == ():
        print("It is PC pole")
    else: 
        result = cv2.drawKeypoints(hsv, keypoints, np.array([]), (0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Red-Edged Holes Detected", result)
        print("It is a RC pole")
    
    # Draw detected blobs as red circles on the original image
else:
    print("pole not detected")

cv2.waitKey(0)
cv2.destroyAllWindows()