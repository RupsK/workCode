# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:13:26 2024

@author: h4tec
"""

import cv2
import numpy as np
import json

# Load the image
image_file = 'C:/Users/h4tec/Desktop/PC/32.jpg'
image = cv2.imread(image_file)
image_name = image_file.split('/')[-1].split('.')[0]

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("before", gray_image)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

#ret, thresh = cv2.threshold(blurred_image, 50, 255, 0)

# Apply image sharpening using the kernel
kernel = np.array([[0,-1,0],
                   [-1, 4,-1],
                   [0,-1,0]])
 
kernel1 = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])

sharpened_image = cv2.filter2D(gray_image, -1, kernel1)

# Apply Canny edge detection
edges = cv2.Canny(gray_image, 150, 700)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through all contours and filter for cable contours
for contour in contours:
    # Calculate the area of the contour
    area = cv2.contourArea(contour)
    
    # Filter contours based on area (you may need to adjust this threshold)
    if area > 300:
        # Draw contour on the original image
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)



# Create a mask for the cable region
mask = np.zeros_like(gray_image)
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)




# Apply mask to original image
segmented_cable = cv2.bitwise_and(image, image, mask=mask)


# Extract pixel coordinates
coordinates = []

# Get image height
height = mask.shape[0]

# Iterate through all pixels in the mask
for y in range(height):
    for x in range(mask.shape[1]):
        # Check if the pixel is in the upper 75% of the image
        if mask[y, x] == 255 and y < height * 0.55:
            coordinates.append((x, y))
            
json_file = image_name +"_coordinates.json"
# Save coordinates to a JSON file
with open('coordinates.json', 'w') as file:
    json.dump(coordinates, file)


filename = 'C:/Users/h4tec/Desktop/PC/segmented_image.jpg'
# Display the segmented cable
cv2.imshow('Segmented Cable', image)
cv2.imwrite(filename, image)
cv2.waitKey(0)
cv2.destroyAllWindows()