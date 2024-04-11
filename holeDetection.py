# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:26:11 2024

@author: h4tec
"""

import cv2
import numpy as np

# Read the image
image = cv2.imread('C:/Rupali Shinde/JejuSet-3/0 (2).jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection using Canny
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a counter for holes
hole_count = 0

# Loop over the contours
for contour in contours:
    # Approximate the contour to a polygon
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    # If the contour has fewer than 15 points, it's likely a hole
    if len(approx) < 15:
        hole_count += 1

# Display the total number of holes
print("Number of holes:", hole_count)

# Display the original image with contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()