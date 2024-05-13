# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:51:11 2024

@author: h4tec
"""

import cv2
import numpy as np

# Load your image
image = cv2.imread('C:/Users/h4tec/Desktop/json_parsing/6.jpg', cv2.IMREAD_GRAYSCALE)

#image = cv2.resize(image, (400,800))

# Apply Canny edge detector
edges = cv2.Canny(image, 0, 200)


# Invert the edge image
inverted_edges = cv2.bitwise_not(edges)


# Set up SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Filter by area.
params.filterByArea = True
params.minArea = 50  # Adjust as needed
#params.maxArea = 500

params.filterByCircularity = True
#params.minCircularity  = 0.5  # Since holes might not be perfectly circular

params.filterByConvexity = True
params.minConvexity = 0.5

params.filterByInertia = True
params.minInertiaRatio = 0.01


# Filter by Color (blobColor = 0 for black)
params.filterByColor = True
params.blobColor = 0


# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(inverted_edges)
print(keypoints)
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("Black Holes Detected", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()