# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:37:16 2024

@author: h4tec
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read your image (replace 'path/to/your/image.jpg' with the actual path)
image_path = 'C:/Rupali Shinde/shape2.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect circles using Hough Circle Transform
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 300, param1=50, param2=5, minRadius=1, maxRadius=10)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0]:
        center = (circle[0], circle[1])
        radius = circle[2]
        # Check if the circle is black/dark (you can add more conditions here)
        if gray[circle[1], circle[0]] > 100:
            cv2.circle(image, center, radius, (0, 0, 255), 3)

# Save the modified image
cv2.imwrite('modif_' + image_path, image)
plt.figure(figsize=(8, 6))
plt.subplot(121)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")