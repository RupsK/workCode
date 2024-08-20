# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:50:56 2024

@author: h4tec
"""

from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import os
# Set the environment variable to avoid OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Open the new image
new_image_path = "C:/Users/h4tec/Desktop/RCtestPole/6.jpg"

new_image = Image.open(new_image_path)

# Display the original new image
plt.imshow(new_image)
plt.title("Original New Image")
plt.axis('off')
plt.show()

# Enhance the new image
new_enhanced_image = new_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

# Increase contrast
new_enhancer = ImageEnhance.Contrast(new_enhanced_image)
new_enhanced_image = new_enhancer.enhance(2)

# Display the enhanced new image
plt.imshow(new_enhanced_image)
plt.title("Enhanced New Image")
plt.axis('off')
plt.show()

# Convert the enhanced image to a numpy array
image_array = np.array(new_enhanced_image)

# Define a function to check for the presence of red pixels
def check_for_red_pixels(image_array):
    red_threshold =200  # Adjusting the threshold to detect more subtle reds
    red_pixels = np.where(
        (image_array[:, :, 0] > red_threshold) &  # Red channel
        (image_array[:, :, 1] < red_threshold) &  # Green channel
        (image_array[:, :, 2] < red_threshold)    # Blue channel
    )
    return red_pixels

# Check for any red pixels in the image
red_pixels = check_for_red_pixels(image_array)
contains_red = len(red_pixels[0]) > 0
print("Contains red pixels:", contains_red)
