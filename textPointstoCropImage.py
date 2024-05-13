# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:13:40 2024

@author: h4tec
"""
import json
from PIL import Image

# Path to the image you want to crop
image_file_path = 'C:/Users/h4tec/Desktop/test/images/0 (54).jpg'
json_file_path = 'C:/Users/h4tec/Desktop/test/images/0 (54).json'
# Load the coordinates from the JSON file
with open(json_file_path, 'r') as json_file:
    coords = json.load(json_file)

# Extract the coordinates
x = coords['x']
y = coords['y']
width = coords['width']
height = coords['height']

# Open the image
image = Image.open(image_file_path)

# Crop the image using the coordinates
cropped_image = image.crop((x, y, x + width, y + height))

# Show the cropped image
cropped_image.show()

# Optionally, save the cropped image to a file
# cropped_image.save('path/to/save/cropped_image.jpg')


