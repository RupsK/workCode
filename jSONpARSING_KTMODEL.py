
from PIL import Image
import os

import cv2

import os
import glob

import json    

# Load the JSON data
print("trying to opne json")
with open('C:/Users/h4tec/Desktop/json_parsing/output.json', 'r') as file:
    print("json file is open")
    data = json.load(file)

# Function to segment and save image based on vertex coordinates
def segment_image(image_path, vertices, output_name):
    try:
        image = Image.open(image_path,'r')
        
        # Convert vertices into a tuple format that PIL expects (left, right, lower, lower)
        TL = min([vertex[0] for vertex in vertices])
        TP = min([vertex[1] for vertex in vertices])
        BL = max([vertex[0] for vertex in vertices])
        BR = max([vertex[1] for vertex in vertices])
        # Crop the image using the calculated vertices
        cropped_image = image.crop((TL, TP, BL, BR))
        # Save or display the cropped image
        cropped_image.save(output_name)
        print("image saved")
    except FileNotFoundError as e:
            print(e)
    
image_directory_path = 'C:/Users/h4tec/Desktop/json_parsing/'
# Loop through each detected object in the JSON and segment the image
for item in data:
    image_id = item['image_id']
    image_path= image_directory_path + image_id
    for detection in item['detection_list']:
        vertices = detection['vertex']
        output_name = f"{image_id[:-4]}_test{detection['order']}.jpg"  # Remove .jpg and add suffix
        segment_image(image_path, vertices, output_name)