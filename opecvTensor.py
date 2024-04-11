# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:09:12 2024

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

import matplotlib.pyplot as plt

import cv2
import torch
from torchvision import transforms

# Load your image
image_path = 'path/to/your/image.jpg'
image = cv2.imread('15 (2).jpg')

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize the image
image = cv2.resize(image, (800, 800))

# Convert to tensor
image_tensor = transforms.ToTensor()(image)

# Add batch dimension and convert to float
image_tensor = image_tensor.unsqueeze(0).float()

# Load your model (make sure it's in eval mode)

model = YOLO('best.pt')
results = model.predict(image_tensor, save =True)

result = results[0]

len(result.boxes)
box = result.boxes[0]


print("Coordinates:",box.xyxy[0])

cords = box.xyxy[0].tolist()

print("Coordinates:after to list()", cords)
cords = [round(x) for x in cords]
print("Coordinates:", cords)

crop_image = image.crop(cords)
crop_image.save('crop_image.jpg')

crop_image.show()


# Perform inference
with torch.no_grad():
    preds = model(image_tensor)