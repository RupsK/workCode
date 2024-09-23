# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:11:32 2024

@author: h4tec
"""

import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Function to correct the image tilt based on a camera angle (if provided)
def correct_image_with_camera_angle(image, camera_angle):
    if camera_angle is None:
        return image  # No rotation if camera angle is None
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    # Use the camera angle to create the rotation matrix
    M = cv2.getRotationMatrix2D(center, -camera_angle, 1.0)  # Negative for clockwise rotation
    # Apply the affine transformation (rotation)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image

# Load the YOLO segmentation model
model = YOLO("C:/Users/h4tec/Downloads/cable_sagging.pt")

# Load the input image
image_path = "C:/Users/h4tec/Desktop/sagging/21 (2).jpg"
image = cv2.imread(image_path)

# Define the camera angle (pass None if no rotation is necessary)
camera_angle = None  # Replace with the actual camera angle if needed

# Correct the image using the camera angle (if provided)
image = correct_image_with_camera_angle(image, camera_angle)

# Perform detection and segmentation using YOLOv8-seg
results = model(image, conf=0.5, iou=0.6)

# Initialize lists to store detected cables and suplines
cable_boxes = []
supline_masks = []

# Define confidence threshold for selecting relevant detections
confidence_threshold = 0.5  # You can adjust this value as needed

# Iterate through detected objects and store those that match the conditions
for i, cls in enumerate(results[0].boxes.cls):
    class_name = results[0].names[int(cls)]
    conf = results[0].boxes.conf[i].cpu().numpy()  # Extract confidence score for each detection

    if class_name == "supline" and conf >= confidence_threshold:
        # Store the supline mask if it meets the confidence threshold
        supline_mask = results[0].masks.data[i].cpu().numpy()
        supline_masks.append(supline_mask)

    if class_name == "cable" and conf >= confidence_threshold:
        # Store the bounding box if it meets the confidence threshold
        cable_box = results[0].boxes.xyxy[i].cpu().numpy()
        cable_boxes.append(cable_box)

# Process each detected supline mask
for idx, supline_mask in enumerate(supline_masks):
    # Convert supline mask to binary and resize to the original image dimensions
    supline_mask = (supline_mask > 0.5).astype(np.uint8)
    supline_mask_resized = cv2.resize(supline_mask, (image.shape[1], image.shape[0]))

    # Get the coordinates of the supline from the mask
    supline_points = np.column_stack(np.where(supline_mask_resized > 0))

    # Fit a straight line to the detected supline points (first-degree polynomial)
    x_vals_supline = supline_points[:, 1]  # X coordinates of supline
    y_vals_supline = supline_points[:, 0]  # Y coordinates of supline

    supline_poly_fit = np.polyfit(x_vals_supline, y_vals_supline, 1)
    supline_poly_func = np.poly1d(supline_poly_fit)

    # Calculate the supline slope and angle
    supline_slope = supline_poly_fit[0]
    supline_angle = np.arctan(supline_slope)
    pole_angle = np.pi / 2  # Pole is considered vertical (90 degrees)

    # Calculate the absolute angle difference between the supline and pole
    angle_between_supline_and_pole = np.abs(np.degrees(pole_angle - supline_angle))
    angle_between_supline_and_pole = 180 - angle_between_supline_and_pole
    # Print and plot each detected supline
    print(f"Supline {idx + 1}: Angle between supline and pole: {angle_between_supline_and_pole:.2f} degrees")

    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    x_curve = np.linspace(min(x_vals_supline), max(x_vals_supline), 100)
    y_curve = supline_poly_func(x_curve)
    plt.plot(x_curve, y_curve, color='red', label=f'Supline {idx + 1}')
    plt.legend()
    plt.title(f'Supline {idx + 1} Detection and Angle Calculation')
    plt.axis('off')
    plt.show()

# Process each detected cable bounding box
for idx, cable_box in enumerate(cable_boxes):
    x_min, y_min, x_max, y_max = cable_box
    cable_width = x_max - x_min
    cable_height = y_max - y_min

    print(f"Cable {idx + 1}: Bounding box width: {cable_width:.2f} px, height: {cable_height:.2f} px")

    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.gca().add_patch(plt.Rectangle((x_min, y_min), cable_width, cable_height,
                                      fill=False, edgecolor='blue', linewidth=2, label=f'Cable {idx + 1}'))
    plt.legend()
    plt.title(f'Cable {idx + 1} Bounding Box')
    plt.axis('off')
    plt.show()
