# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:08:10 2024

@author: h4tec
"""

import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Function to correct the image tilt based on a known angle (OpenCV rotation)
def correct_image_with_angle(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    # Use the angle to create the rotation matrix (rotate clockwise by default with negative angle)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    # Apply the affine transformation (rotation)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image

# Load the YOLO segmentation model
model = YOLO("C:/Users/h4tec/Downloads/cable_sagging.pt")

# Load the input image
image_path = "C:/Users/h4tec/Desktop/sagging/21 (2).jpg"
image = cv2.imread(image_path)

# Perform detection and segmentation using YOLOv8-seg
results = model(image, conf=0.1, iou=0.6)

# Variables to store the cable and supline with maximum confidence
best_cable_box = None
best_supline_mask = None
max_cable_conf = -1
max_supline_conf = -1

# Iterate through detected objects and find the cable and supline with the highest confidence
for i, cls in enumerate(results[0].boxes.cls):
    class_name = results[0].names[int(cls)]
    conf = results[0].boxes.conf[i].cpu().numpy()  # Extract confidence score for each detection
    
    if class_name == "supline" and conf > max_supline_conf:  # Find the supline with maximum confidence
        best_supline_mask = results[0].masks.data[i].cpu().numpy()  # Store the supline mask
        max_supline_conf = conf
    
    if class_name == "cable" and conf > max_cable_conf:  # Find the cable with maximum confidence
        best_cable_box = results[0].boxes.xyxy[i].cpu().numpy()  # Store the bounding box for the cable
        max_cable_conf = conf

# If the best supline mask is found, proceed with line fitting and angle calculation
if best_supline_mask is not None:
    
    # Convert supline mask to binary and resize to the original image dimensions
    best_supline_mask = (best_supline_mask > 0.5).astype(np.uint8)
    supline_mask_resized = cv2.resize(best_supline_mask, (image.shape[1], image.shape[0]))

    # Get the coordinates of the supline from the mask
    supline_points = np.column_stack(np.where(supline_mask_resized > 0))

    # Fit a straight line to the detected supline points (first-degree polynomial)
    x_vals_supline = supline_points[:, 1]  # X coordinates of supline
    y_vals_supline = supline_points[:, 0]  # Y coordinates of supline

    supline_poly_fit = np.polyfit(x_vals_supline, y_vals_supline, 1)
    supline_poly_func = np.poly1d(supline_poly_fit)

    # The supline slope
    supline_slope = supline_poly_fit[0]

    # Approximate the pole as a vertical line (undefined slope, near x = x_pole)
    x_pole = np.mean(x_vals_supline)  # Use mean x-coordinate of supline as an approximation for the pole's location
    pole_slope = float('inf')  # Vertical slope is considered infinite

    # Calculate the angle between the supline and the pole
    supline_angle = np.arctan(supline_slope)  # Angle of the supline in radians
    pole_angle = np.pi / 2  # Pole is vertical, 90 degrees or pi/2 radians

    # Calculate the angle difference between the supline and pole
    angle_between_supline_and_pole = np.degrees(pole_angle - supline_angle)  # Keep direction of angle

    # Print the angle between supline and pole
    print(f"Angle between the supline (max confidence) and pole: {angle_between_supline_and_pole:.2f} degrees")

    # If the angle is significant (say more than 1 degree), apply correction
    if abs(angle_between_supline_and_pole) > 1:
        image = correct_image_with_angle(image, angle_between_supline_and_pole)
        print(f"Applied rotation to correct tilt by {angle_between_supline_and_pole:.2f} degrees.")

    # Plot the corrected image and the detected supline
    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Generate x values for plotting the supline
    x_curve = np.linspace(min(x_vals_supline), max(x_vals_supline), 100)
    y_curve = supline_poly_func(x_curve)

    # Plot the supline
    plt.plot(x_curve, y_curve, color='red', label=f'Supline (max confidence)')
    plt.legend()
    plt.title(f'Supline (Max Confidence) Detection and Angle Calculation')
    plt.axis('off')
    plt.show()

else:
    print("Supline not found.")

# If the best cable bounding box is found, calculate width and height
if best_cable_box is not None:
    x_min, y_min, x_max, y_max = best_cable_box

    # Calculate width and height of the bounding box
    cable_width = x_max - x_min
    cable_height = y_max - y_min

    print(f"Cable (max confidence): Bounding box width: {cable_width:.2f} px")
    print(f"Cable (max confidence): Bounding box height: {cable_height:.2f} px")

    # Plot the image with the bounding box
    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Draw the bounding box on the image
    plt.gca().add_patch(plt.Rectangle((x_min, y_min), cable_width, cable_height,
                                      fill=False, edgecolor='blue', linewidth=2, label='Cable Bounding Box (Max Confidence)'))

    plt.legend()
    plt.title(f'Cable Bounding Box (Max Confidence)')
    plt.axis('off')
    plt.show()

else:
    print("Cable not found.")
