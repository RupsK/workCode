import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
import os
import sys
import json

# Check if image directory is provided as a command-line argument
if len(sys.argv) > 1:
    image_folder_path = sys.argv[1]
else:
    print("Usage: python script.py <image_folder_path>")
    sys.exit(1)

# Load the YOLO models
wangum_model_path = "C:/Users/h4tec/Downloads/bestWangum.pt"
hole_model_path = "C:/Users/h4tec/Downloads/yolo8DB2Imgsz2000.pt"
wangum_model = YOLO(wangum_model_path)
hole_model = YOLO(hole_model_path)

# Define the threshold for the upper 25% of the pole
threshold_percentage = 0.25

# Define the tolerance level for x-coordinate alignment (in pixels)
tolerance = 10

# Function to draw detections and return their coordinates
def draw_detections(image, detections, color, class_name_map, threshold_y):
    coords = []
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        confidence = detection.conf[0]
        class_id = detection.cls[0]
        
        # Check if the detection is in the upper 25% of the pole
        if y1 < threshold_y:
            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Put the class name and confidence score
            label = f'{class_name_map[int(class_id)]}: {confidence:.2f}'
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Append the center coordinates of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            coords.append((center_x, center_y))
    return coords

# Path to the JSON file for storing detection results
json_output_path = os.path.join('output', 'views', 'detection_info.json')

# Load existing detection results if the file exists
if os.path.exists(json_output_path):
    with open(json_output_path, 'r', encoding='utf-8') as f:
        result_data = json.load(f)
else:
    result_data = {}

# Ensure that the JSON structure is correct
if 'details' not in result_data:
    result_data['details'] = []

# Process each image in the folder
for image_filename in os.listdir(image_folder_path):
    if image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder_path, image_filename)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform detection with both models
        wangum_results = wangum_model(image_rgb, conf=0.05, iou=0.30)
        hole_results = hole_model(image_rgb, conf=0.01, iou=0.30)

        # Get the height of the image
        image_height = image_rgb.shape[0]

        # Define the threshold for the upper 25% of the pole
        threshold_y = image_height * threshold_percentage

        # Draw wangum detections in green and get their coordinates
        wangum_coords = draw_detections(image_rgb, wangum_results[0].boxes, (0, 255, 0), wangum_model.names, threshold_y)

        # Draw hole detections in red and get their coordinates
        hole_coords = draw_detections(image_rgb, hole_results[0].boxes, (255, 0, 0), hole_model.names, threshold_y)

        # Check if there are any vertically aligned detections
        aligned = False
        for wangum in wangum_coords:
            for hole in hole_coords:
                # Check if x-coordinates are within the tolerance level
                if abs(wangum[0] - hole[0]) <= tolerance:
                    aligned = True
                    break
            if aligned:
                break

        # Check if there are two or more vertically aligned holes
        hole_aligned = False
        for i in range(len(hole_coords)):
            count_aligned_holes = 1
            for j in range(i + 1, len(hole_coords)):
                if abs(hole_coords[i][0] - hole_coords[j][0]) <= tolerance:
                    count_aligned_holes += 1
                if count_aligned_holes >= 2:
                    hole_aligned = True
                    break
            if hole_aligned:
                break

        # Convert back to BGR for saving with OpenCV
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Save the image with detections
        output_path = os.path.join(image_folder_path, f'predicted_{image_filename}')
        cv2.imwrite(output_path, image_bgr)

        # Display the image with detections
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()

        print(f"Predicted image saved as {output_path}")

        # Determine the result of alignment check
        if aligned or hole_aligned:
            detection_result = {
                "image": image_filename,
                "type": "RC pole",
                "message": "holes or wangum align top 25 %"
            }
            
            # Print RC pole detected
            print("RC pole detected")
        
            # Append detection results to the JSON structure
            result_data['details'].append(detection_result)
            
            # Update the detection flag for RC pole
            result_data["RC_pole_detected"] = True
        else:
            # Ensure the flag is present in the data
            result_data["RC_pole_detected"] = result_data.get("RC_pole_detected", False)

        # Save the detection flag for RC enforced pole
        rc_pole_detected = result_data["RC_pole_detected"]

        # Save the flag to a temporary file
        with open('rc_pole_flag.txt', 'w') as f:
            f.write(str(rc_pole_detected))
       

# Save the updated detection results to the JSON file
with open(json_output_path, 'w', encoding='utf-8') as f:
    json.dump(result_data, f, ensure_ascii=False, indent=4)

print(f"Detection results saved to {json_output_path}")
