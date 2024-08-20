import cv2
import os
import json
import numpy as np
import sys
from ultralytics import YOLO
from PIL import Image, ImageFilter, ImageEnhance

# Set the environment variable to avoid OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load your YOLO model
model_path = "C:/Users/h4tec/Downloads/yolo8DB2Imgsz2000.pt"
model = YOLO(model_path)

# Get the input directory from command line arguments
if len(sys.argv) > 1:
    image_folder_path = sys.argv[1]
else:
    print("No input directory provided.")
    sys.exit(1)

# Create output directories if they don't exist
output_folder = 'HOLEcropped'
views_folder = 'output/views'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(views_folder):
    os.makedirs(views_folder)

# Function to check for red pixels in the middle 50% of the pole
def check_for_red_pixels(image_array):
    height, width, _ = image_array.shape
    start_y = int(height * 0.25)
    end_y = int(height * 0.75)
    
    red_threshold = 400  # Adjusting the threshold to detect more subtle reds
    red_pixels = np.where(
        (image_array[start_y:end_y, :, 0] > red_threshold) &  # Red channel
        (image_array[start_y:end_y, :, 1] < red_threshold) &  # Green channel
        (image_array[start_y:end_y, :, 2] < red_threshold)    # Blue channel
    )
    return red_pixels

# Load existing detection results if the file exists
json_output_path = os.path.join(views_folder, "detection_infoCategory.json")
if os.path.exists(json_output_path):
    with open(json_output_path, 'r', encoding='utf-8') as f:
        result_data = json.load(f)
else:
    result_data = {}

# Ensure 'details' key exists
if "details" not in result_data:
    result_data["details"] = []

# Ensure 'RC_pole_detected' and 'RC_enforced_pole_detected' keys exist
if "RC_pole_detected" not in result_data:
    result_data["RC_pole_detected"] = False
if "RC_enforced_pole_detected" not in result_data:
    result_data["RC_enforced_pole_detected"] = False

# Loop through each image in the folder
for image_filename in os.listdir(image_folder_path):
    if image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder_path, image_filename)
        image = cv2.imread(image_path)

        # Perform prediction
        results = model(image_path, conf=0.01, iou=0.30)

        # Convert results to a dictionary format similar to Roboflow output
        detections = []
        for result in results:
            for bbox in result.boxes:
                x_center, y_center, width, height = bbox.xywh[0].cpu().numpy()
                detections.append({
                    'class': result.names[int(bbox.cls)],
                    'x': float(x_center),
                    'y': float(y_center),
                    'width': float(width),
                    'height': float(height)
                })

        test = {'predictions': detections}

        # Save the detection results to a JSON file
        json_file_path = os.path.join(output_folder, f'{os.path.splitext(image_filename)[0]}_detections.json')
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(test, f, ensure_ascii=False, indent=4)

        # Load the JSON file
        with open(json_file_path, 'r') as f:
            detection_data = json.load(f)

        count = 0
        cropped_images_paths = []
        image_height = image.shape[0]
        lower_25_percent_threshold = image_height * 0.75
        hole_in_lower_25_percent = False

        for detection in detection_data['predictions']:
            if "hole" in detection['class']:  # Filter only the detections with "hole" in the class name
                x_center = detection['x']
                y_center = detection['y']
                width = detection['width']
                height = detection['height']
                
                x_min = int(x_center - (width / 2))
                y_min = int(y_center - (height / 2))
                x_max = int(x_center + (width / 2))
                y_max = int(y_center + (height / 2))

                # Check if the hole is in the lower 25% of the pole
                if y_center > lower_25_percent_threshold:
                    hole_in_lower_25_percent = True

                cropped_image = image[y_min:y_max, x_min:x_max]
                
                scale_factor = 4

                # Get the dimensions of the original image
                original_height, original_width = cropped_image.shape[:2]
                
                # Calculate the dimensions of the upscaled image
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                
                # Resize the image using interpolation method
                upscaled_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

                output_path = os.path.join(output_folder, f'{os.path.splitext(image_filename)[0]}_hole_{count}.jpg')
                cv2.imwrite(output_path, upscaled_image)
                cropped_images_paths.append(output_path)
                count += 1

        # If a hole is detected in the lower 25% of the pole height, print the message and save the result
        if hole_in_lower_25_percent:
            print("This is an RC category pole")
            result_data["RC_pole_detected"] = True
            result_data["RC_enforced_pole_detected"] = False  # Ensure this is false if RC pole is detected
            result_data["details"].append({
                "image": image_filename,
                "type": "RC pole",
                "message": "Hole detected in the lower 25% of the pole height"
            })
            # Save the detection results to a JSON file
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=4)
            print(f"Detection results saved to {json_output_path}")
            sys.exit(0)  # Exit the program

        print(f"Cropped images saved in {output_folder}")

        for cropped_image_path in cropped_images_paths:
            new_image = Image.open(cropped_image_path)
            
            cropped_image = image[y_min:y_max, x_min:x_max]
            
            scale_factor = 4

            # Get the dimensions of the original image
            original_height, original_width = cropped_image.shape[:2]
            
            # Calculate the dimensions of the upscaled image
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            # Resize the image using interpolation method
            upscaled_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            # Enhance the new image
            new_enhanced_image = new_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

            # Increase contrast
            new_enhancer = ImageEnhance.Contrast(new_enhanced_image)
            new_enhanced_image = new_enhancer.enhance(2)

            # Convert the enhanced image to a numpy array
            image_array = np.array(new_enhanced_image)

            # Check for any red pixels in the middle 50% of the image
            red_pixels = check_for_red_pixels(image_array)
            contains_red = len(red_pixels[0]) > 0
            print(f"Contains red pixels in {os.path.basename(cropped_image_path)}: {contains_red}")

            # If red pixels are detected, print the message and save the result
            if contains_red:
                print("This is an RC enforced pole category")
                result_data["RC_enforced_pole_detected"] = True
                result_data["details"].append({
                    "image": image_filename,
                    "type": "RC enforced pole",
                    "message": "Red pixels detected in the middle 50% of the pole"
                })
                break

# Save the detection results to a JSON file
with open(json_output_path, 'w', encoding='utf-8') as f:
    json.dump(result_data, f, ensure_ascii=False, indent=4)
print(f"Detection results saved to {json_output_path}")

# Save the detection flag for RC enforced pole
rc_enforced_pole_detected = result_data["RC_enforced_pole_detected"]

# Save the flag to a temporary file
with open('rc_enforced_pole_flag.txt', 'w') as f:
    f.write(str(rc_enforced_pole_detected))
