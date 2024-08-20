import os
import cv2
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ExifTags
import glob
from ultralytics import YOLO
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


image_dir = "C:/Users/h4tec/Desktop/images_PCPole/images_522862672_20240604132535_4"
json_file = "C:/Users/h4tec/Desktop/images_PCPole/images_522862672_20240604132535_4/output.json" 
"""
# Set the environment variable to avoid OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Ensure the script receives the necessary arguments
if len(sys.argv) != 3:
    print("Usage: python script_name.py <image_directory> <json_file>")
    sys.exit(1)

# Directory paths from arguments
image_dir = sys.argv[1]
json_file = sys.argv[2]
"""
output_dir = "outputTest_522862672"
views_dir = os.path.join(output_dir, "views")
cropped_images_dir = os.path.join(output_dir, "cropped")
cropped_views_dir = os.path.join(views_dir, "cropped_view")

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(views_dir, exist_ok=True)
os.makedirs(cropped_images_dir, exist_ok=True)
os.makedirs(cropped_views_dir, exist_ok=True)

# Read the JSON data
with open(json_file) as f:
    data = json.load(f)

# Check the structure of the JSON data
if not isinstance(data, list):
    logging.error("JSON data is not a list. Please check the input file format.")
    raise TypeError("JSON data is not a list. Please check the input file format.")

# Initialize YOLO model
model = YOLO('C:/Users/h4tec/Downloads/yolo8DB2Imgsz2000.pt')

def get_pole_info(data, image_id):
    center_list = []
    bounding_box = None

    logging.info(f"Processing image ID: {image_id}")

    for item in data:
        if not isinstance(item, dict):
            logging.warning(f"Unexpected item format: {item}")
            continue

        if item['image_id'] == image_id:
            for detection in item['detection_list']:
                if detection['object_type'] == 'pole' and detection['label'] == 'concrete_commu_columns':
                    if 'target_masks' in detection:
                        if bounding_box is None:
                            bounding_box = detection['vertex']
                        for coordinate_pair in detection['target_masks']:
                            if isinstance(coordinate_pair, list) and len(coordinate_pair) == 2:
                                point1, point2 = coordinate_pair
                                if isinstance(point1, list) and isinstance(point2, list) and len(point1) == 2 and len(point2) == 2:
                                    x1, y1 = point1
                                    x2, y2 = point2
                                    center_x = (x1 + x2) / 2
                                    center_y = (y1 + y2) / 2
                                    center_list.append((center_x, center_y))
                                else:
                                    logging.warning(f"Invalid coordinate pair: {coordinate_pair}")
                            else:
                                logging.warning(f"Invalid coordinate pair format: {coordinate_pair}")
                        
    return center_list, bounding_box

def lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    def direction(xi, yi, xj, yj, xk, yk):
        return (xk - xi) * (yj - yi) - (xj - xi) * (yk - yi)

    d1 = direction(x1, y1, x2, y2, x3, y3)
    d2 = direction(x1, y1, x2, y2, x4, y4)
    d3 = direction(x3, y3, x4, y4, x1, y1)
    d4 = direction(x3, y3, x4, y4, x2, y2)

    return (((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and 
            ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)))

def get_image_usercomment(image_file_path):
    try:
        with Image.open(image_file_path) as img:
            exif_data = img._getexif()
            if exif_data is not None:
                for tag_id, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag_name == 'UserComment':
                        return value
        return None
    except IOError:
        return None

def get_angle_info(image_file_path):
    usercomment = get_image_usercomment(image_file_path)
    if usercomment is None:
        logging.warning(f"No EXIF UserComment found in image: {image_file_path}")
        return None

    image_data = str(usercomment).split(',')
    if len(image_data) < 27:
        logging.warning(f"Image contains unavailable information: {image_data}")
        return None

    try:
        angle = float(image_data[26])
        return angle
    except ValueError:
        logging.warning(f"Invalid angle value in image: {image_file_path}")
        return None

def get_left_right_view(angle):
    if angle is None:
        logging.warning("No valid angle information available.")
        return None

    img_path = image_dir

    # Angle information for the front view image
    front_view_angle = angle

    # Define angle targets for left and right views
    left_view_target = (front_view_angle + 90) % 360
    right_view_target = (front_view_angle - 90) % 360

    # Initialize variables to keep track of best left and right view images
    best_left_image = None
    best_right_image = None
    min_left_angle_diff = float('inf')
    min_right_angle_diff = float('inf')

    # Loop through all images in the folder
    for img in glob.glob(os.path.join(img_path, "*.jpg")):
        image_path = os.path.abspath(img)
        angle = get_angle_info(image_path)
        
        if angle is not None:
            angle_difference = abs(angle - front_view_angle)
            if angle_difference == 0:
                continue  # Skip the front view image itself
            
            # Normalize angle within 0 to 360 degrees
            angle = angle % 360
            
            # Calculate angular differences to the target left and right view angles
            left_angle_diff = min(abs(angle - left_view_target), 360 - abs(angle - left_view_target))
            right_angle_diff = min(abs(angle - right_view_target), 360 - abs(angle - right_view_target))

            # Check for left view
            if left_angle_diff < min_left_angle_diff:
                min_left_angle_diff = left_angle_diff
                best_left_image = img
            
            # Check for right view
            if right_angle_diff < min_right_angle_diff:
                min_right_angle_diff = right_angle_diff
                best_right_image = img

    if best_left_image:
        logging.info(f"Best Left View Image: {best_left_image} (Angle difference: {min_left_angle_diff} degrees)")
    else:
        logging.warning("No left view image found.")

    if best_right_image:
        logging.info(f"Best Right View Image: {best_right_image} (Angle difference: {min_right_angle_diff} degrees)")
    else:
        logging.warning("No right view image found.")

    return best_left_image, best_right_image

def enhance_image(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

def increase_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def detect_red_pixels(image):
    lower_red = np.array([0, 0, 100])
    upper_red = np.array([100, 100, 255])
    mask = cv2.inRange(image, lower_red, upper_red)
    red_pixels = cv2.countNonZero(mask)
    return red_pixels > 0

def process_image(image_file):
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    image_id = image_file

    # Get coordinates from JSON file
    center_list, bounding_box = get_pole_info(data, image_id)

    if not center_list or not bounding_box:
        logging.warning(f"Image: {image_file} does not have center coordinates or bounding box data in the JSON file.")
        return 0, [], [], None

    # Convert floating point coordinates to integers
    int_coordinates = [(int(x), int(y)) for x, y in center_list]

    # Save the center coordinates to a CSV file
    output_csv_file = os.path.join(output_dir, f'{os.path.splitext(image_file)[0]}_center_coordinates.csv')
    with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Center_X", "Center_Y"])
        writer.writerows(center_list)

    # Extract the bounding box for cropping
    x_min = min([point[0] for point in bounding_box]) - 15
    y_min = min([point[1] for point in bounding_box]) - 15
    x_max = max([point[0] for point in bounding_box]) + 15
    y_max = max([point[1] for point in bounding_box]) + 15

    # Crop the image using the bounding box
    cropped_image = image[y_min:y_max, x_min:x_max]
    
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    cropped_image = cv2.filter2D(src=cropped_image, ddepth=-1, kernel=kernel)
    cropped_image_file = os.path.join(cropped_images_dir, f'{os.path.splitext(image_file)[0]}_cropped.jpg')
    cv2.imwrite(cropped_image_file, cropped_image)

    # Adjust center coordinates relative to the cropped image
    adjusted_center_list = [(x - x_min, y - y_min) for (x, y) in center_list]

    # Save adjusted center coordinates to a JSON file
    adjusted_json_file = os.path.join(output_dir, f'{os.path.splitext(image_file)[0]}_adjusted_center_coordinates.json')
    with open(adjusted_json_file, 'w') as json_file:
        json.dump({"adjusted_center_coordinates": adjusted_center_list}, json_file, indent=4)

    # Use YOLO model to detect holes and stepbolts
    detection_results = model.predict(cropped_image_file, conf=0.01, iou=0.3)

    # Extract detection results
    holes = []
    stepbolts = []
    for result in detection_results:
        for box in result.boxes:
            if int(box.cls) == 0:  # Assuming '0' is the class ID for holes
                holes.append({
                    "x_min": box.xyxy[0][0].item(),
                    "y_min": box.xyxy[0][1].item(),
                    "x_max": box.xyxy[0][2].item(),
                    "y_max": box.xyxy[0][3].item()
                })
            elif int(box.cls) == 1:  # Assuming '1' is the class ID for stepbolts
                stepbolts.append({
                    "x_min": box.xyxy[0][0].item(),
                    "y_min": box.xyxy[0][1].item(),
                    "x_max": box.xyxy[0][2].item(),
                    "y_max": box.xyxy[0][3].item()
                })

    # Save detection results to a JSON file
    predicted_json_file = os.path.join(output_dir, f'{os.path.splitext(image_file)[0]}_detections.json')
    with open(predicted_json_file, 'w', encoding='utf-8') as f:
        json.dump({"holes": holes, "stepbolts": stepbolts}, f, ensure_ascii=False, indent=4)

    image = cv2.imread(cropped_image_file)
    center_coords = adjusted_center_list

    for i in range(len(center_coords) - 1):
        x1, y1 = center_coords[i]
        x2, y2 = center_coords[i + 1]
        cv2.circle(image, (int(x1), int(y1)), 5, (0, 255, 0), -1)
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    touch_count = 0

    for hole in holes:
        x_min = int(hole['x_min'])
        y_min = int(hole['y_min'])
        x_max = int(hole['x_max'])
        y_max = int(hole['y_max'])
        top_left = (x_min, y_min)
        bottom_right = (x_max, y_max)

        # Check for intersections
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        for i in range(len(center_coords) - 1):
            x1, y1 = center_coords[i]
            x2, y2 = center_coords[i + 1]

            if lines_intersect(x1, y1, x2, y2, x_min, y_min, x_max, y_min) or \
               lines_intersect(x1, y1, x2, y2, x_min, y_min, x_min, y_max) or \
               lines_intersect(x1, y1, x2, y2, x_max, y_min, x_max, y_max) or \
               lines_intersect(x1, y1, x2, y2, x_min, y_max, x_max, y_max):
                touch_count += 1

    output_checked_image = os.path.join(output_dir, f'{os.path.splitext(image_file)[0]}_checked_image_with_bboxes.jpg')
    cv2.imwrite(output_checked_image, image)

    logging.info(f"Image: {image_file}, Number of intersections found on the whole pole: {touch_count}")

    return touch_count, adjusted_center_list, holes, stepbolts, cropped_image_file

# Initialize variables
stepbolt_detected = False
hole_detected = False
detection_data = []
max_touch_count = 0
image_with_max_touch = None

# Process each image in the folder
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    touch_count, adjusted_center_list, holes, stepbolts, cropped_image_file = process_image(image_file)
    
    if touch_count > max_touch_count:
        max_touch_count = touch_count
        image_with_max_touch = image_file

    if cropped_image_file and os.path.exists(cropped_image_file):
        plt.figure(figsize=(10, 20))
        plt.imshow(cv2.cvtColor(cv2.imread(cropped_image_file), cv2.COLOR_BGR2RGB))
        plt.title(f'Checked Image with Bounding Boxes for {image_file}')
        plt.show()

# Save detection results in the specified format
detection_results = {
    "front_view_image": None,
    "left_view_image": None,
    "right_view_image": None
}

def create_detection_info(image_file, holes, stepbolts):
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    detection_info = {
        "image_id": image_file,
        "image_size": [width, height],
        "match_decision": True,
        "model_name": "yolo8DB2Imgsz2000.pt",
        "detection_number": len(holes) + len(stepbolts),
        "detection_list": []
    }

    for index, hole in enumerate(holes):
        detection_info["detection_list"].append({
            "order": index,
            "object_type": "object",
            "label": "hole",
            "scores": 0.9999,  # Dummy score for illustration
            "length": int(hole["x_max"] - hole["x_min"]),
            "area": int((hole["x_max"] - hole["x_min"]) * (hole["y_max"] - hole["y_min"])),
            "center_masks": [
                int((hole["x_min"] + hole["x_max"]) / 2),
                int((hole["y_min"] + hole["y_max"]) / 2)
            ],
            "target_masks": [
                [
                    [hole["x_min"], hole["y_min"]],
                    [hole["x_max"], hole["y_max"]]
                ]
            ],
            "vertex": [
                [hole["x_min"], hole["y_min"]],
                [hole["x_max"], hole["y_min"]],
                [hole["x_min"], hole["y_max"]],
                [hole["x_max"], hole["y_max"]]
            ],
            "attach_list": []
        })

    for index, stepbolt in enumerate(stepbolts):
        detection_info["detection_list"].append({
            "order": len(holes) + index,
            "object_type": "object",
            "label": "stepbolt",
            "scores": 0.9999,  # Dummy score for illustration
            "length": int(stepbolt["x_max"] - stepbolt["x_min"]),
            "area": int((stepbolt["x_max"] - stepbolt["x_min"]) * (stepbolt["y_max"] - stepbolt["y_min"])),
            "center_masks": [
                int((stepbolt["x_min"] + stepbolt["x_max"]) / 2),
                int((stepbolt["y_min"] + stepbolt["y_max"]) / 2)
            ],
            "target_masks": [
                [
                    [stepbolt["x_min"], stepbolt["y_min"]],
                    [stepbolt["x_max"], stepbolt["y_max"]]
                ]
            ],
            "vertex": [
                [stepbolt["x_min"], stepbolt["y_min"]],
                [stepbolt["x_max"], stepbolt["y_min"]],
                [stepbolt["x_min"], stepbolt["y_max"]],
                [stepbolt["x_max"], stepbolt["y_max"]]
            ],
            "attach_list": []
        })
    
    return detection_info

if image_with_max_touch:
    detection_results["front_view_image"] = create_detection_info(image_with_max_touch, holes, stepbolts)
else:
    logging.warning("No suitable front view image found based on holes detection.")

# Get the angle of the front view image
front_view_angle = get_angle_info(os.path.join(image_dir, image_with_max_touch))
logging.info(f"Front view angle: {front_view_angle}")

# Get the left and right view images based on the front view angle
left_view_image, right_view_image = get_left_right_view(front_view_angle)

if left_view_image:
    detection_results["left_view_image"] = create_detection_info(left_view_image, holes, stepbolts)

if right_view_image:
    detection_results["right_view_image"] = create_detection_info(right_view_image, holes, stepbolts)

# Save the selected views to the "views" folder
front_view_path = os.path.join(views_dir, os.path.basename(image_with_max_touch))
left_view_path = os.path.join(views_dir, os.path.basename(left_view_image))
right_view_path = os.path.join(views_dir, os.path.basename(right_view_image))

os.rename(os.path.join(image_dir, image_with_max_touch), front_view_path)
os.rename(os.path.join(image_dir, left_view_image), left_view_path)
os.rename(os.path.join(image_dir, right_view_image), right_view_path)

# Save the cropped images of the selected views to the "views/cropped_view" folder
for view_image in [image_with_max_touch, left_view_image, right_view_image]:
    view_image_id = os.path.splitext(os.path.basename(view_image))[0]
    cropped_image_path = os.path.join(cropped_images_dir, f'{view_image_id}_cropped.jpg')
    if os.path.exists(cropped_image_path):
        os.rename(cropped_image_path, os.path.join(cropped_views_dir, f'{view_image_id}_cropped.jpg'))

logging.info(f"Cropped images for selected views saved to {cropped_views_dir}")

# Save detection results in JSON format
detection_results_file = os.path.join(output_dir, 'detection_results.json')
with open(detection_results_file, 'w', encoding='utf-8') as f:
    json.dump(detection_results, f, ensure_ascii=False, indent=4)

logging.info(f"Detection results saved to {detection_results_file}")
