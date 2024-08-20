



import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('C:/Users/h4tec/Downloads/yolo8DB2Imgsz2000.pt')

# Read the input image
image = cv2.imread("C:/Users/h4tec/Desktop/redHoleImages/16.jpg")

# Perform detection
results = model(image, conf=0.01, iou= 0.2)

# Extract bounding boxes, labels, and scores
boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
labels = results[0].boxes.cls.cpu().numpy()  # Class labels

# Define colors for the classes
color_map = {
    0: (0, 255, 0),   # Green for class 0
    1: (0, 0, 255)    # Red for class 1
}

# Draw bounding boxes on the image
for box, score, label in zip(boxes, scores, labels):
   
    x1, y1, x2, y2 = map(int, box)
    
    color = color_map.get(label, (255, 255, 255))  # Default to white if class not found
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    print(x1, y1, x2, y2, f'{label}')

# Convert BGR to RGB for display
annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Save the annotated image
new_image = "C:/Users/h4tec/Desktop/redHoleImages/new_image.jpg"
cv2.imwrite(new_image, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

# Display the image with detections
plt.imshow(annotated_image)
plt.axis('off')
plt.show()

"""

def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(image_slice)[0]
    return sv.Detections.from_ultralytics(result)


slicer = sv.InferenceSlicer(callback=callback)
sliced_detections = slicer(image=image)

label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoxAnnotator()
# You can also use sv.MaskAnnotator() for instance segmentation models
# mask_annotator = sv.MaskAnnotator()

annotated_image = box_annotator.annotate(
    scene=image.copy(), detections=sliced_detections)
# annotated_image = mask_annotator.annotate(
#    scene=image.copy(), detections=sliced_detections)

annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=sliced_detections)

sv.plot_image(annotated_image)



detections = slicer(image)
"""