import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# Load the YOLOv8 detection model
det_model = YOLO('C:/Users/h4tec/Downloads/bestTunnelDetection.pt')


def detect_rois(image_path, model):
    # Run inference
    results = model(image_path, conf = 0.1)
    
    # Get bounding boxes
    bboxes = results[0].boxes.xyxy.numpy()  # Adjusted to access the bounding boxes correctly
    return bboxes

# Load the YOLOv8 segmentation model
seg_model = YOLO('C:/Users/h4tec/Downloads/bestTunnelSegment.pt')


import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
image_path = "C:/Users/h4tec/Desktop/testReba/reba1.jpg"
def segment_rois(image, bboxes, seg_model):
    image_np = np.array(image)

    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox[:4])  # Ensure bbox coordinates are integers
        roi = image_np[y1:y2, x1:x2]
        roi_pil = Image.fromarray(roi)
        
        # Run segmentation model on the ROI
        roi_results = seg_model(roi_pil)
        
        # Extract masks from the results
        if len(roi_results) > 0 and hasattr(roi_results[0], 'masks'):
            mask = roi_results[0].masks.data[0].numpy()  # Convert mask to numpy array
        
            # Ensure mask is a 2D array
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
        
            # Resize the mask to fit the ROI
            mask_resized = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        
            # Overlay the mask on the original image
            mask_overlay = (mask_resized > 0.5).astype(np.uint8) * 255  # Threshold and convert to binary mask
            image_np[y1:y2, x1:x2, 0] = np.maximum(image_np[y1:y2, x1:x2, 0], mask_overlay)
    
    # Convert the result to an image
    result_image = Image.fromarray(image_np)
    return result_image

# Run detection and segmentationimage_path = 'path/to/your/tunnel_image.jpg'
image = Image.open(image_path).convert("RGB")

# Detect ROIs
bboxes = detect_rois(image_path, det_model)

# Segment ROIs
result_image = segment_rois(image, bboxes, seg_model)

# Display the result
plt.imshow(result_image)
plt.axis('off')
plt.show()