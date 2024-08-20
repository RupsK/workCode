import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

# Load the YOLOv8 detection model
det_model = YOLO('C:/Users/h4tec/Downloads/bestTunnelDetection.pt')

# Load the YOLOv8 segmentation model
seg_model = YOLO('C:/Users/h4tec/Downloads/bestTunnelSegment.pt')

def detect_rois(image_path, model):
    # Run inference
    results = model(image_path, conf=0.1)
    
    # Get bounding boxes and class labels
    bboxes = results[0].boxes.xyxy.numpy()  # Adjusted to access the bounding boxes correctly
    classes = results[0].boxes.cls.numpy()
    return bboxes, classes

def segment_rois(image, bboxes, classes, seg_model):
    image_np = np.array(image)
    combined_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for bbox, cls in zip(bboxes, classes):
        x1, y1, x2, y2 = map(int, bbox[:4])  # Ensure bbox coordinates are integers
        roi = image_np[y1:y2, x1:x2]
        roi_pil = Image.fromarray(roi)
        
        # Run segmentation model on the ROI
        roi_results = seg_model(roi_pil)
        
        # Extract masks from the results
        if len(roi_results) > 0 and hasattr(roi_results[0], 'masks') and roi_results[0].masks is not None:
            mask = roi_results[0].masks.data[0].numpy()  # Convert mask to numpy array
        
            # Ensure mask is a 2D array
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
        
            # Resize the mask to fit the ROI
            mask_resized = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        
            # Overlay the mask on the original image
            mask_overlay = (mask_resized > 0.5).astype(np.uint8) * 255  # Threshold and convert to binary mask
            combined_mask[y1:y2, x1:x2] = np.maximum(combined_mask[y1:y2, x1:x2], mask_overlay)
            image_np[y1:y2, x1:x2, 0] = np.maximum(image_np[y1:y2, x1:x2, 0], mask_overlay)
        
        # Draw bounding box and class label
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"Class {int(cls)}", fill="red", font=font)
    
    # Convert the result to an image
    result_image = Image.fromarray(image_np)
    return result_image, combined_mask

def segment_image(image, seg_model):
    # Run segmentation model on the whole image
    results = seg_model(image, conf=0.1 )
    
    # Extract masks from the results
    if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
        mask = results[0].masks.data[0].numpy()  # Convert mask to numpy array
    
        # Ensure mask is a 2D array
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
    
        # Resize the mask to fit the image
        mask_resized = cv2.resize(mask, image.size, interpolation=cv2.INTER_NEAREST)
    
        # Overlay the mask on the original image
        mask_overlay = (mask_resized > 0.5).astype(np.uint8) * 255  # Threshold and convert to binary mask
        image_np = np.array(image)
        image_np[:, :, 0] = np.maximum(image_np[:, :, 0], mask_overlay)
        
        # Convert the result to an image
        result_image = Image.fromarray(image_np)
        return result_image, mask_resized
    else:
        return image, None

# Path to your test image
image_path =  "C:/Users/h4tec/Desktop/reba1.jpg"
image = Image.open(image_path).convert("RGB")

# Detect ROIs
bboxes, classes = detect_rois(image_path, det_model)

# Segment ROIs
result_image_combined, combined_mask = segment_rois(image, bboxes, classes, seg_model)

# Segment entire image without detection
result_image_segmentation, segmentation_mask = segment_image(image, seg_model)

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(result_image_combined)
plt.title('Combined Detection and Segmentation')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(result_image_segmentation)
plt.title('Only Segmentation')
plt.axis('off')

plt.show()
