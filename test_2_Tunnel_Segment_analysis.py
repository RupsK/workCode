import os
from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

# Ensure environment is set up correctly
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the YOLOv8 detection and segmentation models
det_model = YOLO('C:/Users/h4tec/Downloads/bestTunnelDetection.pt')
seg_model = YOLO('C:/Users/h4tec/Downloads/bestTunnelSegment.pt')

def detect_rois(image_path, model):
    results = model(image_path, conf=0.1)
    bboxes = results[0].boxes.xyxy.numpy()
    classes = results[0].boxes.cls.numpy()
    return bboxes, classes

def segment_rois(image, bboxes, classes, seg_model):
    image_np = np.array(image)
    combined_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for bbox, cls in zip(bboxes, classes):
        x1, y1, x2, y2 = map(int, bbox[:4])
        roi = image_np[y1:y2, x1:x2]
        roi_pil = Image.fromarray(roi)
        
        roi_results = seg_model(roi_pil)
        
        if len(roi_results) > 0 and hasattr(roi_results[0], 'masks') and roi_results[0].masks is not None:
            mask = roi_results[0].masks.data[0].numpy()

            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)

            mask_resized = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_overlay = (mask_resized > 0.5).astype(np.uint8) * 255
            combined_mask[y1:y2, x1:x2] = np.maximum(combined_mask[y1:y2, x1:x2], mask_overlay)
            image_np[y1:y2, x1:x2, 0] = np.maximum(image_np[y1:y2, x1:x2, 0], mask_overlay)
        
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"Class {int(cls)}", fill="red", font=font)
    
    result_image = Image.fromarray(image_np)
    return result_image, combined_mask

def segment_image(image, seg_model):
    results = seg_model(image)
    
    if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
        mask = results[0].masks.data[0].numpy()

        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        mask_resized = cv2.resize(mask, image.size, interpolation=cv2.INTER_NEAREST)
        mask_overlay = (mask_resized > 0.5).astype(np.uint8) * 255
        image_np = np.array(image)
        image_np[:, :, 0] = np.maximum(image_np[:, :, 0], mask_overlay)
        
        result_image = Image.fromarray(image_np)
        return result_image, mask_resized
    else:
        return image, None

# Path to the folder containing images
folder_path = "C:/Users/h4tec/Desktop/testReba/"
# Loop through each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
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

        plt.suptitle(filename)
        plt.show()
