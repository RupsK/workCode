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
    results = model(image_path, conf=0.01)
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
        
        roi_results = seg_model(roi_pil, conf= 0.1)
        
        if len(roi_results) > 0 and hasattr(roi_results[0], 'masks') and roi_results[0].masks is not None:
            mask = roi_results[0].masks.data[0].numpy()

            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)

            mask_resized = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_overlay = (mask_resized > 0.5).astype(np.uint8)
            combined_mask[y1:y2, x1:x2] = np.logical_or(combined_mask[y1:y2, x1:x2], mask_overlay)
            # Highlight the segmented region in the red channel
            image_np[y1:y2, x1:x2, 0] = np.maximum(image_np[y1:y2, x1:x2, 0], mask_overlay * 255)
        
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"Class {int(cls)}", fill="red", font=font)
    
    result_image = Image.fromarray(image_np)
    return result_image, combined_mask

def segment_image(image, seg_model):
    results = seg_model(image, conf = 0.1)
    
    if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
        mask = results[0].masks.data[0].numpy()

        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        mask_resized = cv2.resize(mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
        mask_overlay = (mask_resized > 0.5).astype(np.uint8)
        image_np = np.array(image)
        image_np[:, :, 1] = np.maximum(image_np[:, :, 1], mask_overlay * 255)
        
        result_image = Image.fromarray(image_np)
        return result_image, mask_overlay
    else:
        return image, None

def combine_masks(combined_mask, segmentation_mask):
    if combined_mask is None:
        return segmentation_mask
    if segmentation_mask is None:
        return combined_mask
    return np.logical_or(combined_mask, segmentation_mask).astype(np.uint8)

def overlay_masks(image, combined_mask, segmentation_mask):
    combined_mask_colored = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    
    if combined_mask is not None:
        combined_colored_mask = np.zeros_like(combined_mask_colored)
        combined_colored_mask[:, :, 0] = combined_mask * 255  # Red channel for combined mask
        combined_mask_colored = np.maximum(combined_mask_colored, combined_colored_mask)

    if segmentation_mask is not None:
        segmentation_colored_mask = np.zeros_like(combined_mask_colored)
        segmentation_colored_mask[:, :, 1] = segmentation_mask * 255  # Green channel for segmentation mask
        combined_mask_colored = np.maximum(combined_mask_colored, segmentation_colored_mask)
    
    return Image.fromarray(combined_mask_colored)

# Path to the folder containing images
folder_path = "C:/Users/h4tec/Desktop/testReba"
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

        # Combine masks from both methods
        combined_mask_final = combine_masks(combined_mask, segmentation_mask)
        combined_segmentation_image = overlay_masks(image, combined_mask, segmentation_mask)

        # Display the results
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(result_image_combined)
        plt.title('Combined Detection and Segmentation')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(result_image_segmentation)
        plt.title('Only Segmentation')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(combined_segmentation_image)
        plt.title('Overlayed Segmentation')
        plt.axis('off')

        plt.suptitle(filename)
        plt.show()
