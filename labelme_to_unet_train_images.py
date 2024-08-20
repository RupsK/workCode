import os
import cv2
import numpy as np
from tqdm import tqdm

# Directories
image_dir = 'dataset/tunnel-8/train/images'
label_dir = 'ataset/tunnel-8/train/lables'
mask_dir = 'dataset/tunnel-8/train_mask'

# Ensure mask directory exists
os.makedirs(mask_dir, exist_ok=True)

# Helper function to create mask from label file
def create_mask(label_file, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    with open(label_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            x_center *= image_shape[1]
            y_center *= image_shape[0]
            width *= image_shape[1]
            height *= image_shape[0]

            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)

            mask[y_min:y_max, x_min:x_max] = 255  # Assuming binary masks
    return mask

# Process each image and corresponding label
for image_filename in tqdm(os.listdir(image_dir)):
    if image_filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, image_filename)
        label_path = os.path.join(label_dir, image_filename.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt'))

        if os.path.exists(label_path):
            image = cv2.imread(image_path)
            mask = create_mask(label_path, image.shape)
            mask_path = os.path.join(mask_dir, image_filename)
            cv2.imwrite(mask_path, mask)
        else:
            print(f"Label file not found for {image_filename}")

print("Masks have been created and saved successfully.")