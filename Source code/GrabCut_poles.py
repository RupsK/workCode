import cv2
import numpy as np
import matplotlib.pyplot as plt

def grabcut_segmentation(image_path):
    # Read the input image
    img = cv2.imread(image_path)
    
    # Create a mask initialized with background (2), and set probable foreground (3)
    mask = np.zeros(img.shape[:2], np.uint8)
    rect = (50, 50, img.shape[1]-50, img.shape[0]-50)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Apply GrabCut algorithm
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Modify the mask to create a binary mask for the foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply the mask to the original image
    result = img * mask2[:, :, np.newaxis]
    
    # Display the result
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(122), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('Segmented Image')
    plt.show()

# Replace 'your_image.jpg' with the path to the image you want to segment
grabcut_segmentation('C:/Rupali Shinde/transformers.jpg')