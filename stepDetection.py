import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_image(path):
    # Load an image in grayscale
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or path is incorrect")
    return image

def preprocess_image(image):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Increase contrast
    high_contrast = cv2.equalizeHist(blurred)
    return high_contrast

def detect_lines(image):
    # Canny edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    # Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    return lines

def filter_horizontal_lines(lines, angle_threshold=10):
    # Angle threshold for horizontal lines in degrees
    if lines is not None:
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < angle_threshold:
                horizontal_lines.append(line)
        return horizontal_lines
    return []

def draw_lines(image, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return image

# Main processing function
def process_image(path):
    img = load_image(path)
    processed_img = preprocess_image(img)
    lines = detect_lines(processed_img)
    horizontal_lines = filter_horizontal_lines(lines)
    result_img = draw_lines(img, horizontal_lines)
    return result_img

# Load, process, and display an image
image_path = 'C:/Users/h4tec/Desktop/test/images/0 (115).jpg'  # Change to your image path
result = process_image(image_path)

plt.imshow(result, cmap='gray')
plt.title('Detected Steps on Telecom Pole')
plt.show()
