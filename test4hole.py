

import cv2
import numpy as np

# Load the image
image_path = 'C:/Users/h4tec/Desktop/sample/43.jpg'  # Make sure to provide the correct path
image = cv2.imread(image_path)
#image = cv2.resize(image, (400,800))
# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Invert the grayscale image to make black areas appear as white
inverted_image = cv2.bitwise_not(gray_image)

# Threshold the inverted image to further enhance the detection of black holes
_, thresh_image = cv2.threshold(inverted_image, 220, 255, cv2.THRESH_BINARY)

# Initialize the SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 255  # Since the black areas are now white after inversion
params.filterByArea = True
params.minArea = 100  # Min area of blobs

params.maxArea = 500
params.filterByCircularity = True  # Depending on the shape of the holes
params.minCircularity  = 0.5
params.filterByConvexity = True
params.minConvexity = 0.5
 
params.filterByInertia =True
params.minInertiaRatio = 0.2

# Create a blob detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs in the thresholded image
keypoints = detector.detect(thresh_image)
print(keypoints)

blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Show blobs
cv2.imshow("Black Holes Detected", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()