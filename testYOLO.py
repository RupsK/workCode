import cv2
import numpy as np

# Load your image
image = cv2.imread('C:/Users/h4tec/Desktop/sample/11.jpg') #C:/Users/h4tec/Desktop/redHole.jpg

 # Save the modified pixels as .png')
image = cv2.resize(image, (400,800))


# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range of red color in HSV
# These ranges may need to be adjusted for your specific image
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Threshold the HSV image to get only red colors
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = mask1 + mask2

# Apply edge detection to the red regions
edges = cv2.Canny(red_mask, 100, 200)

# Dilate the edges to make them more pronounced
kernel = np.ones((3,3), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)

# Use the dilated edges as a mask to find blobs
# Invert the mask for blob detection (blobs should be white on black background)
inverted_mask = cv2.bitwise_not(dilated_edges)

# Set up the SimpleBlobDetector parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 180  # Adjust as needed
params.maxArea = 500
params.filterByCircularity = True
params.minCircularity  = 0.5  # Since holes might not be perfectly circular
params.filterByConvexity = False
#params.minConvexity = 0.5
params.filterByInertia = False
#params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(inverted_mask)
print(keypoints)
if keypoints == ():
    print("It is PC pole")
else: 
    result = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Red-Edged Holes Detected", result)

# Show the final image with detected red-edged holes

cv2.waitKey(0)
cv2.destroyAllWindows()