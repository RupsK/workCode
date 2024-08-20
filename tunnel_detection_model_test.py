




from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO('C:/Users/h4tec/Downloads/bestTunnelDetection.onnx')

# Load an image
image_path ="C:/Users/h4tec/Desktop/testReba/reba2.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Perform prediction
results = model.predict(source=image_path)

# Get bounding boxes and plot the results
for result in results:
    # Draw bounding boxes on the image
    for bbox in result.boxes.xyxy:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Display the image with bounding boxes
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()



