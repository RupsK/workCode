





import cv2
import json

# Read the image
import numpy as np
image_file = 'G:/PC POLE IMAGES/input_1/BUSAN-CHUNGRYONG3E-1NA1-PC(50)-24.jpg'


#image = cv2.imread(image_file)


image = cv2.imread(image_file)
#image_name = image_file.split('/')[-1].split('.')[0]

with open('G:/PC POLE IMAGES/test.json') as f:
  data = json.load(f)

# Function to get target mask coordinates for a given image_id
def get_target_mask_coordinates(data, image_id):
    coordinates_list = []
    global center_list
    center_list= []
    for detection in data['detection_list']:
        if 'target_masks' in detection:
            # Append each coordinate pair as a tuple
            for coordinate_pair in detection['target_masks']:
                coordinates_list.append(tuple(coordinate_pair))
                point1, point2 = coordinate_pair
                #print(coordinate_pair)
                # Unpack coordinates of each point
                x1, y1 = point1
                x2, y2 = point2         

                # Define the circle color (BGR format)
                circle_color = (0, 0, 255)  # Red color

                # Define the circle radius
                circle_radius = 5  # You can adjust this according to your preference

                # Calculate the center of the line segment
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Print or use the center coordinates
                #print("Center of the line segment:", (center_x, center_y))
                
                center_list.append((center_x, center_y))
                

                center_x_int = int(round(center_x))
                center_y_int = int(round(center_y))

                cv2.circle(image, (center_x_int, center_y_int), circle_radius, (0, 255, 0), -1)
   
    return center_list
   
               
    

# Replace 'your_image_id_here' with the actual image_id you want to search for
image_id = "BUSAN-CHUNGRYONG3E-1NA1-PC(50)-24.jpg"
coordinates = get_target_mask_coordinates(data, image_id)
print (center_list)



 # Convert floating point coordinates to integers
int_coordinates = [(int(x), int(y)) for x, y in center_list]# Draw line between each pair of points


for i in range(len(center_list) - 1):
    cv2.line(image, int_coordinates[i], int_coordinates[i + 1], (0, 255, 0), thickness=2)

# Display the image with the line
cv2.imshow('Image with Line', image)
cv2.waitKey(0)  # Wait for a key press to close the displayed window
cv2.destroyAllWindows()

# Optionally, save the image to a file
cv2.imwrite('output_image.jpg', image)         


cv2.imshow('Image with Circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
