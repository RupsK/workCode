from PIL import Image

# Function to convert YOLO coordinates to pixel coordinates
def yolo_to_pixels(yolo_coords, image_width, image_height):
    x_center, y_center, width, height = yolo_coords
    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height
    
    # Calculate the top-left corner of the bounding box
    x_min = int(x_center - (width / 2)) -20
    y_min = int(y_center - (height / 2))-20
    
    # Calculate the bottom-right corner of the bounding box
    x_max = int(x_center + (width / 2))+20

    y_max = int(y_center + (height / 2))+20
    
    return (x_min, y_min, x_max, y_max)


import os
import glob
i = 0
j = 0
for img in glob.glob("C:/Users/h4tec/Downloads/train/images/*.jpg"):
    
    image= Image.open(img)
    s = os.path.basename(img)
    name =s.replace('.jpg', '')
    image_width, image_height = image.size
    text_file_path = "C:/Users/h4tec/Downloads/train/labels/" + name + ".txt" # Replace with the path to your text file
    
    image_width, image_height = image.size
    
    
    with open(text_file_path, 'r') as file:
        lines = file.readlines()
    # Loop over each line in the text file
    
    for i, line in enumerate(lines):
        _, x_center, y_center, width, height = map(float, line.strip().split())
        yolo_coords = (x_center, y_center, width, height)
        pixel_coords = yolo_to_pixels(yolo_coords, image_width, image_height)
    
        # Crop the image using the calculated pixel coordinates
        cropped_image = image.crop(pixel_coords)
    
        # Save or show the cropped image
        cropped_image_path = 'C:/Users/h4tec/Downloads/train/images/'+'image' +str(i)+str(j)+'.jpg' # Replace with the path where you want to save the cropped image
        cropped_image.save(cropped_image_path)
        j = j + 1
    i = i + 1  # Save each crop with a unique file name
      
        
        
        
        

    
    
    
    
    
    
    
    
    