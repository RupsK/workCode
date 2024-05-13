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
    
  
   
    
    with open(text_file_path, 'r') as file:
        check_file = os.stat(text_file_path).st_size
        
        if(check_file == 0):
            print("The file is empty.")
        else:
            print("The file is not empty.")
        
            _, x_center, y_center, width, height = map(float, file.readline().strip().split())
            
          
    yolo_coords = (x_center, y_center, width, height)
    pixel_coords = yolo_to_pixels(yolo_coords, image_width, image_height)

   # Crop the image using the calculated pixel coordinates
    cropped_image = image.crop(pixel_coords)
   
        # Save or show the cropped image
    cropped_image_path = 'C:/Users/h4tec/Downloads/cropedFinal 4/'+ 'crop' +str(i)+'.jpg' # Replace with the path where you want to save the cropped image

    cropped_image.save(cropped_image_path)
    
    i = i + 1  # Save each crop with a unique file name
      
        
        
        
        

    
    
    
    
    
    
    
    
    