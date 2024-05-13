




from PIL import Image
import os



import os
import glob
i = 0
j = 0
for img in glob.glob("C:/Users/h4tec/Desktop/DBCopy/Cropped dataset KT/*.jpg"):
    
    print("now open img")
    image= Image.open(img)
    s = os.path.basename(img)
    name =s.replace('.jpg', '')
    
    image_width, image_height = image.size
    text_file_path = "C:/Users/h4tec/Downloads/1767image/obj_train_data/" + name + ".txt"  #C:Users/h4tec/Downloads/1767image/obj_train_data/" + name + ".txt
    if os.path.exists(text_file_path):
        with open(text_file_path, 'r') as file:
            check_file = os.stat(text_file_path).st_size
            if check_file > 0:
                print("The file is not empty.")
            else:
                           
                print(" empty") 
                file.close()
                image.close()
                os.remove(text_file_path)
                os.remove(img)
    else:
        print("file does not exist")
          
        image.close()
        os.remove(img)

