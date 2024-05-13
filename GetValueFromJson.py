# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:45:47 2024

@author: h4tec
"""

import cv2
import numpy as np
import json 
import os
import argparse

parser=argparse.ArgumentParser("Crop Images from Json Files")
parser.add_argument('--json_dir',type=str, default="C:/Users/h4tec/Desktop/test/images", help= "Directory of JSON files")
parser.add_argument('--img_dir', type=str, default="C:/Users/h4tec/Desktop/test/images", help= "Directory of Image files")
args=parser.parse_args()


json_directory_name=args.json_dir
file_names=[]
for file in os.listdir(json_directory_name):
    if file.endswith(".json"):
        file_names.append(os.path.join(json_directory_name, file))


image_directory_name=args.img_dir

cropped_directory_name="cropped_imgs"


if os.path.exists(cropped_directory_name):
    pass
else:
    os.mkdir(cropped_directory_name)

for file in file_names:
    f=open(file)
    json_data=json.load(f)
    points = json_data['shapes'][0]['points']
    json_img_path=json_data['imagePath']
    true_json_img_path=os.path.join(image_directory_name,json_img_path)
    
    x_s=points[0][0]
    y_s=points[0][1]
    x_e=points[1][0]
    y_e=points[1][1]
    
    #print(x_s)
    #print(y_s)
    #print(x_e)
    #print(y_e)
    
    json_img=cv2.imread(true_json_img_path)
    #print(json_img.shape)	
    img_cropped=json_img[int(y_s):int(y_e),int(x_s),:]
    

    true_cropped_path=os.path.join(cropped_directory_name,json_img_path)
    cv2.imwrite(true_cropped_path,img_cropped)
    print(true_cropped_path)
  
    
