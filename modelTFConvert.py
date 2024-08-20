

import torch
import torch.onnx
import sys
import os


from ultralytics import YOLO

model = YOLO("C:/Users/h4tec/Downloads/bestTunnelDetection.pt")





# Load the YOLOv8 model (replace this with the correct path to your model)
model = torch.load("C:/Users/h4tec/Downloads/bestTunnelDetection.pt")
model.eval()

# Ensure model is not loaded as a dict
if isinstance(model, dict):
    model = model['model']  # Adjust this line based on how your model is saved

# Input to the model
dummy_input = torch.randn(1, 3, 640, 640)

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, "bestTunnelDetection.onnx", verbose=True, opset_version=11)

print(f"Model successfully converted to ONNX format at")

"""
print(f"Model successfully converted to ONNX format at {onnx_file_path}")

import onnx
from onnx_tf.backend import prepare

# Load the ONNX model
onnx_model = onnx.load("C:/Users/h4tec/Downloads/best (5).onnx")

# Convert the ONNX model to TensorFlow
tf_rep = prepare(onnx_model)
tf_model_path = "model_tf"
tf_rep.export_graph(tf_model_path)

print(f"Model successfully converted to TensorFlow format at {tf_model_path}")


import tensorflow as tf

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model("model_tf")
tflite_model = converter.convert()

# Save the TensorFlow Lite model
tflite_model_path = "model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Model successfully converted to TensorFlow Lite format at {tflite_model_path}")

"""

#'C:/Users/h4tec/Downloads/best (5).pt'"""