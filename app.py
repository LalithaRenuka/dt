from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load YOLO model (assumes you have yolov3.weights, yolov3.cfg, and coco.names in the root directory)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

vehicle_classes = ['car', 'bus', 'truck', 'motorbike', 'bicycle']
emergency_classes = ['ambulance', 'fire truck', 'police car']

# Function to detect vehicles and emergency vehicles
def detect_vehicles(image):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    # Detection code omitted here, use your existing `detect_vehicles` function logic
    
@app.route('/')
def index():
    # Get list of images for user selection
    image_files = os.listdir("static/images")
    return render_template("index.html", images=image_files)

@app.route('/process', methods=['POST'])
def process():
    image_file = request.form.get("selected_image")
    if not image_file:
        return redirect(url_for('index'))

    # Load and process the image
    image_path = os.path.join("static/images", image_file)
    image = cv2.imread(image_path)
    if image is None:
        return "Could not load image", 400

    # Detect vehicles and determine signal time
    vehicle_count, has_emergency_vehicle = detect_vehicles(image)
    signal_time = calculate_signal_time(vehicle_count, has_emergency_vehicle)

    return render_template("results.html", image=image_file, vehicle_count=vehicle_count, 
                           emergency_detected=has_emergency_vehicle, signal_time=signal_time)

if __name__ == "__main__":
    app.run(debug=True)
