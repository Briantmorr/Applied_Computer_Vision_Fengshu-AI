#  Fengshu - AI Web API

#  Create a Flask REST Web API


from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
import os

# Initialize Flask app
app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv8 model
model_path = "C:/USD/Applied Computer Vision for AI (AAI-521)/Final Project/furniture_detection/content/furniture_detection/yolov8_furniture/weights/best.pt"  
model = YOLO(model_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image file found in the request"}), 400

    image_file = request.files["image"]
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    results = model.predict(source=image_path)

    # Define a mapping of labels to more meaningful descriptions
    label_descriptions = {
        "Sofa": "Sofa - Living Room",
        "Chair": "Chair - Dining Room or Study",
        "Table": "Table - Dining or Coffee Table",
        "Curtains": "Curtains - Window Coverings",
        "Lamp": "Lamp - Lighting Fixture",
        "Frame": "Picture Frame - Wall Decor",
        # Add more label mappings as needed
    }

    predictions = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            label = model.names[cls]
            description = label_descriptions.get(label, label)  # Use description if available
            conf = float(box.conf)
            xyxy = box.xyxy.tolist()
            predictions.append({"label": description, "confidence": conf, "bbox": xyxy})

    os.remove(image_path)
    return jsonify({"predictions": predictions})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)