from flask import Flask, request, jsonify, render_template, send_file
from ultralytics import YOLO
import os
from PIL import Image
from io import BytesIO

from src.furnituredetector import FurnitureDetector 
from src.fengshuiadvisor import FengshuiAdvisor

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

detector = FurnitureDetector(model_path="yolo11n.pt")
advisor = FengshuiAdvisor(config_path='config.json', font_path='data/Virgil.ttf')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image file found in the request"}), 400

    image_file = request.files["image"]
    image_path = os.path.join(UPLOAD_FOLDER, "current_image.jpg")
    image_file.save(image_path)

    # Return the uploaded image directly to the frontend for preview
    with open(image_path, "rb") as f:
        data = f.read()
    return send_file(BytesIO(data), mimetype='image/jpeg')

@app.route("/predict", methods=["POST"])
def predict():
    image_path = os.path.join(UPLOAD_FOLDER, "current_image.jpg")
    if not os.path.exists(image_path):
        return jsonify({"error": "No image has been uploaded yet"}), 400

    # Run detection and Feng Shui advisor
    base_image = detector.pre_process_image(image_path)
    annotated_image, detected_items = detector.detect_objects(Image.fromarray(base_image))
    annotated_image_with_advice, resp = advisor.process_image(base_image, annotated_image, detected_items)

    # Return the processed image
    output = BytesIO()
    annotated_image_with_advice.save(output, format='JPEG')
    output.seek(0)
    return send_file(output, mimetype='image/jpeg', as_attachment=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
