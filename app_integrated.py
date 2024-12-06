from flask import Flask, request, jsonify, render_template, send_file
import os
from PIL import Image
from io import BytesIO
import base64
import numpy as np

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

    # Return the uploaded image directly for preview
    with open(image_path, "rb") as f:
        data = f.read()
    return send_file(BytesIO(data), mimetype='image/jpeg')

@app.route("/predict_detector", methods=["POST"])
def predict_detector():
    image_path = os.path.join(UPLOAD_FOLDER, "current_image.jpg")
    if not os.path.exists(image_path):
        return jsonify({"error": "No image uploaded yet"}), 400

    # Run detection
    base_image = detector.pre_process_image(image_path)
    annotated_image, detected_items = detector.detect_objects(Image.fromarray(base_image))

    # Convert annotated_image (np array or PIL image) to PIL if needed
    if isinstance(annotated_image, np.ndarray):
        annotated_image = Image.fromarray(annotated_image)

    # Save intermediate result
    intermediate_path = os.path.join(UPLOAD_FOLDER, "detector_output.jpg")
    annotated_image.save(intermediate_path, format='JPEG')

    # Convert annotated image to base64 for sending back in JSON
    buffer = BytesIO()
    annotated_image.save(buffer, format='JPEG')
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode('utf-8')

    # Return JSON with detected_items and base64 image
    return jsonify({
        "detected_items": detected_items,
        "image_base64": encoded_image
    }), 200

@app.route("/predict_advisor", methods=["POST"])
def predict_advisor():
    data = request.get_json()
    if not data or "detected_items" not in data:
        return jsonify({"error": "detected_items not provided"}), 400

    detected_items = data["detected_items"]

    intermediate_path = os.path.join(UPLOAD_FOLDER, "detector_output.jpg")
    if not os.path.exists(intermediate_path):
        return jsonify({"error": "No detector output found"}), 400

    # Load the previously processed images
    base_image_path = os.path.join(UPLOAD_FOLDER, "current_image.jpg")
    base_image = detector.pre_process_image(base_image_path)
    detector_output_image = Image.open(intermediate_path)

    # Apply Feng Shui advice using the detected_items from the client
    annotated_image_with_advice, resp = advisor.process_image(base_image, detector_output_image, detected_items)

    final_path = os.path.join(UPLOAD_FOLDER, "advisor_output.jpg")
    annotated_image_with_advice.save(final_path, format='JPEG')

    # Return the final image as base64
    buffer = BytesIO()
    annotated_image_with_advice.save(buffer, format='JPEG')
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode('utf-8')

    return jsonify({
        "recommendations": resp,
        "image_base64": encoded_image
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
