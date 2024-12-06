from flask import Flask, request, jsonify, render_template, send_file
import os
from PIL import Image
from io import BytesIO
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
    # If annotated_image is a NumPy array, convert it to PIL Image:
    if isinstance(annotated_image, np.ndarray):
        annotated_image = Image.fromarray(annotated_image)

    # Now save the intermediate result
    intermediate_path = os.path.join(UPLOAD_FOLDER, "detector_output.jpg")
    annotated_image.save(intermediate_path, format='JPEG')

    # Return the annotated image (detector result)
    with open(intermediate_path, "rb") as f:
        data = f.read()
    return send_file(BytesIO(data), mimetype='image/jpeg')

@app.route("/predict_advisor", methods=["POST"])
def predict_advisor():
    intermediate_path = os.path.join(UPLOAD_FOLDER, "detector_output.jpg")
    if not os.path.exists(intermediate_path):
        return jsonify({"error": "No detector output found"}), 400

    # Load the previously processed image
    base_image_path = os.path.join(UPLOAD_FOLDER, "current_image.jpg")
    base_image = detector.pre_process_image(base_image_path)

    detector_output_image = Image.open(intermediate_path)

    # In the previous step, we had `detected_items`. To get these again,
    # you may need to store them or re-run detection. For simplicity here,
    # we might re-run detection to get the items:
    # (Better approach is to store `detected_items` somewhere persistent)
    _, detected_items = detector.detect_objects(Image.fromarray(base_image))

    # Apply Feng Shui advice
    annotated_image_with_advice, resp = advisor.process_image(base_image, detector_output_image, detected_items)

    final_path = os.path.join(UPLOAD_FOLDER, "advisor_output.jpg")
    annotated_image_with_advice.save(final_path, format='JPEG')

    # Return the final annotated image
    with open(final_path, "rb") as f:
        data = f.read()
    return send_file(BytesIO(data), mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
