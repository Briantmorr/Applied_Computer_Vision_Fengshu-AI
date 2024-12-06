# Fengshu - AI Web API

from flask import Flask, request, jsonify, render_template, send_file
from ultralytics import YOLO
import os
from PIL import Image
from io import BytesIO

from src.furnituredetector import FurnitureDetector 
from src.fengshuiadvisor import FengshuiAdvisor


# Initialize Flask app
app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize YOLOObjectDetector and FengShuiAdvisor instances
detector = FurnitureDetector(model_path="yolo11n.pt")
advisor = FengshuiAdvisor(config_path='config.json', font_path='data/Virgil.ttf')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image file found in the request"}), 400

    # Save the uploaded image
    image_file = request.files["image"]
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)
    print("starting")
    # First, use YOLOObjectDetector to detect objects and create an annotated image
    base_image = detector.pre_process_image(image_path)

    annotated_image, detected_items = detector.detect_objects(Image.fromarray(base_image))

    print("detected")
    print(detected_items)
    # Use FengShuiAdvisor to add Feng Shui recommendations to the annotated image
    annotated_image_with_advice, resp = advisor.process_image(base_image, annotated_image, detected_items)

    print("finished llm annotation")
    print("llm response", resp)

    print(annotated_image_with_advice.size)
    print(type(annotated_image_with_advice))
    # Save the final annotated image with advice to a BytesIO object for returning in response
    output = BytesIO()
    annotated_image_with_advice.save(output, format='JPEG')
    output.seek(0)

    # Clean up by removing the uploaded image from the server
    os.remove(image_path)

    # Return the image with LLM suggestions
    return send_file(output, mimetype='image/jpeg', as_attachment=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
