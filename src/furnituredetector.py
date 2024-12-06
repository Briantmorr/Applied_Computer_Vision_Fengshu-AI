from ultralytics import YOLO
import cv2
from PIL import Image
from ultralytics.utils.plotting import Annotator


# Class 1: YOLO Object Detector
class FurnitureDetector:
    def __init__(self, model_path='yolo11n.pt'):
        self.model = YOLO(model_path)
        self.interior_classes = {
            13: ('bench', (0, 255, 255)),
            56: ('chair', (255, 0, 0)),
            57: ('couch', (0, 255, 0)),
            58: ('potted_plant', (255, 165, 0)),
            59: ('bed', (255, 0, 255)),
            60: ('dining table', (0, 0, 255)),
            61: ('toilet', (0, 128, 255)),
            62: ('tv', (128, 0, 128)),
            75: ('vase', (128, 128, 0))
        }

    def detect_objects(self, image):
        results = self.model(image)
        detected_items = {}

        # Initialize annotator (annotator utility can be part of the YOLO utils or can be customized)
        annotator = Annotator(image)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                if class_id in self.interior_classes:
                    label, color = self.interior_classes[class_id]

                    if label not in detected_items:
                        detected_items[label] = []

                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detected_items[label].append((x1, y1, x2, y2))

                    # Draw the bounding box and label using annotator
                    annotator.box_label(box.xyxy[0], label, color=color)

        # Get the annotated image from the annotator
        annotated_image = annotator.result()

        return annotated_image, detected_items

    def pre_process_image(self, image_path):
        image = Image.open(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
        # annotated_image, detected_items = self.detect_objects(image)
        # return annotated_image, detected_items


