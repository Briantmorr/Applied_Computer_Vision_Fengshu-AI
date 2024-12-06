from openai import OpenAI
import json
import cv2
import base64
from PIL import Image, ImageDraw, ImageFont
import textwrap


class FengshuiAdvisor:
    def __init__(self, config_path='../config.json', font_path='../data/Virgil.ttf'):
        self.config = self.get_credentials(config_path)
        self.client = OpenAI(api_key=self.config['open_api_key'])
        self.llm_model = "gpt-4o-mini"
        self.font_path = font_path

    def get_credentials(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def encode_image(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def get_prompt(self, detected_classes):
        prompt = f"""
        Based on Feng Shui principles, provide brief, specific recommendations to improve the energy flow and harmony of the room. 
        Only comment if you have useful recommendations. 

        Follow these principles as a guideline
        Dos
        1. Maximize Light Exposure: Position furniture to allow for maximum natural light.
        2. Incorporate Plants: Use plants like aloe vera, bamboo, or ficus to promote positive energy flow.
        3. Element of Fire: Include elements representing fire to promote passion and energy.
        4. Functional Design: Ensure the room layout allows for maximum comfort and enjoyment.
        5. Keep it Clean: Maintain a clutter-free environment for better energy flow.
        6. Element of Earth: Incorporate earth elements to promote stability and balance.
        7. Commanding Position: Place furniture so you can see the door from where you're sitting.
        8. Mindful TV Placement: Mount TVs appropriately to avoid crowding and distractions.
        9. Create Balance: Use a mix of light and dark colors, and hard and soft furnishings.
        10. Thoughtful Color Choices: Select colors that promote the desired energy (e.g., blue for calmness, red for energy).

        Don'ts
        1. Overuse Colors: Avoid using too many colors to prevent a chaotic atmosphere.
        2. Excessive Furniture: Don't overcrowd the room; use only necessary pieces.
        3. Ignore Energy Flow: Arrange furniture to allow smooth and uninterrupted energy flow.
        4. Furniture Against Walls: Leave space between furniture and walls to avoid blocking energy.
        5. High-Hung Artwork: Hang photos and artwork at eye level to prevent scattered energy.
        
        Assess the key fenghsui principles to see if one of the major points are missing from the photo, if so, add address it as a general recommendation.
        Only output the 3 top recommendations including the general recommendation in JSON format with keys as the names of the furniture you have suggestions for that are present in the image.
        image classes: {detected_classes}.

        Example response:
        {{
            "bed": "The bed should be facing the entrance, not the wall."
            "general": "Try adding a potted plant to improve the energy"
        }}
        """
        return prompt

    def handle_box_collision(self, draw, previous_boxes, bbox, padding, text, font):
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x1, y1 = bbox[0], bbox[1]  # Extract x1, y1 from bbox

        collision = True
        attempt_count = 0
        while collision and attempt_count < 10:  # Try adjusting up to 5 times per direction
            collision = False
            for prev_box in previous_boxes:
                prev_x1, prev_y1, prev_x2, prev_y2 = prev_box
                if not (bbox[2] < prev_x1 + padding or bbox[0] > prev_x2 + padding or bbox[3] < prev_y1 + padding or bbox[1] > prev_y2 + padding):
                    # Collision detected, adjust in the direction with the least distance
                    vertical_distance = abs(y1 - prev_y2) if y1 < prev_y2 else abs(y1 - prev_y1)
                    horizontal_distance = abs(x1 - prev_x2) if x1 < prev_x2 else abs(x1 - prev_x1)
                    if vertical_distance < horizontal_distance:
                        # Adjust vertically
                        if attempt_count < 5:
                            y1 = max(0, y1 - (text_height + 10)) if y1 > prev_y1 else y1 + (text_height + 10)
                        else:
                            x1 = max(0, x1 - (text_width + 10)) if x1 > prev_x1 else x1 + (text_width + 10)
                    else:
                        # Adjust horizontally
                        if attempt_count < 5:
                            x1 = max(0, x1 - (text_width + 10)) if x1 > prev_x1 else x1 + (text_width + 10)
                        else:
                            y1 = max(0, y1 - (text_height + 10)) if y1 > prev_y1 else y1 + (text_height + 10)
                    text_position = (x1, y1 - 20)
                    bbox = draw.multiline_textbbox(text_position, text, font=font)
                    collision = True
                    attempt_count += 1
                    break
        return bbox


    def handle_edge_collision(self, bbox, text_position, image_width, image_height, edge_margin, y2):
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if bbox[2] > image_width:  # Text exceeds right boundary
            text_position = (image_width - text_width - edge_margin, text_position[1])
        if bbox[3] > image_height:  # Text exceeds bottom boundary
            text_position = (text_position[0], image_height - text_height - edge_margin)
        if bbox[0] < edge_margin:  # Text goes beyond left boundary
            text_position = (edge_margin, text_position[1])
        if bbox[1] < edge_margin:  # Text goes above top boundary
            text_position = (text_position[0], y2 + edge_margin)  # Place below the bounding box instead
        return text_position

    def get_gpt4_completion_with_image(self, image, detected_classes):
        base64_image = self.encode_image(image)
        image_data_url = f"data:image/jpeg;base64,{base64_image}"
        
        prompt = self.get_prompt(detected_classes)
        completion = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You are a Feng Shui expert who helps customers re-arrange their room according to Feng Shui principles."},
                {"role": "user",
                 "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url
                                },
                            },
                        ], 
                 },
            ],
            response_format={"type": "json_object"},
            max_tokens=200,
        )
        try:
            response_json = json.loads(completion.model_dump_json(indent=2))
            content_json = json.loads(response_json['choices'][0]['message']['content'])
            return content_json
        except Exception as e:
            return None

    def annotate_llm_response(self, image, detected_items, resp):
        draw = ImageDraw.Draw(image)
        image_width, image_height = image.size
        dynamic_size = max(18, (image_width + image_height) // 120)
        font = ImageFont.truetype(self.font_path, size=dynamic_size)
        padding = dynamic_size / 1.5
        border_width = dynamic_size // 4
        max_text_width = 25

        previous_boxes = []

        sorted_items = sorted(resp.keys(), key=lambda item: min(
            detected_items[item][0][0],
            image_width - detected_items[item][0][2],
            detected_items[item][0][1],
            image_height - detected_items[item][0][3]
        ))

        for item in sorted_items:
            x1, y1, _, y2 = detected_items[item][0]
            text = resp[item]
            if len(text) > max_text_width:
                lines = textwrap.wrap(text, width=max_text_width)
                text = '\n'.join(lines)
            text_position = (x1, y1 - 20)

            bbox = draw.multiline_textbbox(text_position, text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

            # Adjust if boxes are on top of each other or text is more than 3 lines tall
            if len(lines) > 3:
                y1 = max(0, y1 - (text_height + 10))  # Adjust upward
                text_position = (x1, y1 - 20)


            edge_margin = dynamic_size * 1.5 
            # Adjust if text goes out of image boundaries
            text_position = self.handle_edge_collision(bbox, text_position, image_width, image_height, edge_margin, y2)
            bbox = draw.multiline_textbbox(text_position, text, font=font)
            bbox = self.handle_box_collision(draw, previous_boxes, bbox, padding, text, font)

            rect_position = [
                bbox[0] - padding, bbox[1] - padding,
                bbox[2] + padding, bbox[3] + padding / 2
            ]

            fill_color = (255, 236, 153)
            draw.rounded_rectangle(rect_position, fill=fill_color, outline="black", width=border_width, radius=20)

            draw.multiline_text(text_position, text, fill="black", font=font)
            previous_boxes.append(bbox)

        return image

    def process_image(self, base_image, annotated_image, detected_items):
        detected_classes = [key for key in detected_items]
        resp = self.get_gpt4_completion_with_image(annotated_image, detected_classes)
        if 'general' in resp:
            detected_items['general'] = [(10, 10, 10, 10)]

        image_with_annotations = self.annotate_llm_response(Image.fromarray(base_image), detected_items, resp)
        return image_with_annotations, resp


# # Usage Example:
# # Instantiate YOLOObjectDetector to detect objects and annotate an image
# detector = YOLOObjectDetector()
# annotated_image, detected_items = detector.process_image('path/to/your/image.jpg')

# # Instantiate FengShuiAdvisor to add Feng Shui recommendations based on detected items
# advisor = FengShuiAdvisor()
# annotated_image_with_advice = advisor.process_image(annotated_image, detected_items)

# # Display the final annotated image
# annotated_image_with_advice.show()