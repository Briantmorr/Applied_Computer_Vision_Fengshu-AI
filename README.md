# Fengshu-ai

## Overview

**Fengshu-ai** is a project aimed at helping users redesign their living spaces using Feng Shui and interior design principles. By leveraging AI and computer vision technologies, Fengshu-ai provides practical suggestions for rearranging existing furniture and enhancing room layouts to promote positive energy flow and harmony.

## Getting Started

To get started with **Fengshu-ai**, you will need to download the dataset and place it in the appropriate folder.

1. Download the dataset from [here](https://universe.roboflow.com/mokhamed-nagy-u69zl/furniture-detection-qiufc).
2. Place the downloaded dataset in the `data` folder.

## Features

- **Room Analysis**: Upload a photo of your room for analysis.
- **Furniture Identification**: Use AI to extract and identify current room furniture.
- **Feng Shui Recommendations**: Receive alternatives to redesign your space using Feng Shui principles.
- **Improvement Suggestions**: Get actionable suggestions for improvements with various pieces of furniture.

## Technologies Used

- **Computer Vision**: For object detection and furniture identification.
- **Vision and Language Models (VLMs)**: To process visual data and apply textual Feng Shui rules.
- **Large Language Models (LLMs)**: For generating design suggestions based on identified furniture and Feng Shui principles.

## Roadmap

- **Furniture Detection**: Develop methods for extracting and identifying furniture from a single image.
- **Feng Shui Integration**: Apply Feng Shui rules to provide actionable interventions based on furniture identification.
- **User Interface Development**: Create an intuitive interface for users to upload photos and receive suggestions.
- **Visualization Enhancements**: Implement techniques to show visual representations of suggested redesigns.
- **Expert Insights**: Integrate expert interior design advice for more effective suggestions.
- **Technology Exploration**:
  - Evaluate tools like [Text-to-Room](https://lukashoel.github.io/text-to-room/) for room modeling.
  - Consider [Polycam](https://poly.cam/) for 3D scanning using LiDAR technology.
  - Explore Vision and Language Models as per [Hugging Face's VLMs](https://huggingface.co/blog/vlms).
  - Research on [F-VLM](https://research.google/blog/f-vlm-open-vocabulary-object-detection-upon-frozen-vision-and-language-models/) for open-vocabulary object detection.

## Feng Shui Principles Applied

### Dos

1. **Maximize Light Exposure**: Position furniture to allow for maximum natural light.
2. **Incorporate Plants**: Use plants like aloe vera, bamboo, or ficus to promote positive energy flow.
3. **Element of Fire**: Include elements representing fire to promote passion and energy.
4. **Functional Design**: Ensure the room layout allows for maximum comfort and enjoyment.
5. **Keep it Clean**: Maintain a clutter-free environment for better energy flow.
6. **Element of Earth**: Incorporate earth elements to promote stability and balance.
7. **Commanding Position**: Place furniture so you can see the door from where you're sitting.
8. **Mindful TV Placement**: Mount TVs appropriately to avoid crowding and distractions.
9. **Create Balance**: Use a mix of light and dark colors, and hard and soft furnishings.
10. **Thoughtful Color Choices**: Select colors that promote the desired energy (e.g., blue for calmness, red for energy).

### Don'ts

1. **Overuse Colors**: Avoid using too many colors to prevent a chaotic atmosphere.
2. **Excessive Furniture**: Don't overcrowd the room; use only necessary pieces.
3. **Ignore Energy Flow**: Arrange furniture to allow smooth and uninterrupted energy flow.
4. **Furniture Against Walls**: Leave space between furniture and walls to avoid blocking energy.
5. **High-Hung Artwork**: Hang photos and artwork at eye level to prevent scattered energy.

## References

- [The Ultimate List of Feng Shui Dos and Don'ts](https://www.qcdesignschool.com/2022/11/the-ultimate-list-of-feng-shui-dos-and-donts/)
- [InteriorAI](https://interiorai.com/)
- [Text-to-Room](https://lukashoel.github.io/text-to-room/)
- [Polycam](https://poly.cam/)
- [Vision and Language Models](https://huggingface.co/blog/vlms)
- [F-VLM by Google Research](https://research.google/blog/f-vlm-open-vocabulary-object-detection-upon-frozen-vision-and-language-models/)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License

[MIT License](LICENSE)

## Fengshu-AI: Furniture Detection Web User Interface (Web API)
This application predicts furniture placements from an uploaded image and displays the results, including labels, confidence scores, and bounding box details, on a web interface. It is built using Python Flask for the backend and HTML/JavaScript for the frontend.

Features
Upload an image of a room to detect furniture (e.g., Sofa, Table).

View predictions with:
* Furniture Label: Detected furniture type.
* Confidence Score: The probability of the detection.
* Bounding Box Details: The exact location of the detected objects.
Easy-to-use web interface for interaction.

Prerequisites
Python: Version 3.7 or above.
Flask: For creating the backend API.
YOLOv8 Model: Pre-trained object detection model.
HTML/JavaScript: For the frontend.

Setup Instructions
1. Clone the Repository
Download or clone this project to your local machine.

bash
Copy code
git clone <repository_url>
cd <project_folder>
2. Install Python Dependencies
Ensure Python is installed on your system. Install required packages using:

bash
Copy code
pip install -r requirements.txt
Note: Ensure the torch library matches your hardware (CPU/GPU).

3. Place YOLOv8 Model
Download the best.pt model file (trained YOLOv8 weights).
Place it in the furniture_detection directory or update its location in the app.py file.

Running the Application
1. Start the Backend Server
Run the app.py file using:

bash
Copy code
python app.py
This will start a Flask server. You should see something like:

csharp
Copy code
 * Running on http://127.0.0.1:5000/

 * 2. Access the Frontend
Open a web browser and navigate to http://127.0.0.1:5000/.
3. Upload an Image and View Predictions
Click the Choose File button to upload an image of a room.
Click the Predict button.
View the detected furniture labels, confidence scores, and bounding box details in the predictions table.
