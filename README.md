# Fengshu-AI

## Quick Start

1. **Install Dependencies**:  
   ```bash
   pipenv install
   ```
   
2. **Configure Your OpenAI API Key**:
   - Create or edit `config.json` in the project root and add your OpenAI API key:
     ```json
     {
       "OPENAI_API_KEY": "your-openai-api-key-here"
     }
     ```
   
3. **Run the Application**:  
   ```bash
   pipenv run python app_integrated.py
   ```
   
4. **Access the Web Interface**:  
   Open your browser and go to [http://127.0.0.1:3000](http://127.0.0.1:3000).

5. **Upload Your First Image**:  
   Click on "Choose File" to select an image and it will auto-upload. Once uploaded, click "Predict" to see furniture detection and Feng Shui recommendations.

---

## Overview

**Fengshu-AI** uses AI and computer vision to analyze a room from a single image, detect furniture, and offer Feng Shui-based design improvements.

## Setup Data (Optional)

If you have a custom dataset:

1. Download the dataset from [here](https://universe.roboflow.com/mokhamed-nagy-u69zl/furniture-detection-qiufc).
2. Place the downloaded dataset in the `data` folder.

## Features

- **Room Analysis**: Upload a room image for analysis.
- **Furniture Identification**: Detect furniture in the image using computer vision.
- **Feng Shui Recommendations**: Receive suggestions to improve the space based on Feng Shui principles.
- **Actionable Insights**: Get practical advice on rearranging furniture and adding elements (e.g., plants, lighting) to enhance harmony.

## Technologies

- **Computer Vision**: YOLO-based object detection.
- **Vision-Language Models (VLMs)**: Combine image and textual data for smarter suggestions.
- **LLMs (OpenAI)**: Generate Feng Shui recommendations based on detected furniture and layout.

## Feng Shui Principles Applied

- **Do**: Maximize natural light, use plants, incorporate elements of fire and earth, maintain cleanliness, and create a balanced space.
- **Don’t**: Overcrowd with furniture, use too many colors, or block energy flow.

## References

- [Feng Shui Dos and Don'ts](https://www.qcdesignschool.com/2022/11/the-ultimate-list-of-feng-shui-dos-and-donts/)
- [InteriorAI](https://interiorai.com/)
- [Text-to-Room](https://lukashoel.github.io/text-to-room/)
- [Polycam](https://poly.cam/)
- [VLMs on Hugging Face](https://huggingface.co/blog/vlms)
- [F-VLM by Google Research](https://research.google/blog/f-vlm-open-vocabulary-object-detection-upon-frozen-vision-and-language-models/)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

[MIT License](LICENSE)