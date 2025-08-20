

# 🌸 Flower Detection Project



## Overview

An interactive web application built with Streamlit that uses deep learning to classify flowers into 5 different categories. Simply upload an image of a flower, and the AI model will identify the flower type with confidence scores.

## Features

- 🌺 **Real-time Flower Classification** - Upload any flower image for instant detection
- 📊 **Confidence Scoring** - Get probability scores for all 5 flower classes
- 🎨 **Interactive Interface** - Clean, user-friendly Streamlit web interface
- 📱 **Responsive Design** - Works on desktop and mobile devices
- 🔍 **Visual Analysis** - See both original and processed images side-by-side


## Supported Flower Types

- 🌼 Daisy
- 🌻 Dandelion
- 🌹 Rose
- 🌻 Sunflower
- 🌷 Tulip


## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/flower-detection-project.git
cd flower-detection-project
```

2. **Create virtual environment**

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```


## Usage

1. **Run the application**

```bash
streamlit run flower_detection_app.py
```

2. **Open your browser** and navigate to `http://localhost:8501`
3. **Upload a flower image** using the file uploader
4. **Click "Detect Flower"** to get classification results

## Project Structure

```
flower-detection-project/
│
├── flower_detection_app.py    # Main Streamlit application
├── my_model.keras           # Trained CNN model (add your model here)
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── demo_image.png           # Demo screenshot

```


## Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 124x124 pixels
- **Classes**: 5 flower types
- **Framework**: TensorFlow/Keras


## Requirements

- Python 3.8+
- TensorFlow 2.12+
- Streamlit 1.28+
- OpenCV
- Pillow
- NumPy
- Matplotlib


## Screenshots

### Main Interface
<img src="Screenshot 2025-08-20 140806.jpg" style="height:px;margin-right:32px"/>

### Results Display
<img src="Screenshot 2025-08-20 140750.jpg" style="height:px;margin-right:32px"/>


## Future Enhancements

- [ ] Add more flower classes
- [ ] Implement batch image processing
- [ ] Add data augmentation for better accuracy
- [ ] Deploy to cloud platforms (Heroku, AWS, etc.)
- [ ] Add flower information database
- [ ] Mobile app version


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: [Flower Dataset Source]







