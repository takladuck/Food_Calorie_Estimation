# Food Calorie Estimator

A simple Flask application that predicts food type from an uploaded image and estimates its calorie content using a pre-trained model on the Food-101 dataset.

## Features
- Upload an image of a food item.
- The server classifies the food type.
- Displays the estimated calories for the predicted food.
## Photos
| ![Image](images/1.png) | ![Image](images/2.png) |
|------------------------|------------------------|
| ![Image](images/3.png) | ![Image](images/4.png) |
## Requirements
- Python 3.x
- Flask
- OpenCV
- NumPy
- TensorFlow
- tensorflow-datasets
- Werkzeug
- (Optional) GPU support for faster inference

## Installation
1. Clone this repository or download the source code.
2. Create and activate a virtual environment (recommended).
3. Install the dependencies:
4. Place the trained model file (`food_calorie_model_s.keras`) in the project folder.
```pip install -r requirements.txt```

## Usage
1. Run the Flask application:
```python main.py```
2. 2. Navigate to `http://127.0.0.1:5000` in your browser.
3. Upload an image and view the estimated calories.

## Project Structure
- `main.py` contains the Flask app and logic for image prediction.
- `templates/index.html` handles the web interface for uploading files and displaying results.
- `static/uploads/` stores uploaded images.

## **A brief explanation of the training logic**

### **1. Dataset Loading (TFDS - TensorFlow Datasets)**
- The **Food-101 dataset** is automatically downloaded and loaded using `tensorflow_datasets.load("food101")`.
- It contains **101 food classes** with **75,750 training images** and **25,250 validation images**.
- The dataset is split into **train (75%)** and **validation (25%)** sets.

### **2. Preprocessing Steps**
- **Resize** all images to `224x224` (required input size for MobileNetV2).
- **Normalize** pixel values to the range `[0,1]` by dividing by 255.
- **Batching & Prefetching** is applied for efficient training:
  - `batch(BATCH_SIZE)` groups images for parallel training.
  - `prefetch(AUTOTUNE)` speeds up data loading.

### **3. Transfer Learning with MobileNetV2**
- **MobileNetV2** is used as a **pre-trained model** (trained on ImageNet dataset).
- The **base model is frozen** (pre-trained weights are not updated).
- A **global average pooling layer** is added to extract important features.
- A **fully connected output layer** with `101` neurons (softmax activation) classifies the food items.

### **4. Model Compilation**
- **Loss Function**: `sparse_categorical_crossentropy` (for multi-class classification).
- **Optimizer**: `adam` (adaptive learning rate for efficient training).
- **Metrics**: `accuracy` (to evaluate model performance).


## License
This project is distributed under an open-source license. Refer to your chosen license for additional details.
