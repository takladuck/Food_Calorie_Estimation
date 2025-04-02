from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import tensorflow_datasets as tfds
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load trained model
model = tf.keras.models.load_model("food_calorie_model_s.keras")
# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Food-101 dataset info
ds_info = tfds.builder("food101").info
class_labels = ds_info.features["label"].names  # Get correct class names

# Create a food-to-calorie mapping (You need real values)
food_calories = {
    "apple_pie": 320,
    "baby_back_ribs": 550,
    "baklava": 350,
    "beef_carpaccio": 150,
    "beef_tartare": 200,
    "beet_salad": 180,
    "beignets": 290,
    "bibimbap": 550,
    "bread_pudding": 300,
    "breakfast_burrito": 500,
    "bruschetta": 180,
    "caesar_salad": 360,
    "cannoli": 250,
    "caprese_salad": 300,
    "carrot_cake": 450,
    "ceviche": 200,
    "cheesecake": 400,
    "cheese_plate": 500,
    "chicken_curry": 450,
    "chicken_quesadilla": 500,
    "chicken_wings": 400,
    "chocolate_cake": 450,
    "chocolate_mousse": 300,
    "churros": 220,
    "clam_chowder": 250,
    "club_sandwich": 600,
    "crab_cakes": 350,
    "creme_brulee": 300,
    "croque_madame": 600,
    "cup_cakes": 200,
    "deviled_eggs": 200,
    "donuts": 250,
    "dumplings": 300,
    "edamame": 120,
    "eggs_benedict": 550,
    "escargots": 200,
    "falafel": 300,
    "filet_mignon": 450,
    "fish_and_chips": 800,
    "foie_gras": 400,
    "french_fries": 400,
    "french_onion_soup": 300,
    "french_toast": 400,
    "fried_calamari": 300,
    "fried_rice": 450,
    "frozen_yogurt": 150,
    "garlic_bread": 300,
    "gnocchi": 400,
    "greek_salad": 220,
    "grilled_cheese_sandwich": 500,
    "grilled_salmon": 350,
    "guacamole": 250,
    "gyoza": 350,
    "hamburger": 600,
    "hot_and_sour_soup": 150,
    "hot_dog": 550,
    "huevos_rancheros": 500,
    "hummus": 200,
    "ice_cream": 200,
    "lasagna": 600,
    "lobster_bisque": 300,
    "lobster_roll": 400,
    "mac_and_cheese": 600,
    "macarons": 70,
    "miso_soup": 80,
    "mussels": 300,
    "nachos": 700,
    "omelette": 250,
    "onion_rings": 400,
    "oysters": 100,
    "pad_thai": 600,
    "paella": 500,
    "pancakes": 350,
    "panna_cotta": 200,
    "peking_duck": 450,
    "pho": 450,
    "pizza": 285,
    "pork_chop": 450,
    "poutine": 700,
    "prime_rib": 600,
    "pulled_pork_sandwich": 500,
    "ramen": 550,
    "ravioli": 400,
    "red_velvet_cake": 500,
    "risotto": 450,
    "samosa": 250,
    "sashimi": 150,
    "scallops": 250,
    "seaweed_salad": 100,
    "shrimp_and_grits": 500,
    "spaghetti_bolognese": 600,
    "spaghetti_carbonara": 650,
    "spring_rolls": 200,
    "steak": 600,
    "strawberry_shortcake": 400,
    "sushi": 200,
    "tacos": 300,
    "takoyaki": 300,
    "tiramisu": 400,
    "tuna_tartare": 250,
    "waffles": 350
}

def preprocess_image(image_path):
    """Preprocess the image (resize, normalize, etc.)."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    image = image.astype('float32') / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension


def predict_calories(image_path):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)

    predicted_class = np.argmax(predictions)  # Get highest probability class
    predicted_food = class_labels[predicted_class]  # Get actual food label

    print(f"Predicted class index: {predicted_class}, Predicted food: {predicted_food}")  # Debugging

    return food_calories.get(predicted_food, "Unknown")


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Predict calories using the model
        predicted_calories = predict_calories(file_path)

        return render_template('index.html', image_url=file_path, calories=predicted_calories, dark_mode=True)

    return render_template('index.html', image_url=None, calories=None, dark_mode=True)


if __name__ == '__main__':
    app.run(debug=True)
