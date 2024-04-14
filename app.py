from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import json
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('model/plant_disease_prediction_model.h5')

predictions = {0: 'Apple Scab',
    1: 'Apple Black Rot',
    2: 'Cedar Apple Rust',
    3: 'Apple Healthy',
    4: 'Blueberry Healthy',
    5: 'Cherry Powdery Mildew (including sour)',
    6: 'Cherry Healthy (including sour)',
    7: 'Corn Cercospora Leaf Spot',
    8: 'Corn Common Rust',
    9: 'Corn Northern Leaf Blight',
    10: 'Corn Healthy',
    11: 'Grape Black Rot',
    12: 'Grape Esca(Black Measles)',
    13: 'Grape Leaf Blight(Isariopsis Leaf Spot)',
    14: 'Grape Healthy',
    15: 'Orange Haunglongbing (Citrus Greening)',
    16: 'Peach Bacterial Spot',
    17: 'Peach Healthy',
    18: 'Pepperbell Bacterial Spot',
    19: 'Pepperbell Healthy',
    20: 'Potato Early Blight',
    21: 'Potato Late Blight',
    22: 'Potato Healthy',
    23: 'Raspberry Healthy',
    24: 'Soybean Healthy',
    25: 'Squash Powderry Mildew',
    26: 'Strawberry Leaf Scorch',
    27: 'Strawberry Healthy',
    28: 'Tomato Bacterial Spot',
    29: 'Tomato Early Blight',
    30: 'Tomato Late Blight',
    31: 'Tomato Leaf Mold',
    32: 'Tomato Septora Leaf Spot',
    33: 'Tomato Spidermites Two Spotted Spider Mite',
    34: 'Tomato Target Spot',
    35: 'Tomato Yellow Leaf Curl Virus',
    36: 'Tomato Mosaic Virus',
    37: 'Tomato Healthy'
    }

# Function to preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict image class
def predict_image_class(image_path):
    preprocessed_img = preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    # predicted_class_name = predictions[predicted_class_index]
    # return predicted_class_index
    return int(predicted_class_index)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['imagefile']
        if file:
            image_path = 'images/' + file.filename
            file.save(image_path)
            predicted_class_index = predict_image_class(image_path)
            predicted_class_name = predictions[predicted_class_index]
            return jsonify({'prediction': predicted_class_name})
    return jsonify({'error': 'No file uploaded'})

if __name__ == '__main__':
    app.run(debug=True)
