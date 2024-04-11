from unittest import result
from flask import Flask, jsonify, render_template, request

import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.preprocessing.image import load_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import decode_predictions

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/api/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/img.jpeg"
    imagefile.save(image_path)

    model = keras.models.load_model("./model/plant_disease_prediction_model.h5")
    img = tf.keras.utils.load_img(image_path, target_size=(256,256))
    i = tf.keras.preprocessing.image.img_to_array(img)
    im = preprocess_input(i)
    img = np.expand_dims(im, axis = 0)
    pred = np.argmax(model.predict(img))

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

    result = pred

    return jsonify(Result = result)

if __name__ == '__main__':
    app.run(debug= True)