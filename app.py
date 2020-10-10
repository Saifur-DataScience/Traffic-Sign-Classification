from __future__ import division, print_function

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from PIL import Image
from swish_package import swish

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='traffic_classifier.h5'

# Load your trained model
model = load_model(MODEL_PATH)

signs = {0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 
         2:'Speed limit (50km/h)', 3:'Speed limit (60km/h)', 
         4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 
         6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 
         8:'Speed limit (120km/h)', 9:'No passing', 
         10:'No passing for vehicles over 3.5 metric tons', 
         11:'Right-of-way at the next intersection', 12:'Priority road', 
         13:'Yield', 14:'Stop', 15:'No vehicles', 
         16:'Vehicles over 3.5 metric tons prohibited', 17:'No entry', 
         18:'General caution', 19:'Dangerous curve to the left', 
         20:'Dangerous curve to the right', 21:'Double curve', 
         22:'Bumpy road', 23:'Slippery road', 
         24:'Road narrows on the right', 25:'Road work', 
         26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 
         29:'Bicycles crossing', 30:'Beware of ice/snow', 
         31:'Wild animals crossing', 32:'End of all speed and passing limits', 
         33:'Turn right ahead', 34:'Turn left ahead', 35:'Ahead only', 
         36:'Go straight or right', 37:'Go straight or left', 
         38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory', 
         41:'End of no passing', 42:'End of no passing by vehicles over 3.5 metric tons'}


def model_predict(img_path, model): 
    # Loading images in a variable
    img = image.load_img(img_path, target_size=(30, 30))
    
    # Preprocessing the image
    x = image.img_to_array(img)
    
    ## Scaling
    x = x/255
    x = np.expand_dims(x, axis=0)
   
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    
    for i in range(0, 43): 
        if preds==i: 
            preds = signs[i]
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)