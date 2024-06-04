from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle

app = Flask(__name__)

# Set the upload folder
app.config['UPLOAD_FOLDER'] = 'D:/final mini project/uploads'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

model = load_model('D:/Mini Project 3-2/Final project/models/model.h5')


@app.route('/')
def index():
    return render_template('D:/Mini Project 3-2/Final project/templates/index1.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = predict_cancer(filepath)
            return render_template('upload.html', result=result)
    return render_template('result.html')

def predict_cancer(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust target size to your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # If your model expects pixel values to be in range [0, 1]

    prediction = model.predict(img_array)
    # Assuming your model outputs probabilities for two classes
    if prediction[0][0] > 0.5:
        return "Cancer Detected"
    else:
        return "No Cancer Detected"

if __name__ == '__main__':
    app.run(debug=True)
