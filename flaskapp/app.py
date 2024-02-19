from flask import Flask, render_template, request
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import base64

app = Flask(__name__)

# Load the model
new_model = load_model(r"G:\My Drive\final.h5")

# Function to process uploaded image
def process_image(image):
    new_height, new_width = 256, 256
    resized_image = cv2.resize(image, (new_width, new_height))
    pred = new_model.predict(np.expand_dims(resized_image/256, 0))
    return pred

# Function to convert image to base64
def image_to_base64(image):
    _, img_encoded = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return img_base64

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Result page
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        # If file is available and valid, process it
        if file:
            # Read image
            image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
            # Process image
            pred = process_image(image)

            img_base64 = image_to_base64(image)

            if pred > 0.5:
                Result="Provided Logo is Fake"
                return render_template('result.html', prediction=Result, image=img_base64)
            else:
             
                Result="Provided Logo is Original"
            return render_template('result.html', prediction=Result, image=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
