from flask import Flask, render_template, request, redirect, url_for
import os
import requests
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model from Hugging Face on startup
MODEL_PATH = 'plant_disease_final_model.h5'

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    url = "https://huggingface.co/mohanevs/plant-disease-model/resolve/main/plant_disease_final_model.h5"
    r = requests.get(url, stream=True)
    with open(MODEL_PATH, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model downloaded!")

model = load_model(MODEL_PATH)

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_labels = [
        'Apple Apple Scab', 'Apple Black Rot', 'Apple Cedar Apple Rust', 'Apple Healthy',
        'Corn Cercospora Leaf Spot Gray Leaf Spot', 'Corn Common Rust',
        'Corn Northern Leaf Blight', 'Corn Healthy', 'Grape Black Rot',
        'Grape Esca Black Measles', 'Grape Leaf Blight Isariopsis Leaf Spot', 'Grape Healthy',
        'Potato Early Blight', 'Potato Late Blight', 'Potato Healthy', 'Strawberry Leaf Scorch',
        'Strawberry Healthy', 'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight',
        'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot', 'Tomato Yellow Leaf Curl Virus',
        'Tomato Healthy'
    ]
    return class_labels[class_idx]

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        prediction = predict_disease(filepath)
        return render_template('index.html', filename=file.filename, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)