from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('tomato_disease_model.h5')

# Dictionary of disease classes
disease_classes = {
    0: "Bacterial Spot",
    1: "Early Blight",
    2: "Late Blight",
    3: "Leaf Mold",
    4: "Septoria Leaf Spot",

    5: "Spider Mites",
    6: "Target Spot",
    7: "Yellow Leaf Curl Virus",
    8: "Mosaic Virus",
    9: "Healthy"
}

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = image.astype("float") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to classify the tomato disease
def classify_disease(image_path):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions)
    disease = disease_classes[predicted_class]
    confidence = predictions[0][predicted_class] * 100
    return disease, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']
        image_path = 'static/uploads/image.jpg'
        file.save(image_path)

        # Classify the disease
        disease, confidence = classify_disease(image_path)

        return render_template('result.html', disease=disease, confidence=confidence)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
