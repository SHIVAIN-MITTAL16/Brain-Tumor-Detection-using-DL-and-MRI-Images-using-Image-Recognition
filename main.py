from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join('models', 'model.h5')
model = load_model(MODEL_PATH)

# Class labels (update as per your model)
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Uploads folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Prediction function
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    label = class_labels[predicted_index]
    if label == 'notumor':
        return "No Tumor Detected", confidence
    else:
        return f"Tumor Detected: {label.title()}", confidence

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            result, confidence = predict_tumor(filepath)

            return render_template('index.html',
                                   result=result,
                                   confidence=f"{confidence * 100:.2f}%",
                                   file_path=f'/uploads/{file.filename}')
        else:
            return render_template('index.html', result="Invalid file format. Please upload a .jpg, .jpeg, or .png image.")

    return render_template('index.html', result=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Start the server
if __name__ == '__main__':
    app.run(debug=True)
