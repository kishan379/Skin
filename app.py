import os
import cv2
import numpy as np
import base64
import random
import string
import tensorflow as tf
from flask import Flask, request, render_template, session, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CORS(app, resources={r"/*": {"origins": "*"}})

app.secret_key = "supersecretkey"  # Required for session management

model_path = "model/skin_disease_model.h5"

if not os.path.exists(model_path):
    print(f"❌ Model file missing: {model_path}")
else:
    # print("✅ Model found! Loading...")
    model = tf.keras.models.load_model(model_path)
    # print("✅ Model Loaded Successfully!")


# Class Labels
class_labels = ['BA-cellulitis', 'BA-impetigo', 'FU-athlete-foot', 'FU-nail-fungus', 'FU-ringworm',
                'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles']

# Simulated disease prediction function (Replace with ML model)
def predict_disease(image_path):
    diseases = ['BA-cellulitis', 'BA-impetigo', 'FU-athlete-foot', 'FU-nail-fungus', 'FU-ringworm',
                'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles']
    prediction = random.choice(diseases)  # Random prediction
    return prediction

@app.route("/")
def home():
    return "Hello, Flask is Running!"


@app.route("/upload_base64", methods=["POST"])
def upload_base64():
    try:
        data = request.json  
        image_data = data.get("image")  
       

        if not image_data:
            return jsonify({"error": "No image data received"}), 400

        # Base64 string cleanup
        image_data = image_data.replace("data:image/jpeg;base64,", "").replace("data:image/png;base64,", "")
        
        # Decode Base64 image
        image_bytes = base64.b64decode(image_data)
        
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        
        # Generate the unique file name
        file_path = "static/uploads/"+"/"+random_str+".jpg"
        # img_path = "static/uploads/" + img_file.filename

        # Stored in file
        with open(file_path, "wb") as image_file:
            image_file.write(image_bytes)
    
        # **Detecting if the image is skin-related**
        if not is_skin_image(file_path):
            os.remove(file_path)  # Remove invalid image
            return jsonify({"error": "Invalid Image! Please upload a skin-related image."}), 400

        # **Disease Prediction**
        predicted_disease = predict_disease(file_path)
        random_number = random.randint(60, 90)
        random_number1 = random.randint(20, 30)

        return jsonify({
            "message": "Image uploaded successfully",
            "image_url": "http://127.0.0.1:5000/"+file_path,
            "prediction": predicted_disease,
            "red":random_number,
            "green":random_number1,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Here the OpenCV module is defining
def is_skin_image(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        return False  # Invalid image

    # Convert to HSV for skin detection
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define skin color range in HSV
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)  
    upper_skin = np.array([20, 180, 255], dtype=np.uint8)  

    # Create a mask
    mask = cv2.inRange(img_hsv, lower_skin, upper_skin)
    
    # Calculate skin ratio
    skin_ratio = np.sum(mask > 0) / (img.shape[0] * img.shape[1])
    
    if skin_ratio < 0.15:  # Minimum 15% skin required
        return False
    
    # Additional check for cartoons (Texture Analysis)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)  # Edge detection

    edge_ratio = np.sum(edges > 0) / (img.shape[0] * img.shape[1])

    if edge_ratio > 0.01:  # Cartoon images have high edges
        return False  

    return True
    

@app.route("/upload", methods=["POST"])
def upload():
    
    if request.method == "POST":
        img_file = request.files.get("image")
        print("imag_file",img_file)
        
        if img_file:
            img_path = "static/uploads/" + img_file.filename
            img_file.save(img_path)
            
            # Detecting the skin related image
            if not is_skin_image(img_path):
                os.remove(img_path)    # Remove the invalid image
                return jsonify({"error": "Invalid Image! Please upload a skin-related image."}), 400
            

            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            if model:
                preds = model.predict(img_array)
                session['prediction'] = class_labels[np.argmax(preds)]  # Store prediction in session
                prediction = class_labels[np.argmax(preds)]
                prediction_probability = float(np.max(preds)) * 100
            else:
                prediction = "Unknown (Model not loaded)"
                prediction_probability = 0.0
                
            
            random_number = random.randint(60, 80)
            random_number1 = random.randint(20, 30)
            
            print("prediction:", prediction)

            response_data = {
               
                "prediction": prediction,
                "prediction_probability": prediction_probability,
                "survival_probability": 100 - prediction_probability,
                "image_url": "http://127.0.0.1:5000/"+img_path,
                "red":random_number,
                "green":random_number1,
            }
            return jsonify(response_data)
            
        else:
            session.pop('prediction', None)  # Remove prediction if no image uploaded

    return jsonify({"prediction": prediction, "image_url": img_path}) 
    # return render_template("index.html", prediction=session.get('prediction', None))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)