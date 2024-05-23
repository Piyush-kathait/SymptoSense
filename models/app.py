import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image
from keras.preprocessing import image

import numpy as np
import joblib
from tensorflow.keras.models import load_model
app = Flask(__name__)

# Load the trained models
kidney_model = joblib.load('kidney_pred_model.pkl')
symptoms_model = joblib.load('random_forest_model.pkl')

# Load the breast cancer model
breast_cancer_model = tf.keras.models.load_model("ann_model.h5")

# Load the brain tumor model
brain_tumor_model = tf.keras.models.load_model("brainmy_densenet_model.h5")

# Define default values for missing inputs for kidney prediction
kidney_default_values = {
    "age": 50.0, "bp": 80.0, "sg": 1.020, "al": 0.0, "su": 0.0, "rbc": "normal", "pc": "normal",
    "pcc": "notpresent", "ba": "notpresent", "bgr": 120.0, "bu": 35.0, "sc": 1.2, "sod": 140.0,
    "pot": 4.5, "hemo": 15.0, "pcv": 44, "wc": 8000, "rc": 5.0, "htn": "no", "dm": "no",
    "cad": "no", "appet": "good", "pe": "no", "ane": "no"
}

# Define the input features expected by the kidney model
kidney_input_features = ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", "sod", "pot",
                         "hemo", "pcv", "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane"]

# Symptoms list for the symptoms-based prediction model
symptoms_list = ["itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", "chills",
                 "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting",
                 "burning_micturition", "spotting_urination", "fatigue", "weight_gain", "anxiety",
                 "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness", "lethargy", "patches_in_throat",
                 "irregular_sugar_level", "cough", "high_fever", "sunken_eyes", "breathlessness", "sweating",
                 "dehydration", "indigestion", "headache", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite",
                 "pain_behind_the_eyes", "back_pain", "constipation", "abdominal_pain", "diarrhoea", "mild_fever",
                 "yellow_urine", "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", "swelling_of_stomach",
                 "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision", "phlegm", "throat_irritation",
                 "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", "chest_pain", "weakness_in_limbs",
                 "fast_heart_rate", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool",
                 "irritation_in_anus", "neck_pain", "dizziness", "cramps", "bruising", "obesity", "swollen_legs",
                 "swollen_blood_vessels", "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails",
                 "swollen_extremeties", "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips",
                 "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints",
                 "movement_stiffness", "spinning_movements", "loss_of_balance", "unsteadiness", "weakness_of_one_body_side",
                 "loss_of_smell", "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine",
                 "passage_of_gases", "internal_itching", "toxic_look_(typhos)", "depression", "irritability",
                 "muscle_pain", "altered_sensorium", "red_spots_over_body", "belly_pain", "abnormal_menstruation",
                 "dischromic_patches", "watering_from_eyes", "increased_appetite", "polyuria", "family_history",
                 "mucoid_sputum", "rusty_sputum", "lack_of_concentration", "visual_disturbances",
                 "receiving_blood_transfusion", "receiving_unsterile_injections", "coma", "stomach_bleeding",
                 "distention_of_abdomen", "history_of_alcohol_consumption", "fluid_overload", "blood_in_sputum",
                 "prominent_veins_on_calf", "palpitations", "painful_walking", "pus_filled_pimples", "blackheads",
                 "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails",
                 "blister", "red_sore_around_nose", "yellow_crust_ooze"]

# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/kidney', methods=['GET', 'POST'])
def kidney():
    if request.method == 'POST':
        # Get user input from form
        user_input = {}
        for feature in kidney_input_features:
            value = request.form.get(feature, None)
            if value is None or value == '':
                user_input[feature] = kidney_default_values[feature]
            else:
                user_input[feature] = value

        # Convert categorical data to numerical format as needed
        user_input['rbc'] = 1 if user_input['rbc'] == 'normal' else 0
        user_input['pc'] = 1 if user_input['pc'] == 'normal' else 0
        user_input['pcc'] = 1 if user_input['pcc'] == 'present' else 0
        user_input['ba'] = 1 if user_input['ba'] == 'present' else 0
        user_input['htn'] = 1 if user_input['htn'] == 'yes' else 0
        user_input['dm'] = 1 if user_input['dm'] == 'yes' else 0
        user_input['cad'] = 1 if user_input['cad'] == 'yes' else 0
        user_input['appet'] = 1 if user_input['appet'] == 'good' else 0
        user_input['pe'] = 1 if user_input['pe'] == 'yes' else 0
        user_input['ane'] = 1 if user_input['ane'] == 'yes' else 0

        # Prepare the feature vector for prediction
        feature_vector = np.array([float(user_input[feature]) for feature in kidney_input_features]).reshape(1, -1)
        
        # Make prediction
        prediction = kidney_model.predict(feature_vector)
        
        # Convert prediction to a human-readable format
        prediction_text = 'CKD' if prediction[0] == 1 else 'Not CKD'
        
        return render_template('kidney.html', prediction=prediction_text, kidney_input_features=kidney_input_features)

    return render_template('kidney.html', kidney_input_features=kidney_input_features)

@app.route('/symptoms', methods=['GET', 'POST'])
def symptoms_view():
    if request.method == 'POST':
        user_input_vector = [0] * len(symptoms_list)
        selected_symptoms = request.form.getlist('symptoms')
        
        for symptom in selected_symptoms:
            if symptom in symptoms_list:
                idx = symptoms_list.index(symptom)
                user_input_vector[idx] = 1
        
        prediction = symptoms_model.predict([user_input_vector])
        
        return render_template('symptoms.html', prediction=prediction[0], symptoms_list=symptoms_list)
    
    return render_template('symptoms.html', symptoms_list=symptoms_list)

# Route for breast cancer prediction using an uploaded image
@app.route('/breast')
def breast():
    return render_template('breast.html')

@app.route('/breast_cancer_prediction', methods=['POST'])
def breast_cancer_prediction():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Save the uploaded file temporarily
        file_path = '/tmp/uploaded_image.jpg'  # Or any other temporary location
        file.save(file_path)
        
        # Load the image and ensure it has 3 channels (RGB)
        img = Image.open(file_path)
        img = img.convert('RGB')
        
        # Resize the image to match the input shape expected by the model
        img = img.resize((28, 28))
        
        # Convert the image to a numpy array and preprocess it
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize pixel values
        
        # Convert to grayscale
        img_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        
        # Reshape to match the input shape of the model
        img_array = img_array.reshape(-1, 28, 28, 1)
        
        # Make predictions using the breast cancer model
        prediction = breast_cancer_model.predict(img_array)
        
        # Assuming the model outputs probabilities for different classes, you can
        # get the class with the highest probability
        predicted_class_index = np.argmax(prediction, axis=1)
        
        # Define class labels
        class_labels = ["benign", "malignant", "no_tumour"]
        
        # Map predicted class to human-readable label
        predicted_class_label = class_labels[predicted_class_index[0]]
        
        return jsonify({'prediction': predicted_class_label}), 200
    else:
        return jsonify({'error': 'Invalid file format'}), 400

# Route for brain tumor prediction using an uploaded image
@app.route('/brain', methods=['GET', 'POST'])
def brain():
    if request.method == 'POST':
        return render_template('brain.html')
    return render_template('brain.html')

@app.route('/brain_tumor_prediction', methods=['POST'])
def brain_tumor_prediction():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Save the uploaded file temporarily
        file_path = '/tmp/uploaded_image.jpg'  # Or any other temporary location
        file.save(file_path)
        
        # Load the image and ensure it has 3 channels (RGB)
        img = Image.open(file_path)
        img = img.convert('RGB')
        
        # Preprocess the image
        img = img.resize((224, 224))  # Resize to match the input shape expected by the model
        img_array = np.array(img)  # Convert image to numpy array
        img_array = img_array / 255.0  # Normalize pixel values
        
        # Make predictions using the brain tumor model
        prediction = brain_tumor_model.predict(np.expand_dims(img_array, axis=0))
        
        # Assuming the model outputs probabilities for different classes, you can
        # get the class with the highest probability
        predicted_class_index = np.argmax(prediction)
        
        # Define class labels
        class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']
        
        # Map predicted class to human-readable label
        predicted_class_label = class_labels[predicted_class_index]
        
        return jsonify({'prediction': predicted_class_label}), 200
    else:
        return jsonify({'error': 'Invalid file format'}), 400


model = load_model('kidney_image.h5')

# Define a function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')  # Ensure RGB channels
    img = img.resize((26, 26))  # Resize to match the model's input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define a function to make predictions on the image
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    return prediction

@app.route('/kidney_image', methods=['GET', 'POST'])
def kidney_image():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            # Save the uploaded file temporarily
            file_path = '/tmp/uploaded_image.jpg'  # Or any other temporary location
            file.save(file_path)
            # Make prediction using the uploaded image
            prediction = predict_image(file_path)
            # Define the labels
            labels = ['Normal', 'Cyst', 'Stone', 'Tumor']
            # Get the index of the class with the highest score
            predicted_class_index = np.argmax(prediction)
            # Get the corresponding label
            predicted_label = labels[predicted_class_index]
            return jsonify({'prediction': predicted_label}), 200

    return render_template('kidney_image.html')

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import logging


# Load the lung cancer model once at the start
lung_cancer_model = load_model('saved_model.h5')

# Print the model summary to check the expected input shape
lung_cancer_model.summary()

# Function to preprocess the image
def preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image array
    return img_array

# Function to predict the class of the image
def predict_image(image_path):
    # Get the target size from the model input shape
    target_size = (lung_cancer_model.input_shape[1], lung_cancer_model.input_shape[2])
    preprocessed_image = preprocess_image(image_path, target_size)
    prediction = lung_cancer_model.predict(preprocessed_image)
    return prediction
import os
@app.route('/lung')
def lung():
    return render_template('lung_cancer.html')

@app.route('/lung_cancer_prediction', methods=['POST'])
def lung_cancer_prediction():
    try:
        if 'file' not in request.files:
            raise ValueError('No file uploaded')

        file = request.files['file']

        if file.filename == '':
            raise ValueError('No selected file')

        if file:
            file_path = '/tmp/uploaded_image.jpg'  # Or any other temporary location
            file.save(file_path)

            prediction = predict_image(file_path)
            predicted_class_index = np.argmax(prediction, axis=1)
            class_labels = ["benign", "malignant", "no tumour"]
            predicted_class_label = class_labels[predicted_class_index[0]]

            return jsonify({'prediction': predicted_class_label}), 200
        else:
            raise ValueError('Invalid file format')
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

model = load_model('my_model.h5')

# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image array
    return img_array

# Route for pneumonia prediction
pneumonia_model = load_model('my_model.h5')

# Function to preprocess the image
def preprocess_pneumonia_image(image_path, target_size):
        img = image.load_img(image_path, target_size=target_size)  # Adjust target size according to your model input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image array
        return img_array

# Function to predict the class of the image
def predict_pneumonia(image_path):
    # Get the target size from the model input shape
        target_size = (pneumonia_model.input_shape[1], pneumonia_model.input_shape[2])
        preprocessed_image = preprocess_image(image_path, target_size)
        prediction = pneumonia_model.predict(preprocessed_image)
        return prediction

@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')

@app.route('/pneumonia_prediction', methods=['POST'])
def pneumonia_prediction():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        file_path = '/tmp/uploaded_image.jpg'
        file.save(file_path)
        
        prediction = predict_pneumonia(file_path)
        
        # Assuming the model outputs probabilities for different classes, get the class with the highest probability
        predicted_class_index = np.argmax(prediction, axis=1)
        
        # Define class labels (adjust according to your model)
        class_labels = ["Normal", "Pneumonia"]
        
        # Map predicted class to human-readable label
        predicted_class_label = class_labels[predicted_class_index[0]]
        
        return jsonify({'prediction': predicted_class_label})

if __name__ == '__main__':
    app.run(debug=True)