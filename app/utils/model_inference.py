import time
import random
import os
import joblib
import pandas as pd

import io
import numpy as np
from PIL import Image

class RealCNNModel:
    def __init__(self):
        self.model = None
        self.class_names = [
            'Apple leaf scab disease',
            'Corn leaf rust disease',
            'Crop leaf calcium deficiency',
            'Crop leaf magnesium deficiency',
            'Healthy crop leaf',
            'Healthy NPK crop leaf',
            'Nitrogen (N) deficiency',
            'Phosphorus (P) deficiency',
            'Potassium (K) deficiency',
            'Potato leaf late blight disease',
            'Tomato leaf early blight disease'
        ]
        self.error_msg = None
        try:
            from tensorflow.keras.models import load_model
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, 'models', 'disease_cnn_model.h5')
            self.model = load_model(model_path)
        except ImportError:
            self.error_msg = "TensorFlow missing. Run 'pip install tensorflow'"
        except Exception as e:
            self.error_msg = f"Model load error: {str(e)}"

    def predict(self, image_data):
        if not self.model:
            return self.error_msg or "Model failed to load"
            
        from tensorflow.keras.preprocessing.image import img_to_array
            
        img = Image.open(io.BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((150, 150))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Rescale matching training generator
        
        predictions = self.model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        return self.class_names[predicted_class_index]

class TrainedMLModels:
    def __init__(self):
        # Define paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.models_dir = os.path.join(self.base_dir, 'models')
        
        # Load models
        try:
            self.crop_rec_model = joblib.load(os.path.join(self.models_dir, 'rf_crop_rec_model.joblib'))
            self.yield_model = joblib.load(os.path.join(self.models_dir, 'rf_yield_pipeline.joblib'))
            self.soil_health_model = joblib.load(os.path.join(self.models_dir, 'rf_soil_health_model.joblib'))
        except FileNotFoundError:
            print("Warning: Model files not found. Ensure train_models.py has been run.")
            self.crop_rec_model = None
            self.yield_model = None
            self.soil_health_model = None

    def predict_crop(self, n, p, k, temp, humidity, ph, rainfall):
        if not self.crop_rec_model:
            return "maize" # fallback
            
        # Features: 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'
        input_data = pd.DataFrame([[n, p, k, temp, humidity, ph, rainfall]], 
                                  columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        prediction = self.crop_rec_model.predict(input_data)[0]
        return prediction
        
    def predict_yield(self, crop, season, area, rainfall, temp):
        if not self.yield_model:
            return 2.5 * area # fallback
            
        # Features: 'crop', 'area (hectares)', 'rainfall', 'temperature'
        input_data = pd.DataFrame([[crop, area, rainfall, temp]], 
                                  columns=['crop', 'area (hectares)', 'rainfall', 'temperature'])
        try:
            prediction = self.yield_model.predict(input_data)[0]
            return round(prediction, 2)
        except Exception as e:
            # The crop might not have been in the training set
            return round(2.5 * area, 2)
        
    def analyze_soil(self, n, p, k, ph, ec, organic_carbon):
        if not self.soil_health_model:
            return 70, "Moderate" # fallback
            
        # Features: 'N', 'P', 'K', 'ph', 'electrical_conductivity (dS/m)', 'organic_matter (%)'
        input_data = pd.DataFrame([[n, p, k, ph, ec, organic_carbon]], 
                                  columns=['N', 'P', 'K', 'ph', 'electrical_conductivity (dS/m)', 'organic_matter (%)'])
        prediction = self.soil_health_model.predict(input_data)[0]
        health_score = int(prediction)
        
        status = "Poor"
        if health_score >= 80:
            status = "Excellent"
        elif health_score >= 60:
            status = "Moderate"
            
        return health_score, status


def analyze_leaf(image_data):
    model = RealCNNModel()
    return model.predict(image_data)

def predict_crop_and_fertilizer(n, p, k, temp, humidity, ph, rainfall):
    model = TrainedMLModels()
    crop = model.predict_crop(n, p, k, temp, humidity, ph, rainfall)
    return {
        "crop": crop,
        "advice": f"For {crop}, adjust NPK levels closely monitoring weather and stage."
    }

def predict_crop_yield(crop, season, area, rainfall, temp):
    model = TrainedMLModels()
    return model.predict_yield(crop, season, area, rainfall, temp)

def assess_soil_health(n, p, k, ph, ec, organic_carbon):
    model = TrainedMLModels()
    return model.analyze_soil(n, p, k, ph, ec, organic_carbon)
