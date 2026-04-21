import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'crop_dataset_text.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_export_models():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    print("Generating synthetic target variables...")
    # 1. Generate Synthetic Soil Health Score
    # Start with a base of 20
    score = 20.0
    
    # Penalize extreme pH
    score -= np.abs(df['ph'] - 6.5) * 5
    
    # Reward NPK proximity to generalized optimal levels (capped)
    score += np.clip((df['N'] / 150) * 20, 0, 20)
    score += np.clip((df['P'] / 60) * 20, 0, 20)
    score += np.clip((df['K'] / 200) * 20, 0, 20)
    
    # Penalize high salinity/EC
    score -= df['electrical_conductivity (dS/m)'] * 2
    
    # Add bonus for good organic matter
    score += df['organic_matter (%)'] * 5
    
    # Ensure bounds between 10 and 100
    df['health_score'] = np.clip(score, 10, 100)
    
    # 2. Generate Synthetic Yield (tons)
    # Different crops have different average yields per hectare
    crop_yield_factors = {}
    crops = df['crop'].unique()
    for i, crop in enumerate(crops):
        crop_yield_factors[crop] = np.random.uniform(2.5, 8.5) # random base tons/hectare for crop
        
    df['base_yield'] = df['crop'].map(crop_yield_factors)
    # yield = base * area + some weather randomness
    noise = np.random.normal(1.0, 0.1, len(df))
    df['yield'] = df['base_yield'] * df['area (hectares)'] * noise
    df['yield'] = np.clip(df['yield'], 0, None)
    
    print("Data preparation complete.")
    
    # --- MODEL 1: CROP RECOMMENDATION (Classification) ---
    print("\nTraining Crop Recommendation Model...")
    X_crop = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y_crop = df['crop']
    
    crop_rec_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    crop_rec_pipeline.fit(X_crop, y_crop)
    joblib.dump(crop_rec_pipeline, os.path.join(MODELS_DIR, 'rf_crop_rec_model.joblib'))
    print("Crop Recommendation Model saved.")
    
    # --- MODEL 2: SOIL HEALTH ANALYSIS (Regression) ---
    print("\nTraining Soil Health Model...")
    X_soil = df[['N', 'P', 'K', 'ph', 'electrical_conductivity (dS/m)', 'organic_matter (%)']]
    y_soil = df['health_score']
    
    soil_health_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
    ])
    soil_health_pipeline.fit(X_soil, y_soil)
    joblib.dump(soil_health_pipeline, os.path.join(MODELS_DIR, 'rf_soil_health_model.joblib'))
    print("Soil Health Model saved.")
    
    # --- MODEL 3: CROP YIELD PREDICTION (Regression) ---
    print("\nTraining Crop Yield Model...")
    X_yield = df[['crop', 'area (hectares)', 'rainfall', 'temperature']]
    y_yield = df['yield']
    
    # We need a ColumnTransformer to handle the categorical 'crop' feature (One-Hot Encoding)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['crop']),
            ('num', StandardScaler(), ['area (hectares)', 'rainfall', 'temperature'])
        ])
        
    yield_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
    ])
    yield_pipeline.fit(X_yield, y_yield)
    joblib.dump(yield_pipeline, os.path.join(MODELS_DIR, 'rf_yield_pipeline.joblib'))
    print("Crop Yield Model saved.")
    
    print("\nAll models trained and saved to the models/ directory!")

if __name__ == "__main__":
    train_and_export_models()
