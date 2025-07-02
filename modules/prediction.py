import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
import google.generativeai as genai
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_PATH = os.path.join(DATA_DIR, "sample_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "health_model.pkl")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

GENAI_API_KEY = "AIzaSyBue5KqNeZoqOD9L8r2d0oQPEtSd3qYGM8" 
genai.configure(api_key=GENAI_API_KEY)

def validate_inputs(activity_level: int, genetic_risk: int):
    """Validate user inputs"""
    errors = []
    if not (1 <= activity_level <= 5):
        errors.append("Activity level must be between 1 and 5")
    if genetic_risk not in {0, 1}:
        errors.append("Genetic risk must be 0 (low) or 1 (high)")
    return errors

def train_model():
    """Train and save the health risk model"""
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
            
        data = pd.read_csv(DATA_PATH)
        if data.empty:
            raise ValueError("Dataset is empty")
            
        required_columns = ['calories', 'activity_level', 'water_intake', 'genetic_risk', 'disease']
        if not all(col in data.columns for col in required_columns):
            missing = set(required_columns) - set(data.columns)
            raise ValueError(f"Missing columns in dataset: {missing}")

        # Prepare data
        X = data[required_columns[:-1]]
        y = data['disease']
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained successfully. Accuracy: {accuracy:.2%}")
        joblib.dump(model, MODEL_PATH)
        return model
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        traceback.print_exc()
        return None

def predict_health_risk(input_data):
    """Predict health risk with error handling"""
    try:
        if not os.path.exists(MODEL_PATH):
            print("Model not found. Training new model...")
            model = train_model()
            if not model:
                return -1 
        else:
            model = joblib.load(MODEL_PATH)
            
        return model.predict(np.array([input_data]))[0]
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return -1

def predict_potential_diseases(genetic_info, activity_level):
    """Get AI health analysis with safety controls"""
    if not genetic_info.strip():
        return "Please provide valid genetic information for analysis"
        
    safety_settings = {
        'HARM_CATEGORY_HARASSMENT': 'BLOCK_ONLY_HIGH',
        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_ONLY_HIGH',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_ONLY_HIGH',
        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_ONLY_HIGH',
    }

    prompt = f"""Analyze health risks for:
    - Genetic predispositions: {genetic_info}
    - Activity level: {activity_level}/5 (1 = sedentary, 5 = very active)
    
    Provide:
    1. Top 3 potential health concerns (max)
    2. Risk level for each (Low/Medium/High)
    3. Brief prevention strategies
    4. Clear disclaimer that this is not medical advice
    
    Use markdown-style bullets. Be conservative in estimates."""
    
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')  
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config={'temperature': 0.4}
        )
        return response.text
    except Exception as e:
        print(f"AI analysis failed: {str(e)}")
        print(f"Full error details: {traceback.format_exc()}")  
        return """**Fallback Health Analysis:**
1. **Potential Health Concerns:**
   - Diabetes (Medium Risk)
   - Heart Disease (Medium Risk)
   - Hypertension (Low Risk)

2. **Prevention Strategies:**
   - Maintain a balanced diet.
   - Exercise regularly.
   - Consult a healthcare professional for personalized advice.

3. **Disclaimer:**
   This is a generic analysis and not a substitute for medical advice."""

def list_supported_models():
    """List all supported models and their methods"""
    try:
        models = genai.list_models()
        print("Supported Models:")
        for model in models:
            print(f"- Name: {model.name}")
            print(f"  Supported Methods: {model.supported_generation_methods}")
            print()
    except Exception as e:
        print(f"Failed to list models: {str(e)}")

if __name__ == '__main__':
    print("=== Health Risk Assessment System ===")
    print("Using provided Gemini API key for analysis")
    
   
    list_supported_models()
    
    try:
        calories = float(input("Enter daily calories: "))
        activity = int(input("Activity level (1-5): "))
        water = float(input("Water intake (liters): "))
        genetic = int(input("Genetic risk (0/1): "))
        
      
        if errors := validate_inputs(activity, genetic):
            print("\nInput errors:")
            for error in errors:
                print(f"- {error}")
            exit(1)
            
        risk = predict_health_risk([calories, activity, water, genetic])
        print("\n--- Machine Learning Prediction ---")
        if risk == -1:
            print("Prediction unavailable")
        else:
            print(f"Health risk: {'High' if risk == 1 else 'Low'} ({risk})")
            
        genetics = input("\nEnter family medical history (e.g., diabetes): ")
        print("\n--- AI Health Analysis ---")
        analysis = predict_potential_diseases(genetics.strip(), activity)
        print(analysis if analysis else "No analysis generated")
        
    except ValueError:
        print("Invalid input format")
    except KeyboardInterrupt:
        print("\nOperation cancelled")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
