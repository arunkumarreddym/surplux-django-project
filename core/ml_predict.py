import os
import joblib

MODEL_PATH = "ml/food_model.pkl"

model = None

# Load model only if file exists
if os.path.exists(MODEL_PATH):
    print("Loading ML model...")
    model = joblib.load(MODEL_PATH)
else:
    print("ML model not found. Running without model.")

def predict_shelf_life(data):
    if model is None:
        return "Model not available in CI environment"

    prediction = model.predict([data])
    return prediction