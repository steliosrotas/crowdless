import pandas as pd
import numpy as np
import json
from flask import Flask, request, jsonify
from pathlib import Path

# --- CONFIGURATION (Must match your training script) ---
AREAS_DIR = Path("./areas_output")
WEATHER_DB_FILE = AREAS_DIR / "weather_database.pkl"
WEATHER_MODEL_FILE = AREAS_DIR / "weather_comfort_model.json"

# --- CORE PREDICTION LOGIC (Copied from weather_model.py) ---

def predict_score_from_weather(temp: float, humidity: float, model: dict) -> float:
    """Calculates a crowdness score based on weather "comfort" using model rules."""
    
    # --- Temperature Score (0-1) ---
    ideal_temp = model['ideal_temp']
    temp_range = model['temp_range'] 
    temp_score = max(0.0, 1.0 - abs(temp - ideal_temp) / temp_range)

    # --- Humidity Score (0-1) ---
    ideal_humidity = model['ideal_humidity']
    humidity_range = model['humidity_range']
    if humidity > ideal_humidity:
        humidity_score = max(0.0, 1.0 - (humidity - ideal_humidity) / humidity_range)
    else:
        humidity_score = 1.0 
        
    # --- Final Weighted Score (0-100) ---
    temp_weight = model['temp_weight']
    humidity_weight = model['humidity_weight']
    
    final_score = (temp_score * temp_weight + humidity_score * humidity_weight) * 100.0
    
    return float(f"{final_score:.2f}")


# --- FLASK API SETUP ---

app = Flask(__name__)
weather_db = None
comfort_model = None

def load_model_assets():
    """Loads the database and comfort model rules into global memory once."""
    global weather_db, comfort_model
    try:
        # Load the weather data DataFrame (the "trained" database)
        weather_db = pd.read_pickle(WEATHER_DB_FILE)
        
        # Load the comfort model rules (the prediction parameters)
        with open(WEATHER_MODEL_FILE, 'r', encoding='utf-8') as f:
            comfort_model = json.load(f)
            
        print(f"✅ Model assets loaded successfully from {AREAS_DIR}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model assets. Did you run 'python weather_model.py train'?", file=sys.stderr)
        print(f"   Details: {e}", file=sys.stderr)
        exit(1)

# Load the assets immediately when the server starts
load_model_assets()

@app.route('/predict', methods=['GET'])
def predict():
    """
    API endpoint to predict crowdness.
    Requires 'area' and 'datetime' URL parameters.
    """
    area = request.args.get('area')
    datetime_str = request.args.get('datetime')

    if not area or not datetime_str:
        return jsonify({
            "error": "Missing parameters. Requires 'area' and 'datetime'.",
            "example": "/predict?area=Syntagma&datetime=2026-07-18T15:00:00Z"
        }), 400

    try:
        input_dt = pd.to_datetime(datetime_str, utc=True)
    except Exception:
        return jsonify({"error": f"Invalid datetime format: {datetime_str}"}), 400

    # 1. Filter database for the requested area
    area_db = weather_db[weather_db['area'].str.lower() == area.lower()]

    if area_db.empty:
        return jsonify({
            "error": f"No data found for area '{area}'.",
            "available_areas": weather_db['area'].unique().tolist()
        }), 404

    # 2. Find the single closest point in time
    closest_index = area_db.index.get_indexer([input_dt], method='nearest')[0]
    closest_row = area_db.iloc[closest_index]
    
    # 3. Get simulated weather data
    temp = closest_row['temp']
    humidity = closest_row['humidity']
    
    # 4. Predict the score
    score = predict_score_from_weather(temp, humidity, comfort_model)
    
    # 5. Return the result
    return jsonify({
        "input_area": area,
        "input_datetime": datetime_str,
        "closest_data_used": {
            "datetime": closest_row.name.isoformat(),
            "temperature_celsius": float(f"{temp:.2f}"),
            "relative_humidity_percent": float(f"{humidity:.2f}")
        },
        "crowdness_score": score
    })

if __name__ == '__main__':
    print("Starting Crowdness Prediction API...")
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)