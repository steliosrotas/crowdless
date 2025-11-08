#!/usr/bin/env python3
import argparse
import json
import math
import sys
import pickle
from pathlib import Path
from typing import Dict, Any

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("This script requires 'pandas' and 'numpy'. Please install them:", file=sys.stderr)
    print("  pip install pandas numpy", file=sys.stderr)
    sys.exit(1)

# Directory where your area JSON files (e.g., Syntagma.json) are stored
AREAS_DIR = Path("./areas_output")

# "Trained" database file. This script will create it.
WEATHER_DB_FILE = AREAS_DIR / "weather_database.pkl"

# "Model" file. This just stores the simple rules we define.
WEATHER_MODEL_FILE = AREAS_DIR / "weather_comfort_model.json"

# --- Model Logic ---

def predict_score_from_weather(temp: float, humidity: float, model: Dict[str, float]) -> float:
    """
    Calculates a crowdness score based on weather "comfort".
    
    The rule is:
    - Score is 100 at the 'ideal' temperature and 0 at the 'bad' ranges.
    - Score is 100 at 'ideal' humidity and 0 at 'bad' humidity.
    - The final score is a weighted average.
    """
    
    # --- Temperature Score (0-1) ---
    # Uses a simple triangular/linear falloff
    ideal_temp = model['ideal_temp']
    temp_range = model['temp_range'] # How "wide" the comfort peak is
    
    # 1.0 at ideal_temp, 0.0 at ideal_temp +/- temp_range
    temp_score = max(0.0, 1.0 - abs(temp - ideal_temp) / temp_range)

    # --- Humidity Score (0-1) ---
    # Assumes lower humidity is generally better, but not *too* low.
    ideal_humidity = model['ideal_humidity']
    humidity_range = model['humidity_range']
    
    # 1.0 at ideal_humidity, 0.0 at ideal_humidity + humidity_range
    # We only penalize *high* humidity
    if humidity > ideal_humidity:
        humidity_score = max(0.0, 1.0 - (humidity - ideal_humidity) / humidity_range)
    else:
        # If humidity is at or below ideal, it's a perfect score
        humidity_score = 1.0 
        
    # --- Final Weighted Score (0-100) ---
    temp_weight = model['temp_weight']
    humidity_weight = model['humidity_weight']
    
    final_score = (temp_score * temp_weight + humidity_score * humidity_weight) * 100.0
    
    return final_score


# --- Command Functions ---

def cmd_train(args):
    """
    Reads all JSON files in AREAS_DIR, extracts weather data, and
    saves it to a fast-lookup database (WEATHER_DB_FILE).
    """
    print(f"Starting 'training' by reading all JSONs from {AREAS_DIR}...")
    
    all_data = []
    
    # Find all area JSON files
    json_files = list(AREAS_DIR.glob("*.json"))
    
    # Files to ignore (e.g., our own output)
    skip_files = {WEATHER_DB_FILE.name, WEATHER_MODEL_FILE.name, "crowdness_scores.json"}
    
    if not json_files:
        print(f"Error: No JSON files found in {AREAS_DIR}", file=sys.stderr)
        print("Please run the data fetching script first.", file=sys.stderr)
        return

    for f_path in json_files:
        if f_path.name in skip_files:
            continue
            
        print(f"  Reading {f_path.name}...")
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            area_name = data.get("area")
            if not area_name:
                print(f"    Warning: Skipping {f_path.name} (no 'area' key).")
                continue

            for item in data.get("items", []):
                dt_str = item.get("datetime")
                temp = item.get("temperature_celsius")
                humidity = item.get("relative_humidity_percent")

                # Only add rows that have all our required data
                if dt_str and temp is not None and humidity is not None:
                    all_data.append({
                        "area": area_name,
                        "datetime": pd.to_datetime(dt_str, utc=True),
                        "temp": float(temp),
                        "humidity": float(humidity)
                    })
                
        except Exception as e:
            print(f"    Error reading {f_path.name}: {e}", file=sys.stderr)

    if not all_data:
        print("Error: No valid weather data (temperature/humidity) found in JSON files.", file=sys.stderr)
        print("Please re-run the data fetching script from the previous step.", file=sys.stderr)
        return

    # Convert to DataFrame for efficient lookups
    df = pd.DataFrame(all_data)
    df = df.dropna().drop_duplicates(subset=["area", "datetime"])
    
    # Set datetime as the index for fast time-based searching
    df = df.set_index('datetime')
    
    # Save the "trained" database
    df.to_pickle(WEATHER_DB_FILE)
    
    print(f"\nTraining complete. Processed {len(all_data)} items.")
    print(f"Saved {len(df)} unique data points to {WEATHER_DB_FILE}")

    # --- Define and save the "comfort model" rules ---
    # These can be tuned!
    model_rules = {
        # Temp: Ideal 22C. Score hits 0 at 7C (22-15) and 37C (22+15).
        "ideal_temp": 22.0,
        "temp_range": 15.0, 
        
        # Humidity: Ideal 45%. Score hits 0 at 95% (45+50).
        "ideal_humidity": 45.0,
        "humidity_range": 50.0,
        
        # Weights: Temperature is more important than humidity.
        "temp_weight": 0.7,
        "humidity_weight": 0.3
    }
    
    with open(WEATHER_MODEL_FILE, 'w', encoding='utf-8') as f:
        json.dump(model_rules, f, indent=2)
        
    print(f"Saved comfort model rules to {WEATHER_MODEL_FILE}")


def cmd_predict(args):
    """
    Predicts crowdness for a given area and datetime by finding
    the closest weather data and applying the comfort model.
    """
    # --- 1. Load Model and Database ---
    if not WEATHER_DB_FILE.exists() or not WEATHER_MODEL_FILE.exists():
        print("Error: Model files not found. Please run the 'train' command first:", file=sys.stderr)
        print(f"  python {sys.argv[0]} train", file=sys.stderr)
        return

    try:
        db = pd.read_pickle(WEATHER_DB_FILE)
        with open(WEATHER_MODEL_FILE, 'r', encoding='utf-8') as f:
            model = json.load(f)
    except Exception as e:
        print(f"Error loading model files: {e}", file=sys.stderr)
        return

    # --- 2. Parse Inputs ---
    try:
        input_dt = pd.to_datetime(args.datetime, utc=True)
    except Exception as e:
        print(f"Error: Could not parse datetime '{args.datetime}'", file=sys.stderr)
        print("Please use a standard format (e.g., '2025-05-01T14:00:00Z')", file=sys.stderr)
        return
        
    input_area = args.area
    
    # --- 3. Find Closest Data ---
    
    # Filter database for the requested area
    # Using .str.lower() for a case-insensitive match
    area_db = db[db['area'].str.lower() == input_area.lower()]
    
    if area_db.empty:
        print(f"Error: No data found for area '{input_area}'.", file=sys.stderr)
        print(f"Available areas: {db['area'].unique()}", file=sys.stderr)
        return

    # Find the single closest point in time
    # `get_indexer` finds the row index in `area_db` that is 'nearest' to our `input_dt`
    closest_index = area_db.index.get_indexer([input_dt], method='nearest')[0]
    
    if closest_index == -1:
        print(f"Error: Could not find any data for '{input_area}'.", file=sys.stderr)
        return

    closest_row = area_db.iloc[closest_index]
    closest_time = closest_row.name # This is the index (the datetime)
    
    # --- 4. Get Weather and Predict ---
    temp = closest_row['temp']
    humidity = closest_row['humidity']
    
    # Calculate the score
    score = predict_score_from_weather(temp, humidity, model)
    
    # --- 5. Show Results ---
    time_diff = abs(closest_time - input_dt)
    
    print("--- Prediction Input ---")
    print(f"  Area:     {input_area} (found '{closest_row['area']}')")
    print(f"  Time:     {input_dt.isoformat()}")
    print("\n--- Closest Data Found ---")
    print(f"  Time:     {closest_time.isoformat()} (Difference: {time_diff})")
    print(f"  Temp:     {temp:.2f} °C")
    print(f"  Humidity: {humidity:.2f} %")
    print("\n--- Result ---")
    print(f"  Predicted crowdness_score: {score:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train or predict crowdness based on weather.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    
    # --- Train Command ---
    p_train = subparsers.add_parser(
        "train",
        help="Builds the weather database from JSON files in ./areas_output"
    )
    p_train.set_defaults(func=cmd_train)
    
    # --- Predict Command ---
    p_pred = subparsers.add_parser(
        "predict",
        help="Predict crowdness for a specific area and time"
    )
    p_pred.add_argument(
        "--area",
        type=str,
        required=True,
        help="Name of the area (e.g., Syntagma, Kallithea)"
    )
    p_pred.add_argument(
        "--datetime",
        type=str,
        required=True,
        help="Timestamp in ISO format (e.g., '2025-07-01T13:00:00Z')"
    )
    p_pred.set_defaults(func=cmd_predict)
    
    # --- Run ---
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()