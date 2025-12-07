import json
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import pandas as pd

from Prediction.weather_api import get_weather_data
from Prediction.data_utils import build_dataset
from Prediction.model import build_model, predict_future


# Load API Keys & Config
with open("api_keys.json") as json_file:
    api_keys = json.load(json_file)
API_KEY_ = api_keys["Weather_API"]["API_key"]

with open("Prediction/config.yaml") as f:
    config = yaml.safe_load(f)

SEQ_LEN = config["lstm"]["sequence_length"]
FEATURE_COLS = config["features"]

app = FastAPI(title="AQI Prediction API")


# Input Schema
class UserInputs(BaseModel):
    traffic_level: int
    dust_road_flag: int
    predict_days: int = 3


# Helper – Add Required Features
def add_time_features(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["month"] = df["timestamp"].dt.month
    df["dayofyear"] = df["timestamp"].dt.dayofyear
    df["dry_season"] = df["month"].isin([11, 12, 1, 2, 3]).astype(int)
    return df


@app.get("/")
def home():
    return {
        "status": "running",
        "message": "AQI Prediction API is active. Visit /docs to test."
    }


# Prediction Endpoint
@app.post("/predict")
def predict_aqi(user: UserInputs):
    api_key = API_KEY_
    location = config["openweather"]["location"]
    units = config["openweather"]["units"]

    # Create 14 days of history
    history = []
    for i in range(14):
        w = get_weather_data(api_key, location, units)

        date = datetime.now() - pd.Timedelta(days=14 - i)
        month = date.month
        dayofyear = date.timetuple().tm_yday
        dry_season = 1 if month in [10, 11, 12, 1, 2] else 0

        history.append({
            "timestamp": date.strftime("%Y-%m-%d"),
            "weather_temp": w["weather_temp"],
            "weather_humidity": w["weather_humidity"],
            "wind_speed": w["wind_speed"],
            "wind_direction": w["wind_direction"],
            "traffic_level": np.random.randint(1, 5),
            "dust_road_flag": np.random.randint(0, 2),
            "month": month,
            "dayofyear": dayofyear,
            "dry_season": dry_season,
            "aqi": np.random.randint(40, 200)
        })

    # Create future inputs
    future_inputs = []
    for i in range(user.predict_days):
        w = get_weather_data(api_key, location, units)

        future_date = datetime.now() + pd.Timedelta(days=i + 1)
        month = future_date.month
        dayofyear = future_date.timetuple().tm_yday
        dry_season = 1 if month in [10, 11, 12, 1, 2] else 0

        future_inputs.append({
            "timestamp": future_date.strftime("%Y-%m-%d"),
            "weather_temp": w["weather_temp"],
            "weather_humidity": w["weather_humidity"],
            "wind_speed": w["wind_speed"],
            "wind_direction": w["wind_direction"],
            "traffic_level": user.traffic_level,
            "dust_road_flag": user.dust_road_flag,
            "month": month,
            "dayofyear": dayofyear,
            "dry_season": dry_season
        })

    # Prepare dataset
    X, y, scaler, X_future_scaled = build_dataset(
        history, future_inputs, SEQ_LEN, FEATURE_COLS
    )

    # Build & Train Model
    model = build_model(
        SEQ_LEN,
        X.shape[2],
        config["lstm"]["lstm_units"],
        config["lstm"]["dense_units"]
    )

    model.fit(
        X, y,
        epochs=config["lstm"]["epochs"],
        batch_size=config["lstm"]["batch_size"],
        verbose=0
    )

    # Predict
    preds = predict_future(
        model, scaler, pd.DataFrame(history),
        X_future_scaled, FEATURE_COLS, SEQ_LEN
    )

    # Convert to normal Python types
    preds_list = [float(x) for x in preds]

    return {"predicted_aqi": preds_list}

