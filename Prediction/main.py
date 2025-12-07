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

    try:
        api_key = API_KEY_
        location = config["openweather"]["location"]
        units = config["openweather"]["units"]

        # 1. Prepare HISTORY Data (past 14 days)
        history = []

        for i in range(14):
            weather = get_weather_data(api_key, location, units)

            history.append({
                "timestamp": (datetime.now() - pd.Timedelta(days=(14 - i))).strftime("%Y-%m-%d"),
                "aqi": np.random.randint(40, 200),  # Replace with actual historical AQI if you have it
                **weather,
                "traffic_level": np.random.randint(1, 5),
                "dust_road_flag": np.random.randint(0, 2)
            })

        # 2. Prepare FUTURE INPUTS
        future_inputs = []
        for i in range(user.predict_days):
            weather = get_weather_data(api_key, location, units)

            future_inputs.append({
                "timestamp": (datetime.now() + pd.Timedelta(days=i+1)).strftime("%Y-%m-%d"),
                **weather,
                "traffic_level": user.traffic_level,
                "dust_road_flag": user.dust_road_flag
            })

        # Convert to DataFrames
        history_df = pd.DataFrame(history)
        future_df = pd.DataFrame(future_inputs)

        # Add required features (month, dayofyear, dry_season)
        history_df = add_time_features(history_df)
        future_df = add_time_features(future_df)

        # 3. Build Dataset
        X, y, scaler, X_future_scaled = build_dataset(
            history_df, future_df, SEQ_LEN, FEATURE_COLS
        )

        # 4. Build + Train Model
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

        # 5. Predict Future AQI
        predictions = predict_future(
            model, scaler, history_df,
            X_future_scaled, FEATURE_COLS, SEQ_LEN
        )

        # 6. Return Clean JSON Response
        return {
            "status": "success",
            "requested_days": user.predict_days,
            "aqi_predictions": predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
