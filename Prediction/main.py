import json
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import pandas as pd

from Prediction.weather_api import get_weather_data
from Prediction.data_utils import build_dataset
from Prediction.model import build_model, predict_future

# Load API keys
with open('api_keys.json') as json_file:
    api_keys = json.load(json_file)
API_KEY_ = api_keys['Weather_API']['API_key']


# Load config
with open("Prediction/config.yaml") as f:
    config = yaml.safe_load(f)

SEQ_LEN = config["lstm"]["sequence_length"]
FEATURE_COLS = config["features"]

app = FastAPI(title="AQI Prediction API")


class UserInputs(BaseModel):
    traffic_level: int
    dust_road_flag: int
    predict_days: int = 3


@app.post("/predict")
def predict_aqi(user: UserInputs):

    api_key = API_KEY_
    location = config["openweather"]["location"]
    units = config["openweather"]["units"]

    # Generate 14 days history
    history = []
    for i in range(14):
        w = get_weather_data(api_key, location, units)
        history.append({
            "timestamp": (datetime.now() - pd.Timedelta(days=14 - i)).strftime("%Y-%m-%d"),
            "aqi": np.random.randint(40, 200),
            **w,
            "traffic_level": np.random.randint(1, 5),
            "dust_road_flag": np.random.randint(0, 2)
        })

    # Future inputs
    future_inputs = []
    for i in range(user.predict_days):
        w = get_weather_data(api_key, location, units)
        future_inputs.append({
            "timestamp": (datetime.now() + pd.Timedelta(days=i+1)).strftime("%Y-%m-%d"),
            **w,
            "traffic_level": user.traffic_level,
            "dust_road_flag": user.dust_road_flag
        })

    # Build dataset
    X, y, scaler, X_future_scaled = build_dataset(history, future_inputs, SEQ_LEN, FEATURE_COLS)

    # Build + Train model
    model = build_model(SEQ_LEN, X.shape[2],
                        config["lstm"]["lstm_units"],
                        config["lstm"]["dense_units"])

    model.fit(X, y, epochs=config["lstm"]["epochs"],
              batch_size=config["lstm"]["batch_size"], verbose=0)

    # Predict
    preds = predict_future(model, scaler, pd.DataFrame(history),
                           X_future_scaled, FEATURE_COLS, SEQ_LEN)

    return {"predicted_aqi": preds}
