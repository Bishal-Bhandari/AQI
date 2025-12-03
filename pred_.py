import json
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
import yaml
import requests

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Load API keys
with open('api_keys.json') as json_file:
    api_keys = json.load(json_file)
API_KEY_ = api_keys['Weather_API']['API_key']

SEQ_LEN = config["lstm"]["sequence_length"]
LSTM_UNITS = config["lstm"]["lstm_units"]
DENSE_UNITS = config["lstm"]["dense_units"]
EPOCHS = config["lstm"]["epochs"]
BATCH_SIZE = config["lstm"]["batch_size"]

HISTORY_DAYS = config["data"]["history_days"]
PREDICT_DAYS = config["data"]["predict_days"]

FEATURE_COLS = config["features"]

# opeanweather api request
def get_weather_from_openweather(api_key, location, units="metric"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units={units}"

    resp = requests.get(url)
    data = resp.json()

    return {
        "weather_temp": data["main"]["temp"],
        "weather_humidity": data["main"]["humidity"],
        "wind_speed": data["wind"]["speed"],
        "wind_direction": data["wind"]["deg"]
    }

# Feature engineering
def add_features(df):
    df["month"] = pd.to_datetime(df["timestamp"]).dt.month
    df["dayofyear"] = pd.to_datetime(df["timestamp"]).dt.dayofyear
    df["dry_season"] = df["month"].apply(lambda m: 1 if 2 <= m <= 4 else 0)
    return df


# Build dataset for LSTM
def build_dataset(history, future_inputs, seq_len):
    # Convert input lists to DataFrames
    hist_df = add_features(pd.DataFrame(history))
    future_df = add_features(pd.DataFrame(future_inputs))

    # Scale features
    scaler = MinMaxScaler()
    scaled_hist = scaler.fit_transform(hist_df[FEATURE_COLS])

    # Build training sequences
    X, y = [], []
    for i in range(len(scaled_hist) - seq_len):
        X.append(scaled_hist[i:i+seq_len])
        y.append(hist_df["aqi"].iloc[i+seq_len])

    X, y = np.array(X), np.array(y)

    # Scale future inputs
    X_future = scaler.transform(future_df[FEATURE_COLS])

    return X, y, scaler, X_future


# build LSTM model
def build_model(seq_len, num_features):
    model = Sequential([
        Input(shape=(seq_len, num_features)),
        LSTM(LSTM_UNITS, activation="tanh"),
        Dense(DENSE_UNITS, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# perform prediction
def predict_next_days(model, scaler, history_df, future_scaled, seq_len):
    hist_df = add_features(pd.DataFrame(history_df))

    last_seq_raw = hist_df[FEATURE_COLS].tail(seq_len)

    last_seq = scaler.transform(last_seq_raw)
    last_seq = last_seq.reshape(1, seq_len, len(FEATURE_COLS))

    predictions = []
    seq = last_seq.copy()

    for i in range(len(future_scaled)):
        # Predict next timestep
        pred = model.predict(seq, verbose=0)[0][0]
        predictions.append(pred)

        # Append next feature vector
        new_input = future_scaled[i].reshape(1, 1, len(FEATURE_COLS))

        # Slide the window
        seq = np.concatenate([seq[:, 1:, :], new_input], axis=1)

    return predictions



if __name__ == "__main__":

    # load API settings
    api_key = API_KEY_
    location = config["openweather"]["location"]
    units = config["openweather"]["units"]

    # generate synthetic AQI + weather for history
    history = []
    for day in range(HISTORY_DAYS):
        weather = get_weather_from_openweather(api_key, location, units)

        history.append({
            "timestamp": (datetime.now() - pd.Timedelta(days=HISTORY_DAYS-day)).strftime("%Y-%m-%d"),
            "aqi": np.random.randint(40, 200),   # Replace with real AQI API if you want
            **weather,
            "traffic_level": np.random.randint(1, 5),
            "dust_road_flag": np.random.randint(0, 2),
        })

    # build future inputs using OpenWeather + user input
    new_inputs = []
    for offset in range(1, PREDICT_DAYS + 1):
        weather = get_weather_from_openweather(api_key, location, units)

        new_inputs.append({
            "timestamp": (datetime.now() + pd.Timedelta(days=offset)).strftime("%Y-%m-%d"),
            **weather,
            "traffic_level": int(input(f"Enter traffic level (1-4) for day {offset}: ")),
            "dust_road_flag": int(input(f"Dusty roads? (0/1) for day {offset}: "))
        })

    # build dataset
    X, y, scaler, X_future_scaled = build_dataset(history, new_inputs, SEQ_LEN)

    # train
    model = build_model(SEQ_LEN, X.shape[2])
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    # predict
    predictions = predict_next_days(model, scaler, history, X_future_scaled, SEQ_LEN)

    # output
    print("\n==============================")
    print("Predicted AQI:")
    for i, p in enumerate(predictions):
        print(f"Day {i+1}: {p:.2f}")

