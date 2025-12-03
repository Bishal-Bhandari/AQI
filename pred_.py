import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

SEQ_LEN = config["lstm"]["sequence_length"]

# Feature engineering
def add_features(df):
    df["month"] = pd.to_datetime(df["timestamp"]).dt.month
    df["dayofyear"] = pd.to_datetime(df["timestamp"]).dt.dayofyear
    df["dry_season"] = df["month"].apply(lambda m: 1 if 2 <= m <= 4 else 0)
    return df


# Build dataset for LSTM
def build_dataset(history, future_inputs, seq_len=7):
    feature_cols = [
        "weather_temp", "weather_humidity", "wind_speed", "wind_direction",
        "traffic_level", "dust_road_flag", "month", "dayofyear", "dry_season"
    ]

    # lists to DataFrames
    hist_df = add_features(pd.DataFrame(history))
    future_df = add_features(pd.DataFrame(future_inputs))

    # prepare scaler
    scaler = MinMaxScaler()
    scaled_hist = scaler.fit_transform(hist_df[feature_cols])

    # build sequences (X) and targets (y)
    X, y = [], []
    for i in range(len(scaled_hist) - seq_len):
        X.append(scaled_hist[i:i+seq_len])
        y.append(hist_df["aqi"].iloc[i+seq_len])

    X, y = np.array(X), np.array(y)

    # scale future inputs
    X_future = scaler.transform(future_df[feature_cols])

    return X, y, scaler, X_future, feature_cols


# build LSTM model
def build_model(seq_len, num_features):
    model = Sequential([
        Input(shape=(seq_len, num_features)),
        LSTM(64, activation="tanh"),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# perform prediction
def predict_next_days(model, scaler, history_df, future_scaled, feature_cols, seq_len=7):
    # build with engineered features
    hist_df = add_features(pd.DataFrame(history_df))
    last_seq_raw = hist_df[feature_cols].tail(seq_len)

   #scale last sequencws
    last_seq = scaler.transform(last_seq_raw)

    last_seq = last_seq.reshape((1, seq_len, len(feature_cols)))

    predictions = []
    seq = last_seq.copy()

    for i in range(len(future_scaled)):
        pred = model.predict(seq)[0][0]
        predictions.append(pred)

        # prepare new timestep
        new_input = future_scaled[i].reshape(1, 1, len(feature_cols))

        # slide window
        seq = np.concatenate([seq[:, 1:, :], new_input], axis=1)

    return predictions


if __name__ == "__main__":
    seq_len = 7

    # generate synthetic 14-day training history
    history = []
    for day in range(14):
        history.append({
            'timestamp': (datetime.now() - pd.Timedelta(days=14-day)).strftime("%Y-%m-%d"),
            'aqi': np.random.randint(40, 200),
            'weather_temp': np.random.uniform(5, 35),
            'weather_humidity': np.random.uniform(20, 90),
            'wind_speed': np.random.uniform(0, 12),
            'wind_direction': np.random.uniform(0, 360),
            'traffic_level': np.random.randint(1, 5),
            'dust_road_flag': np.random.randint(0, 2),
        })

    # user/API-generated inputs for next 3 days
    new_inputs = []
    for offset in range(1, 4):
        new_inputs.append({
            'timestamp': (datetime.now() + pd.Timedelta(days=offset)).strftime("%Y-%m-%d"),
            'weather_temp': np.random.uniform(10, 35),
            'weather_humidity': np.random.uniform(20, 90),
            'wind_speed': np.random.uniform(0, 12),
            'wind_direction': np.random.uniform(0, 360),
            'traffic_level': np.random.randint(1, 5),
            'dust_road_flag': np.random.randint(0, 2),
        })

    # build dataset
    X, y, scaler, X_future_scaled, feature_cols = build_dataset(
        history, new_inputs, seq_len=seq_len
    )

    # train model
    model = build_model(seq_len, X.shape[2])
    model.fit(X, y, epochs=20, batch_size=4, verbose=1)

    # predict next 3 days AQI
    predictions = predict_next_days(
        model, scaler, history, X_future_scaled, feature_cols, seq_len
    )

    print("\n==============================")
    print("\nPredicted AQI for next 3 days:")
    for i, p in enumerate(predictions):
        print(f"Day {i+1}: {p:.2f}")
