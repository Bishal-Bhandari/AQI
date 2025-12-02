import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# build dataset
def build_dataset(history,
                  new_inputs,   # list of dicts: weather, season, extra features, timestamp ...
                  seq_len=7):
    df_hist = pd.DataFrame(history)
    df_new = pd.DataFrame(new_inputs)

    # Combine
    df = pd.concat([df_hist, df_new], ignore_index=True)
    # feature engineering
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.month
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    # Example: dry-season flag (Feb–Apr)
    df['dry_season'] = df['month'].isin([2, 3, 4]).astype(int)

    # define features to use
    features = ['weather_temp', 'weather_humidity',
                'wind_speed', 'wind_direction',  # if available
                'traffic_level', 'dust_road_flag', 'dry_season',
                'month', 'dayofyear']
    target = 'aqi'

# LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

if __name__ == "__main__":
    # historical data (last 14 days)
    history = []
    for day in range(14):
        history.append({
            'timestamp': (datetime.now() - pd.Timedelta(days=14-day)).strftime("%Y-%m-%d"),
            'aqi': np.random.randint(50, 200),                # placeholder
            'weather_temp': np.random.uniform(15, 35),
            'weather_humidity': np.random.uniform(30, 90),
            'wind_speed': np.random.uniform(0, 10),
            'wind_direction': np.random.uniform(0, 360),
            'traffic_level': np.random.randint(1, 5),         # e.g. 1–5 scale
            'dust_road_flag': np.random.randint(0, 2),        # 0 or 1
        })
    # new inputs (next 3 days — maybe from weather forecast / user inputs)
    new_inputs = []
    for offset in range(1, 4):
        new_inputs.append({
            'timestamp': (datetime.now() + pd.Timedelta(days=offset)).strftime("%Y-%m-%d"),
            'weather_temp': np.random.uniform(15, 35),
            'weather_humidity': np.random.uniform(30, 90),
            'wind_speed': np.random.uniform(0, 10),
            'wind_direction': np.random.uniform(0, 360),
            'traffic_level': np.random.randint(1, 5),
            'dust_road_flag': np.random.randint(0, 2),
        })

        # Build data
        seq_len = 7
        X, y, scaler, X_new_scaled = build_dataset(history, new_inputs, seq_len=seq_len)

        # Train model on history
        model = build_model((seq_len, X.shape[2]))
        model.fit(X, y, epochs=20, batch_size=4, verbose=1)

        # take the last seq_len from history + first few new inputs to form input for first prediction
        last_seq = scaler.transform(pd.DataFrame(history)[
                                        ['weather_temp', 'weather_humidity', 'wind_speed', 'wind_direction',
                                         'traffic_level', 'dust_road_flag', 'month', 'dayofyear', 'dry_season']].values[
                                    -seq_len:])
        last_seq = last_seq.reshape((1, seq_len, X.shape[2]))

        predictions = []
        for i in range(len(new_inputs)):
            pred = model.predict(last_seq)[0][0]
            predictions.append(pred)
            # update last_seq by appending new input (scaled) and popping first
            new_feat = X_new_scaled[i].reshape((1, 1, X.shape[1]))
            last_seq = np.concatenate([last_seq[:, 1:, :], new_feat], axis=1)

        print("Predicted AQI for next days:", predictions)