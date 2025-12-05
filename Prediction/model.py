from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import numpy as np

def build_model(seq_len, num_features, lstm_units, dense_units):
    model = Sequential([
        Input(shape=(seq_len, num_features)),
        LSTM(lstm_units, activation="tanh"),
        Dense(dense_units, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def predict_future(model, scaler, history, future_scaled, feature_cols, seq_len):
    predictions = []

    # prepare last window
    hist_df = history.copy()
    last_seq_raw = hist_df[feature_cols].tail(seq_len)
    last_seq = scaler.transform(last_seq_raw)
    seq = last_seq.reshape(1, seq_len, len(feature_cols))

    for i in range(len(future_scaled)):
        pred = model.predict(seq)[0][0]
        predictions.append(pred)

        new_input = future_scaled[i].reshape(1, 1, len(feature_cols))
        seq = np.concatenate([seq[:, 1:, :], new_input], axis=1)

    return predictions
