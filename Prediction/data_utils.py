import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def add_features(df):
    df["month"] = pd.to_datetime(df["timestamp"]).dt.month
    df["dayofyear"] = pd.to_datetime(df["timestamp"]).dt.dayofyear
    df["dry_season"] = df["month"].apply(lambda m: 1 if 2 <= m <= 4 else 0)
    return df


def build_dataset(history, future_inputs, seq_len, feature_cols):
    hist_df = add_features(pd.DataFrame(history))
    future_df = add_features(pd.DataFrame(future_inputs))

    scaler = MinMaxScaler()
    scaled_hist = scaler.fit_transform(hist_df[feature_cols])

    X, y = [], []
    for i in range(len(scaled_hist) - seq_len):
        X.append(scaled_hist[i:i+seq_len])
        y.append(hist_df["aqi"].iloc[i+seq_len])

    X = np.array(X)
    y = np.array(y)

    X_future = scaler.transform(future_df[feature_cols])

    return X, y, scaler, X_future


def prepare_last_sequence(history_df, scaler, feature_cols, seq_len):
    hist_df = add_features(pd.DataFrame(history_df))
    last_seq_raw = hist_df[feature_cols].tail(seq_len)

    last_seq = scaler.transform(last_seq_raw)
    return last_seq.reshape(1, seq_len, len(feature_cols))
