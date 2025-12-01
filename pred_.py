import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam



if __name__ == "__main__":
    # Example historical data (last 14 days)
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