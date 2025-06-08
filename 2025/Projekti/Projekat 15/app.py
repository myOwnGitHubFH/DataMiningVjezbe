import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

DATA_PATH = r"C:/Users/ZIRA/Desktop/MASTER/RP/mahir"
CITY = "New York"
WINDOW_SIZE = 24

csv_files = [
    "humidity.csv",
    "pressure.csv",
    "temperature.csv",
    "wind_direction.csv",
    "wind_speed.csv",
    "weather_description.csv"
]

def load_feature(file):
    df = pd.read_csv(os.path.join(DATA_PATH, file))
    df = df[["datetime", CITY]]
    df["datetime"] = pd.to_datetime(df["datetime"])
    feature_name = file.replace(".csv", "")
    return df.rename(columns={CITY: feature_name}).set_index("datetime")

dfs = [load_feature(f) for f in csv_files]
data = pd.concat(dfs, axis=1)

data = data.fillna(method="ffill").fillna(method="bfill")
print(data)
data = pd.get_dummies(data, columns=["weather_description"])

data = data.fillna(0)

data = data.astype(np.float32)

numerical_cols = [col for col in data.columns if not col.startswith("weather_description_")]
scaler = MinMaxScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

def create_sequences(data, target_col, window):
    X, y = [], []
    target_idx = data.columns.get_loc(target_col)
    for i in range(len(data) - window):
        X.append(data.iloc[i:i+window].values)
        y.append(data.iloc[i+window, target_idx])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

target_column = "temperature"

X, y = create_sequences(data, target_column, WINDOW_SIZE)

print("X dtype:", X.dtype, "shape:", X.shape)
print("y dtype:", y.dtype, "shape:", y.shape)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(WINDOW_SIZE, X.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1) 
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1)

pred = model.predict(X_test)