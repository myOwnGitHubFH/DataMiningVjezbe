import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

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
data["temperature"] = data["temperature"] - 273.15 # transformacija u celzijuse (lakše)

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

def invert_scale(pred_scaled, y_scaled, scaler, feature_name, numerical_cols):
    idx = numerical_cols.index(feature_name)
    pred_padded = np.zeros((len(pred_scaled), len(numerical_cols)))
    y_padded = np.zeros((len(y_scaled), len(numerical_cols)))
    pred_padded[:, idx] = pred_scaled.flatten()
    y_padded[:, idx] = y_scaled.flatten()
    inv_pred = scaler.inverse_transform(pred_padded)[:, idx]
    inv_y = scaler.inverse_transform(y_padded)[:, idx]
    return inv_pred, inv_y

pred_inv, y_test_inv = invert_scale(pred, y_test, scaler, target_column, numerical_cols)

# koliko model pogriješi bez obzira na smjer greške (+- X celzijusa) - uzima apsolutnu vrijednost
mae = mean_absolute_error(y_test_inv, pred_inv)
print(f"MAE: {mae:.2f}")

# ista stvar kao iznad, samo sto ovdje se kvadriraju greske - veća osjetljivost na greške
rmse = np.sqrt(mean_squared_error(y_test_inv, pred_inv))
print(f"Temperature RMSE: {rmse:.2f}")

# Prikaz rezultata
plt.figure(figsize=(14, 5))
plt.plot(y_test_inv[:500], label='Prava temperatura')
plt.plot(pred_inv[:500], label='Predviđena temperatura')
plt.title("Predikcija temperature")
plt.legend()
plt.show()
