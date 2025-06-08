import os
import pandas as pd


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

data = pd.get_dummies(data, columns=["weather_description"])

data = data.fillna(0)