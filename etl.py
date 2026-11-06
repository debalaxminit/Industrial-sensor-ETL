import sqlite3
import numpy as np
import pandas as pd

# ---------- EXTRACT ----------
df = pd.read_csv("data/raw/ai4i2020.csv")
print("Raw shape:", df.shape)

# ---------- TRANSFORM ----------
df = df.rename(columns={
    "UDI": "id",
    "Product ID": "product_id",
    "Type": "quality_type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
})

print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

df["air_temp_c"] = df["air_temp_k"] - 273.15
df["process_temp_c"] = df["process_temp_k"] - 273.15

df["temp_diff_c"] = df["process_temp_c"] - df["air_temp_c"]
df["power_w"] = df["torque_nm"] * df["rpm"] * 2 * np.pi / 60

q1, q3 = df["torque_nm"].quantile([0.25, 0.75])
iqr = q3 - q1
df["torque_outlier"] = (df["torque_nm"] < q1 - 1.5 * iqr) | (df["torque_nm"] > q3 + 1.5 * iqr)
print("Torque outliers:", df["torque_outlier"].sum())

# ---------- LOAD ----------
import os
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/sensor_clean.csv", index=False)
with sqlite3.connect("data/processed/sensor.db") as conn:
    df.to_sql("sensor_readings", conn, if_exists="replace", index=False)

# ---------- REPORT ----------
summary = df.groupby("quality_type").agg(
    readings=("id", "count"),
    avg_torque_nm=("torque_nm", "mean"),
    avg_power_w=("power_w", "mean"),
    failure_rate=("machine_failure", "mean"),
).round(2)
summary.to_csv("data/processed/summary_report.csv")
print(summary)