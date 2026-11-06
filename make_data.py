import os
import numpy as np
import pandas as pd

np.random.seed(42)
n = 10000

air = np.random.normal(300, 2, n)
proc = air + 10 + np.random.normal(0, 1, n)
rpm = np.random.normal(1540, 170, n).astype(int)
torque = np.random.normal(40, 10, n).clip(3, 77).round(1)
wear = np.random.randint(0, 250, n)

df = pd.DataFrame({
    "UDI": range(1, n + 1),
    "Product ID": [f"M{14860 + i}" for i in range(n)],
    "Type": np.random.choice(["L", "M", "H"], n, p=[0.6, 0.3, 0.1]),
    "Air temperature [K]": air.round(1),
    "Process temperature [K]": proc.round(1),
    "Rotational speed [rpm]": rpm,
    "Torque [Nm]": torque,
    "Tool wear [min]": wear,
})

# Failures are more likely with high wear and high torque
risk = (wear > 200) * 0.08 + (torque > 60) * 0.10 + 0.01
df["Machine failure"] = (np.random.rand(n) < risk).astype(int)

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/ai4i2020.csv", index=False)
print("Saved data/raw/ai4i2020.csv", df.shape)