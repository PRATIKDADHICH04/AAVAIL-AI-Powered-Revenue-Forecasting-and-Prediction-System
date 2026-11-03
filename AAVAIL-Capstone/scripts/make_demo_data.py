"""Create clearly synthetic data so the project runs before adding course data."""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
dates = pd.date_range("2022-01-01", periods=540, freq="D")
countries = {"United Kingdom": 900, "USA": 500, "Germany": 250, "Singapore": 180, "EIRE": 120}
rows = []
for country, base in countries.items():
    for i, date in enumerate(dates):
        seasonal = 1 + 0.25 * np.sin(2 * np.pi * date.dayofyear / 365.25)
        trend = 1 + 0.0005 * i
        revenue = max(0, base * seasonal * trend + rng.normal(0, base * 0.08))
        rows.append((date.date().isoformat(), country, round(float(revenue), 2)))
Path("data").mkdir(exist_ok=True)
pd.DataFrame(rows, columns=["date", "country", "revenue"]).to_csv("data/revenue.csv", index=False)
print("Created synthetic demo data at data/revenue.csv. Replace with course data for your submission.")
