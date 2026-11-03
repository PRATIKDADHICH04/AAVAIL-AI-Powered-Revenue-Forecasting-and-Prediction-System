import pandas as pd

REQUIRED_COLUMNS = {"date", "country", "revenue"}

def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "country", "revenue"])
    out["country"] = out["country"].astype(str)
    out["revenue"] = pd.to_numeric(out["revenue"], errors="coerce")
    out = out.dropna(subset=["revenue"])
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day_of_year"] = out["date"].dt.dayofyear
    return out[["country", "year", "month", "day_of_year", "revenue"]]

def prediction_frame(target_date: str, country: str) -> pd.DataFrame:
    dt = pd.to_datetime(target_date, errors="raise")
    return pd.DataFrame([{
        "country": str(country),
        "year": dt.year,
        "month": dt.month,
        "day_of_year": dt.dayofyear,
    }])
