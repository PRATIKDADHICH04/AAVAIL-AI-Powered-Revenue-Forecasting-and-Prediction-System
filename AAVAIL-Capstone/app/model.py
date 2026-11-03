from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from app.features import prepare_training_data

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

def build_pipeline(estimator):
    pre = ColumnTransformer([
        ("country", OneHotEncoder(handle_unknown="ignore"), ["country"]),
        ("numeric", "passthrough", ["year", "month", "day_of_year"]),
    ])
    return Pipeline([("preprocess", pre), ("model", estimator)])

def train_and_compare(csv_path="data/revenue.csv"):
    raw = pd.read_csv(csv_path)
    data = prepare_training_data(raw)
    if len(data) < 10:
        raise ValueError("At least 10 valid rows are required to train.")
    # Time-ordered holdout avoids randomly mixing future observations into training.
    split = max(1, int(len(data) * 0.8))
    train, test = data.iloc[:split], data.iloc[split:]
    if test.empty:
        raise ValueError("Not enough rows for a holdout test set.")
    X_train = train.drop(columns="revenue")
    y_train = train["revenue"]
    X_test = test.drop(columns="revenue")
    y_test = test["revenue"]

    candidates = {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "random_forest": RandomForestRegressor(n_estimators=150, random_state=42, min_samples_leaf=2),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
    }
    results = {}
    fitted = {}
    for name, estimator in candidates.items():
        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(mean_squared_error(y_test, pred) ** 0.5),
            "r2": float(r2_score(y_test, pred)) if len(y_test) > 1 else None,
        }
        fitted[name] = pipe

    # Select the best non-baseline model by MAE.
    winner = min((n for n in results if n != "baseline_mean"), key=lambda n: results[n]["mae"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[winner], MODEL_PATH)
    metadata = {
        "selected_model": winner,
        "metrics": results,
        "countries": sorted(data["country"].unique().tolist()),
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return metadata

def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python -m scripts.train")
    return joblib.load(MODEL_PATH), json.loads(METADATA_PATH.read_text())
