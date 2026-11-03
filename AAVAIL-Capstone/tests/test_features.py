import pandas as pd
import pytest
from app.features import prepare_training_data, prediction_frame

def test_prepare_training_data_creates_date_features():
    df = pd.DataFrame({"date": ["2024-02-03"], "country": ["USA"], "revenue": [10]})
    result = prepare_training_data(df)
    assert result.iloc[0]["month"] == 2
    assert result.iloc[0]["country"] == "USA"

def test_missing_required_column_raises():
    with pytest.raises(ValueError):
        prepare_training_data(pd.DataFrame({"date": [], "country": []}))

def test_prediction_frame():
    row = prediction_frame("2024-02-03", "USA")
    assert row.iloc[0]["day_of_year"] == 34
