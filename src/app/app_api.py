"""
Predictive Maintenance API
==========================

This FastAPI service provides programmatic access to predictive maintenance 
outputs such as Remaining Useful Life (RUL) predictions, sensor health indicators, 
and risk classification. 

Intended Use:
-------------
- Serve predictions to dashboards (Streamlit app, BI tools, etc.)
- Provide integration endpoints for other systems (alerting, CMMS, etc.)
- Allow batch or single-unit queries
- Swagger UI for easy exploration, can be accessed at http://127.0.0.1:8000/docs

Author: Stuart Houston
Date: 18-08-2025
"""

# Import necessary libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
from typing import List

# Initialise FastAPI application
app = FastAPI(
    title="Predictive Maintenance API",
    description="API for engine health monitoring and RUL predictions",
    version="0.1.0"
)

# Configuration paths and constants
FEATURES_PATH = Path("data/features/train_FD001_features.csv")
PREDICTIONS_PATH = Path("data/processed/rul_predictions.csv")

# Pydantic Models (for request/response validation)
class UnitPrediction(BaseModel):
    unit: int
    RUL: float
    risk_level: str

class PredictionResponse(BaseModel):
    predictions: List[UnitPrediction]

# Helper Functions
def load_predictions(path: Path) -> pd.DataFrame:
    """
    Loads predictions from CSV and returns a DataFrame.
    Uses risk_level from CSV if available, otherwise computes it.
    """
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found at {path}")
    
    df = pd.read_csv(path)
    
    if not {"unit", "RUL"}.issubset(df.columns):
        raise ValueError("Predictions file missing required columns 'unit' and 'RUL'")

    # Use risk_level from CSV if present
    if "risk_level" not in df.columns:
        # Fallback: compute risk if column missing
        HIGH_RISK_THRESHOLD = 30
        MEDIUM_RISK_THRESHOLD = 100
        def classify_risk(rul: float) -> str:
            if rul < HIGH_RISK_THRESHOLD:
                return "High"
            elif rul < MEDIUM_RISK_THRESHOLD:
                return "Medium"
            return "Low"
        df["risk_level"] = df["RUL"].apply(classify_risk)
    
    return df

# API Routes

@app.get("/", tags=["health"])
def root():
    """Basic service check."""
    return {"message": "Predictive Maintenance API is running."}

@app.get("/predictions", response_model=PredictionResponse, tags=["predictions"])
def get_all_predictions():
    """
    Returns all available RUL predictions with risk levels.
    """
    try:
        df = load_predictions(PREDICTIONS_PATH)
        preds = [
            UnitPrediction(unit=int(row.unit), RUL=float(row.RUL), risk_level=row.risk_level)
            for row in df.itertuples()
        ]
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/{unit_id}", response_model=UnitPrediction, tags=["predictions"])
def get_unit_prediction(unit_id: int):
    """
    Returns prediction and risk level for a specific unit.
    """
    try:
        df = load_predictions(PREDICTIONS_PATH)
        row = df[df["unit"] == unit_id]
        if row.empty:
            raise HTTPException(status_code=404, detail=f"No prediction found for unit {unit_id}")
        record = row.iloc[0]
        return UnitPrediction(unit=int(record.unit), RUL=float(record.RUL), risk_level=record.risk_level)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Future Endpoints (Markers for Expansion) ---
# @app.post("/predict_batch") -> Accept raw sensor data & return predictions
# @app.get("/health_indicators/{unit_id}") -> Return engineered features for a unit
# @app.get("/model/metadata") -> Expose model version, training date, etc.