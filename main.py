"""
PayGuard Fraud Detection API
Phase 3 deliverable: real-time serving with monitoring hooks.

Run: uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np
import logging
import time
from datetime import datetime

# ---------- Logging setup (for monitoring / drift detection later) ----------
logging.basicConfig(
    filename="predictions.log",
    level=logging.INFO,
    format="%(message)s"
)

app = FastAPI(title="PayGuard Fraud Detection API", version="1.0")

# ---------- Load model artifacts once at startup ----------
MODEL_PATH = "model.joblib"
META_PATH = "model_meta.joblib"

model = joblib.load(MODEL_PATH)
meta = joblib.load(META_PATH)
THRESHOLD = meta["threshold"]
FEATURE_COLS = meta["feature_cols"]
MODEL_NAME = meta["model_name"]


# ---------- Request / Response schemas ----------
class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    risk_score: float
    is_fraud: bool
    threshold_used: float
    model_name: str
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "threshold": THRESHOLD}


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: Transaction):
    start = time.perf_counter()
    try:
        row = np.array([[getattr(txn, col) for col in FEATURE_COLS]])
    except AttributeError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")

    proba = model.predict_proba(row)[0, 1]
    is_fraud = bool(proba >= THRESHOLD)
    latency_ms = (time.perf_counter() - start) * 1000

    # Log every request/response for future drift monitoring (per Best Practices)
    logging.info(
        f"{datetime.utcnow().isoformat()}\t"
        f"amount={txn.Amount}\trisk_score={proba:.4f}\t"
        f"is_fraud={is_fraud}\tlatency_ms={latency_ms:.2f}"
    )

    return PredictionResponse(
        risk_score=round(float(proba), 4),
        is_fraud=is_fraud,
        threshold_used=THRESHOLD,
        model_name=MODEL_NAME,
        latency_ms=round(latency_ms, 2),
    )
