# PayGuard — Real-Time Payment Fraud Detection

Reconstructed ML system for flagging fraudulent card transactions in real time, built on the
[ULB Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions, 492 confirmed frauds, ~0.17%).

---

## 1. Problem Framing & Architecture

### What kind of problem is this?
Binary classification (`Class`: 0 = legitimate, 1 = fraud) — but a **severely imbalanced** one.
Frauds are ~0.17% of transactions, so this is treated as a rare-event detection problem, not a
standard balanced classification task.

### Why accuracy is the wrong metric
A model that predicts "not fraud" for every transaction scores **99.83% accuracy** while catching
zero fraud. That number is dangerously misleading in a business review. Instead we track:

- **PR-AUC (Average Precision)** — the primary model-selection metric. It summarizes
  precision/recall trade-off across all thresholds and is far more informative than ROC-AUC when
  the positive class is this rare.
- **Precision & Recall at the deployed threshold** — what actually gets reported to the business.
- **ROC-AUC** — reported for reference only, not used to pick the winning model.

### Cost-benefit analysis → decision threshold
A false positive (blocking a real customer) and a false negative (letting fraud through) are **not
equally costly**:

| Outcome | Cost driver | Assumed cost (illustrative) |
|---|---|---|
| False Negative (missed fraud) | Chargeback = full transaction loss | ≈ $122 (avg. fraud amount observed in this data) |
| False Positive (blocked real customer) | Abandoned cart, support cost, churn risk | ≈ $5 (estimated friction cost) |

These numbers are **assumptions for this exercise** — in production they'd come from finance/risk
teams (actual chargeback liability, customer lifetime value, support ticket cost). The point is
the *ratio* matters more than the exact figures: missing fraud is treated as ~24x more expensive
than annoying one legitimate customer, so the classifier is tuned to catch more fraud even at the
cost of more false alarms.

We scan the precision-recall curve and pick the threshold that **minimizes total estimated cost**
(`FP_count × $5 + FN_count × $122`) rather than using the default 0.5 cutoff. This produced a
threshold of **~0.107**, much lower than default — meaning we deliberately flag more transactions
for review because missed fraud is so much more expensive.

### Training pipeline (sketch)
```
raw CSV → dedup → time-sorted split (80/20) → feature matrix (Time, V1-V28, Amount)
        → train candidate models (class_weight='balanced')
        → select by PR-AUC → tune threshold by cost curve → save model + threshold
```

**Note on split strategy:** we use a *time-based* split (train on the earlier 80% of transactions,
test on the later 20%) instead of a random split. This mirrors production reality — the model is
always trained on the past and evaluated on the future — and avoids an overly optimistic
evaluation that a random shuffle would give.

---

## 2. Data Lifecycle & Ingestion

### Signals to ingest in production
The ULB dataset only gives PCA-anonymized features (V1–V28) plus Time and Amount, which stands in
for a richer real signal set. In a real PayGuard system we'd ingest:

- **Transaction signals**: amount, merchant category, currency, time of day, transaction velocity
  (# transactions in last 1 min / 1 hr / 1 day for this card).
- **Device/session signals**: device fingerprint, IP geolocation, distance from last known
  transaction location, new-device flag.
- **Account signals**: account age, historical spending pattern, previous chargeback history.

**Freshness requirement:** velocity and device signals need to be available at request time with
near-zero latency (looked up from a fast key-value store, e.g. Redis) — a batch feature computed
overnight is useless for blocking a transaction happening right now.

### Retraining cadence & versioning
Fraud is adversarial — patterns shift as fraudsters adapt, so a model trained once and left alone
degrades. Plan:

- **Retraining cadence**: weekly retrain on a rolling window (e.g. last 90 days), with an
  automatic retrain trigger if drift monitoring (below) crosses a threshold.
- **Data versioning**: snapshot the training dataset each retrain (DVC or equivalent) so any
  model can be traced back to the exact data it was trained on.
- **Model versioning**: track each trained model with its metrics, threshold, and training data
  version (MLflow model registry or similar), so a bad deploy can be rolled back instantly.

---

## 3. Model Serving & Monitoring

### Real-time vs batch — and why real-time
The business requirement is to flag transactions **before they're approved**, which rules out
batch scoring. We serve via a **FastAPI** endpoint (`POST /predict`) that loads the model once at
startup and returns a risk score in milliseconds.

**What real-time serving constrains:**
- Model must be small/fast enough for sub-100ms inference — this ruled out heavier
  stacking/ensemble approaches beyond a single Random Forest for this exercise.
- Feature computation (especially velocity features) must be pre-computed or cheap to fetch, not
  computed from a full historical scan per request.

### API contract
- `POST /predict` — accepts one transaction's features, returns `risk_score` (0–1),
  `is_fraud` (bool, using the cost-optimal threshold), `model_name`, and `latency_ms`.
- `GET /health` — reports which model/threshold is currently loaded.

### Monitoring plan
- **Prediction logging**: every request/response is logged (`predictions.log`) with timestamp,
  amount, risk score, decision, and latency — the raw material for all monitoring below.
- **KPI tracking**: daily aggregation of flagged-rate, and (once labels arrive from chargebacks)
  realized precision/recall — watched on a dashboard.
- **Drift detection**: compare the distribution of incoming feature values and risk scores
  week-over-week (e.g. population stability index, or a tool like Evidently AI) — a shift signals
  either changing customer behavior or changing fraud tactics, and should trigger a retrain review
  before it costs money in missed fraud or angry customers.

---

## Results (this run)

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Logistic Regression (class_weight=balanced) | 0.747 | 0.986 |
| **Random Forest (class_weight=balanced)** ✅ selected | **0.805** | 0.982 |

At the cost-optimal threshold (0.107) on the held-out (time-based) test set:

- Recall: 83.8% (62 / 74 frauds caught)
- Precision: 39.7%
- Accuracy: 99.81% (reported here for completeness only — **not** used for model selection or
  business reporting, since it is misleading for this imbalanced problem)

## Data assumptions & limitations
- Duplicate rows (1,081) were dropped before splitting.
- V1–V28 are already PCA-anonymized by the dataset provider, so no further feature engineering
  was applied to them; Amount and Time were used as-is.
- Cost figures ($122 / $5) are illustrative assumptions, not sourced from real PayGuard financials.
- `xgboost` / `imbalanced-learn` were unavailable in this build environment (no internet access);
  scikit-learn's `class_weight='balanced'` was used instead to handle imbalance. Swapping in
  XGBoost with `scale_pos_weight` or SMOTE resampling is a natural next iteration.

## Repo structure
```
fraud_project/
├── README.md
├── fraud_detection.ipynb    # EDA + cleaning + training + evaluation + threshold selection
├── main.py                  # FastAPI serving code
├── model.joblib              # trained Random Forest
├── model_meta.joblib         # threshold + feature order + model name
└── predictions.log           # created at runtime by the API (not committed)
```

## Dataset
The raw `creditcard.csv` is **not included in this repo** (150MB+, exceeds what's reasonable for
a git repo). Download it directly from Kaggle before running the notebook:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place `creditcard.csv` in the project root (same folder as the notebook) before running.

## How to run
```bash
pip install pandas scikit-learn joblib fastapi uvicorn matplotlib

# 1. Run the notebook top to bottom — trains the model and saves model.joblib / model_meta.joblib
jupyter notebook fraud_detection.ipynb

# 2. Serve the model
uvicorn main:app --reload --port 8000

# 3. Test it
# open http://127.0.0.1:8000/docs in a browser and try the /predict endpoint
```
