# AI Fraud Detection & Transaction Monitoring System

A portfolio-grade fintech analytics demo that simulates how banks score card transactions for fraud risk using machine learning. Built with Python, Streamlit, XGBoost, and the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset.

## Features

- **Fraud prediction** — XGBoost classifier with class-imbalance handling
- **Risk scoring** — Fraud probability plus Low / Medium / High / Critical tiers
- **Dashboard** — KPIs, suspicious transaction table, amount distributions, fraud trends
- **CSV upload** — Score custom transaction files and download results
- **Live simulation** — Near real-time transaction stream with alerts
- **Alert system** — Toasts and banners for high-risk authorizations

## Tech stack

| Layer | Tools |
|-------|--------|
| Language | Python 3.10+ |
| UI | Streamlit |
| ML | XGBoost, scikit-learn |
| Data | Pandas, NumPy |
| Charts | Plotly |

## Project structure

```
├── app.py                 # Streamlit entry
├── train_model.py         # Training CLI
├── data/creditcard.csv    # Kaggle dataset (not in git)
├── models/                # Trained artifacts (not in git)
└── src/
    ├── config.py          # Paths and thresholds
    ├── preprocessing.py   # Load, validate, preprocess
    ├── train_model.py     # Training pipeline
    ├── prediction.py      # Inference and scoring
    ├── utilities.py       # Alerts, simulation, charts
    └── dashboard.py       # Streamlit UI
```

## Setup

### 1. Clone and install

```bash
cd Bofa__project_1
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Download the dataset

**Quick local test (no Kaggle):**

```bash
python scripts/generate_sample_data.py
```

**Production portfolio (real data):**

1. Create a Kaggle account and download [creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), or use the CLI:

   ```bash
   pip install kaggle
   kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
   ```

2. Ensure the file exists at `data/creditcard.csv`.

### 3. Train the model

```bash
python train_model.py
```

This writes `models/fraud_model.joblib` and `models/training_metadata.json`.

### 4. Run the dashboard

```bash
python -m streamlit run app.py
```

On Windows, if `streamlit` is not recognized, always use `python -m streamlit` (Scripts folder not on PATH).

## Banking context (interview talking points)

### Why fraud detection matters

Card-not-present fraud, account takeover, and illicit flows create direct financial loss, regulatory exposure (e.g., BSA/AML), and reputational harm. Banks must identify suspicious activity quickly to limit liability and protect customers.

### How banks use machine learning

Production systems combine **rules** (velocity limits, geography, merchant category) with **ML models** that score each authorization in milliseconds. Features are often anonymized or transformed (PCA components V1–V28 in this public dataset). Models are retrained as fraud patterns drift; elevated scores route to analysts, step-up authentication, or declines.

### Fraud prevention vs customer experience

Stricter thresholds block more fraud but increase false declines—support calls, card blocks, and abandoned purchases. Institutions tune thresholds by customer segment, channel, and loss tolerance.

### False positives

Each false alert consumes analyst time and erodes trust. With ~0.17% fraud prevalence in this dataset, **accuracy is misleading**; **precision–recall** and **PR-AUC** better reflect model quality. This project tunes a decision threshold on validation data and exposes configurable risk bands.

## Model approach

- **Preprocessor:** StandardScaler on `Time` and `Amount`; V1–V28 passed through
- **Classifier:** XGBoost with `scale_pos_weight` for imbalance
- **Threshold:** Chosen on validation to maximize F1 (adjustable in metadata)
- **Metrics:** PR-AUC (primary), ROC-AUC, precision, recall, F1

## Limitations

This is a **portfolio demonstration**, not a production banking system:

- No PCI compliance, encryption, or real payment rails
- Simulated “real-time” stream (not Kafka/API)
- Public dataset with anonymized features
- No case management or SAR workflow

## Optional enhancements

- SHAP explanations per transaction
- Threshold slider with live precision/recall curve
- Export audit log from live session
- Docker image for one-command demo
- Mock case workflow (Approve / Escalate)

## License & data

The Kaggle dataset is subject to its own license. Do not commit `creditcard.csv` or trained models to public repositories if your policy restricts it.

## Author

Built for internship portfolio showcase — Bank of America technology and fintech analytics roles.
