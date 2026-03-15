# Health Analytics Project
## Vaccine Uptake Predictor — Full-Stack Web App
**Deep Learning Course — Dataset: National H1N1 Flu Survey (NHFS)**
**Team 3 — Harish Deepak & Sharvari Shankhwalkar**

---

## Project Structure

```
vaccine-predictor/
├── backend/
│   ├── app.py                ← Flask backend (prediction API)
│   ├── model_h1n1.pkl        ← Trained MLP model (H1N1 vaccine)
│   ├── model_seasonal.pkl    ← Trained MLP model (Seasonal vaccine)
│   ├── requirements.txt      ← Python dependencies
│   └── render.yaml           ← Render deployment config
├── frontend/
│   └── index.html            ← Frontend (vaccine prediction UI)
└── README.md
```

---

## Live App

- **Frontend:** https://harishdeepak-msba.github.io/vaccine-predictor/
- **Backend API:** https://vaccine-predictor.onrender.com

---

## Quick Start

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Export trained models from Colab notebook

The app needs the full sklearn Pipeline (preprocessor + MLP), not just the model weights. Export both targets after Section 6 has run:

```python
import joblib

# Export best MLP models (highest mean ROC-AUC: 0.8426)
joblib.dump(mlp_h1n1, "model_h1n1.pkl")
joblib.dump(mlp_seasonal, "model_seasonal.pkl")

# Download from Colab
from google.colab import files
files.download("model_h1n1.pkl")
files.download("model_seasonal.pkl")
```

**Why the full Pipeline?** The Colab notebook wraps preprocessing (median imputation, standard scaling, one-hot encoding) and MLPClassifier into a single `Pipeline([("pre", preprocessor), ("clf", MLPClassifier())])`. Pickling the whole Pipeline means the API just calls `pipeline.predict_proba(X_raw_df)` with a raw DataFrame — no manual preprocessing needed.

### 3. Run the server

```bash
python app.py
```

### 4. Open the app

Visit: http://localhost:5000

---

## API Endpoints

| Method | URL | Description |
|---|---|---|
| GET | `/api/health` | API status + model loaded check |
| POST | `/api/predict` | Returns H1N1 + seasonal vaccine probabilities |

---

## Example `/api/predict` request

```bash
curl -X POST https://vaccine-predictor.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "doctor_recc_h1n1": 1,
      "doctor_recc_seasonal": 1,
      "opinion_h1n1_vacc_effective": 5,
      "opinion_h1n1_risk": 4,
      "opinion_seas_vacc_effective": 5,
      "opinion_seas_risk": 4,
      "health_worker": 1,
      "health_insurance": 1,
      "h1n1_concern": 3,
      "age_group": "55-64 Years"
    }
  }'
```

## Example response

```json
{
  "h1n1_probability": 80,
  "seasonal_probability": 78,
  "reasoning": "MLP model prediction (ROC-AUC 0.832 / 0.853). Key drivers include doctor recommendations, vaccine effectiveness opinions, and health worker status."
}
```

---

## Feature Mapping (matches NHFS columns)

| Field | Description | Values |
|---|---|---|
| `doctor_recc_h1n1` | Doctor recommended H1N1 vaccine | 0=No, 1=Yes |
| `doctor_recc_seasonal` | Doctor recommended seasonal vaccine | 0=No, 1=Yes |
| `h1n1_concern` | Level of concern about H1N1 | 0–3 |
| `h1n1_knowledge` | Knowledge about H1N1 | 0–2 |
| `opinion_h1n1_vacc_effective` | Opinion: H1N1 vaccine effectiveness | 1–5 |
| `opinion_h1n1_risk` | Opinion: H1N1 personal risk | 1–5 |
| `opinion_h1n1_sick_from_vacc` | Opinion: risk of getting sick from H1N1 vaccine | 1–5 |
| `opinion_seas_vacc_effective` | Opinion: seasonal vaccine effectiveness | 1–5 |
| `opinion_seas_risk` | Opinion: seasonal flu personal risk | 1–5 |
| `opinion_seas_sick_from_vacc` | Opinion: risk of getting sick from seasonal vaccine | 1–5 |
| `health_worker` | Is a healthcare worker | 0=No, 1=Yes |
| `health_insurance` | Has health insurance | 0=No, 1=Yes |
| `chronic_med_condition` | Has chronic medical condition | 0=No, 1=Yes |
| `age_group` | Age bracket | 18-34 / 35-44 / 45-54 / 55-64 / 65+ Years |
| `sex` | Sex | Male / Female |
| `race` | Race | White / Black / Hispanic / Other or Multiple |
| `education` | Education level | < 12 Years / 12 Years / Some College / College Graduate |
| `income_poverty` | Income vs poverty line | Below Poverty / <= $75,000 / > $75,000 |
| `employment_status` | Employment status | Employed / Unemployed / Not in Labor Force |
| `household_adults` | Number of adults in household | 0–3 |
| `household_children` | Number of children in household | 0–3 |

---

## Model Performance

| Model | ROC-AUC H1N1 | ROC-AUC Seasonal | Mean ROC-AUC |
|---|---|---|---|
| **Neural Network (MLP)** ✅ | 0.8324 | 0.8529 | **0.8426** |
| Logistic Regression | 0.8288 | 0.8552 | 0.8420 |
| Random Forest | 0.8285 | 0.8518 | 0.8402 |

The MLP model was selected as the best overall based on mean ROC-AUC across both targets. All three models use the same sklearn Pipeline with median imputation, standard scaling, and one-hot encoding.

---

## Tech Stack

- **Frontend:** HTML/CSS/JavaScript — hosted on GitHub Pages
- **Backend:** Python Flask — hosted on Render (free tier)
- **Model:** scikit-learn MLP Pipeline (35 features, trained on 26,707 records)
- **Dataset:** National H1N1 Flu Survey (NHFS)

---

## Deployment

### Backend (Render)
- Connect GitHub repo to Render
- Set Root Directory to `backend`
- Add environment variable: `ANTHROPIC_API_KEY` (optional, not used in model mode)
- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`

### Frontend (GitHub Pages)
- Go to repo Settings → Pages
- Set source to branch `main`, folder `/` (root)
- Live at: `https://harishdeepak-msba.github.io/vaccine-predictor/`

---

> **Note:** Backend may take ~30 seconds to wake on first request (Render free tier cold start).
