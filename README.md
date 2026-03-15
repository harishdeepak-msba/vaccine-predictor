# Health Analytics Project
## Vaccine Uptake Predictor

Predicts H1N1 and Seasonal Flu vaccine uptake using a trained MLP Neural Network on the NHFS dataset.

## Live App

- **Frontend:** https://harishdeepak-msba.github.io/vaccine-predictor/
- **Backend API:** https://vaccine-predictor.onrender.com

## Model Performance

| Model | ROC-AUC H1N1 | ROC-AUC Seasonal | Mean ROC-AUC |
|---|---|---|---|
| Neural Network (MLP) ✅ | 0.8324 | 0.8529 | 0.8426 |
| Logistic Regression | 0.8288 | 0.8552 | 0.8420 |
| Random Forest | 0.8285 | 0.8518 | 0.8402 |

## Tech Stack

- Frontend: HTML/CSS/JS on GitHub Pages
- Backend: Python Flask on Render
- Model: scikit-learn MLP Pipeline (35 features, 26,707 records)
- Dataset: National H1N1 Flu Survey (NHFS)

## Notes

Backend may take ~30 seconds to wake on first request (Render free tier).
