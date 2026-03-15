\# Health Analytics Project

\*\*Vaccine Uptake Predictor\*\*



Predicts the likelihood of an individual receiving the \*\*H1N1\*\* and \*\*Seasonal Flu\*\* vaccines using a trained Neural Network (MLP) model on the National H1N1 Flu Survey (NHFS) dataset.



\---



\## Live App



| | URL |

|---|---|

| \*\*Frontend\*\* | https://harishdeepak-msba.github.io/vaccine-predictor/ |

| \*\*Backend API\*\* | https://vaccine-predictor.onrender.com |



\---



\## Model Performance



| Model | ROC-AUC H1N1 | ROC-AUC Seasonal | Mean ROC-AUC |

|---|---|---|---|

| Neural Network (MLP) ✅ | 0.8324 | 0.8529 | \*\*0.8426\*\* |

| Logistic Regression | 0.8288 | 0.8552 | 0.8420 |

| Random Forest | 0.8285 | 0.8518 | 0.8402 |



\---



\## API Endpoints



\### GET /api/health

Returns service status and model info.



\### POST /api/predict

\*\*Request:\*\*

```json

{ "features": { "doctor\_recc\_h1n1": 1, "opinion\_h1n1\_vacc\_effective": 5 } }

```

\*\*Response:\*\*

```json

{ "h1n1\_probability": 80, "seasonal\_probability": 78, "reasoning": "..." }

```



\---



\## Tech Stack



\- \*\*Frontend:\*\* HTML/CSS/JS hosted on GitHub Pages

\- \*\*Backend:\*\* Python Flask hosted on Render

\- \*\*Model:\*\* scikit-learn MLP Pipeline (35 features, 26,707 records)

\- \*\*Dataset:\*\* National H1N1 Flu Survey (NHFS)



\---



\## Repo Structure



```

vaccine-predictor/

├── frontend/

│   └── index.html

└── backend/

&#x20;   ├── app.py

&#x20;   ├── model\_h1n1.pkl

&#x20;   ├── model\_seasonal.pkl

&#x20;   ├── requirements.txt

&#x20;   └── render.yaml

```



> Note: Backend may take \~30 seconds to wake on first request (Render free tier).

