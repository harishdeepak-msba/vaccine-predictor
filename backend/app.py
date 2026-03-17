import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_h1n1 = joblib.load(os.path.join(BASE_DIR, "model_h1n1.pkl"))
model_seasonal = joblib.load(os.path.join(BASE_DIR, "model_seasonal.pkl"))

FEATURE_COLS = [
    "h1n1_concern","h1n1_knowledge","behavioral_antiviral_meds","behavioral_avoidance",
    "behavioral_face_mask","behavioral_wash_hands","behavioral_large_gatherings",
    "behavioral_outside_home","behavioral_touch_face","doctor_recc_h1n1",
    "doctor_recc_seasonal","chronic_med_condition","child_under_6_months",
    "health_worker","health_insurance","opinion_h1n1_vacc_effective","opinion_h1n1_risk",
    "opinion_h1n1_sick_from_vacc","opinion_seas_vacc_effective","opinion_seas_risk",
    "opinion_seas_sick_from_vacc","age_group","education","race","sex","income_poverty",
    "marital_status","rent_or_own","employment_status","hhs_geo_region","census_msa",
    "household_adults","household_children","employment_industry","employment_occupation"
]
@app.route("/", methods=["GET"])
def index():
    from flask import redirect
    return redirect("https://harishdeepak-msba.github.io/vaccine-predictor/")

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "vaccine-predictor-api", "model": "MLP"})

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing features"}), 400
    try:
        features = data["features"]
        row = {col: features.get(col, np.nan) for col in FEATURE_COLS}
        df = pd.DataFrame([row])
        p_h1n1 = float(model_h1n1.predict_proba(df)[0][1])
        p_seasonal = float(model_seasonal.predict_proba(df)[0][1])
        return jsonify({
            "h1n1_probability": round(p_h1n1 * 100),
            "seasonal_probability": round(p_seasonal * 100),
            "reasoning": f"MLP model prediction (ROC-AUC 0.832 / 0.853). H1N1 probability: {round(p_h1n1*100)}%, Seasonal probability: {round(p_seasonal*100)}%. Key drivers include doctor recommendations, vaccine effectiveness opinions, and health worker status."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
