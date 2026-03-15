import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic

app = Flask(__name__)
CORS(app)
```
Fix CORS

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a vaccine uptake prediction model trained on the National H1N1 Flu Survey (NHFS) dataset.
When given individual-level features, estimate probabilities for H1N1 and seasonal flu vaccine uptake.

Key predictors (from correlation analysis):
- doctor_recc_h1n1 is the strongest predictor for H1N1 uptake (corr ~0.39)
- doctor_recc_seasonal is the strongest predictor for seasonal uptake (corr ~0.37)
- opinion_h1n1_risk and opinion_h1n1_vacc_effective strongly predict H1N1 uptake
- opinion_seas_risk and opinion_seas_vacc_effective strongly predict seasonal uptake
- High sick_from_vacc scores reduce probability
- health_worker, health_insurance, and chronic_med_condition boost both probabilities
- Older age groups (55+) tend to have higher seasonal uptake

Always respond with ONLY valid JSON — no markdown fences, no prose."""


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "vaccine-predictor-api"})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing features in request body"}), 400

    features = data["features"]

    user_prompt = f"""Given these individual-level survey features:
{json.dumps(features, indent=2)}

Estimate the probability (0-100) that this person receives:
1. The H1N1 vaccine
2. The seasonal flu vaccine

Apply logistic-regression-style reasoning based on the key predictors described.
Respond ONLY with this JSON structure:
{{
  "h1n1_probability": <integer 0-100>,
  "seasonal_probability": <integer 0-100>,
  "reasoning": "<2-3 sentences explaining the main drivers>"
}}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = message.content[0].text.strip()
        # Strip any accidental markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        return jsonify(result)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Model returned invalid JSON: {str(e)}"}), 500
    except anthropic.APIError as e:
        return jsonify({"error": f"Anthropic API error: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
