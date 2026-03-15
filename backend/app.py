import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic

app = Flask(__name__)
CORS(app)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = "You are a vaccine uptake prediction model. Respond ONLY with valid JSON."

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "vaccine-predictor-api"})

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing features"}), 400
    features = data["features"]
    user_prompt = f"""Given these survey features:
{json.dumps(features, indent=2)}

Estimate probability (0-100) for H1N1 and seasonal flu vaccine uptake.
Respond ONLY with this JSON:
{{
  "h1n1_probability": <integer 0-100>,
  "seasonal_probability": <integer 0-100>,
  "reasoning": "<2-3 sentences on main drivers>"
}}"""
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
        )
        raw = message.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)