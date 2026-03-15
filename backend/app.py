import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic

app = Flask(__name__)
CORS(app)

@app.route("/api/health", methods=["GET"])
def health():
    key = os.environ.get("ANTHROPIC_API_KEY")
    return jsonify({"status": "healthy", "key_found": key is not None, "key_prefix": key[:15] if key else "NOT SET"})

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing features"}), 400
    features = data["features"]
    user_prompt = f"Given these survey features:\n{json.dumps(features)}\n\nEstimate probability 0-100 for H1N1 and seasonal flu vaccine uptake.\nRespond ONLY with this JSON:\n{{\"h1n1_probability\": <integer 0-100>, \"seasonal_probability\": <integer 0-100>, \"reasoning\": \"<2-3 sentences>\"}}"
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=400, system="You are a vaccine uptake prediction model. Respond ONLY with valid JSON.", messages=[{"role": "user", "content": user_prompt}])
        raw = message.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        return jsonify(json.loads(raw))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
