"""
app.py
Task 3: API Development

A Flask REST API that loads the trained model and scaler, accepts
patient clinical details as JSON, and returns a heart disease
risk prediction as JSON.
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load trained artifacts once at startup
MODEL = joblib.load("model.pkl")
SCALER = joblib.load("scaler.pkl")
FEATURE_NAMES = joblib.load("feature_names.pkl")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Heart Disease Prediction API is running.",
        "usage": "POST /predict with JSON body containing: " + ", ".join(FEATURE_NAMES)
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Validate that all required features are present
        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Build feature vector in the correct order
        features = np.array([[data[f] for f in FEATURE_NAMES]], dtype=float)

        # Scale and predict
        features_scaled = SCALER.transform(features)
        prediction = MODEL.predict(features_scaled)[0]
        probability = MODEL.predict_proba(features_scaled)[0][1]

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"

        return jsonify({
            "prediction": result,
            "probability": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
