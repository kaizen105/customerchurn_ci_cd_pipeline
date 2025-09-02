import os
from flask import Flask, request, jsonify
import joblib
import numpy as np

# -- Config --
MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.pkl")

# -- App --
app = Flask(__name__)

# Load once at startup
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}, 200

@app.post("/predict")
def predict():
    """
    Accepts:
    {"input": [[...feature vectors...], [...]]}  # 2D list
    or
    {"input": [...feature vector...]}            # 1D list
    """
    try:
        payload = request.get_json(force=True)
        x = payload.get("input")
        if x is None:
            return jsonify(error="Missing 'input'"), 400

        # Normalize to 2D
        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
            X = x
        else:
            X = [x]

        X = np.array(X, dtype=float)
        preds = model.predict(X).tolist()

        try:
            probs = model.predict_proba(X).tolist()
        except Exception:
            probs = None

        return jsonify(predictions=preds, probabilities=probs), 200
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
