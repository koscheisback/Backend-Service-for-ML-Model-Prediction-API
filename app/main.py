"""Flask service that serves predictions from a pre-trained scikit-learn model."""
import logging
import math
import os
from pathlib import Path

import joblib
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

FEATURES = ("sepal_length", "sepal_width", "petal_length", "petal_width")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.joblib"

log = logging.getLogger("predict-service")


class ValidationError(ValueError):
    """Raised when the request payload is missing or invalid."""


def load_model(path):
    """Load the serialized model bundle ({'model': estimator, 'classes': [...]}).

    Only load trusted files: joblib/pickle can execute code on load.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"model file not found: {path}")
    bundle = joblib.load(path)
    return bundle["model"], bundle["classes"]


def validate_payload(payload):
    """Return the feature values in model order, or raise ValidationError."""
    if not isinstance(payload, dict):
        raise ValidationError("request body must be a JSON object")
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        raise ValidationError(f"missing features: {', '.join(missing)}")
    unknown = sorted(set(payload) - set(FEATURES))
    if unknown:
        raise ValidationError(f"unknown features: {', '.join(unknown)}")
    values = []
    for name in FEATURES:
        v = payload[name]
        # bool is a subclass of int in Python, so reject it explicitly.
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValidationError(f"feature '{name}' must be a number")
        if not math.isfinite(v):
            raise ValidationError(f"feature '{name}' must be finite")
        values.append(float(v))
    return values


def create_app(model_path=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = Flask(__name__)
    model, classes = load_model(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    app.config["MODEL"], app.config["CLASSES"] = model, classes

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if payload is None:
            raise ValidationError("body must be valid JSON with Content-Type: application/json")
        row = validate_payload(payload)
        class_id = int(app.config["MODEL"].predict([row])[0])
        return jsonify(prediction=app.config["CLASSES"][class_id], class_id=class_id)

    @app.errorhandler(ValidationError)
    def handle_validation(err):
        return jsonify(error="bad_request", message=str(err)), 400

    @app.errorhandler(HTTPException)
    def handle_http(err):
        return jsonify(error=err.name.lower().replace(" ", "_"), message=err.description), err.code

    @app.errorhandler(Exception)
    def handle_unexpected(err):
        log.exception("unexpected error")
        return jsonify(error="internal_server_error", message="unexpected error"), 500

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
