# ML Prediction Service

Flask microservice that loads a pre-trained scikit-learn model (Iris logistic regression) from `model/model.joblib` and serves predictions at `POST /predict`. The service does **not** train anything.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.main            # listens on :8000 (override with PORT, model with MODEL_PATH)
```

## Try it
```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
# {"class_id":0,"prediction":"setosa"}

curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"sepal_length":5.1}'
# 400 {"error":"bad_request","message":"missing features: sepal_width, petal_length, petal_width"}

curl localhost:8000/health
```

## Status codes
| Code | When |
|------|------|
| 200 | prediction returned |
| 400 | invalid JSON, non-object body, missing / unknown / non-numeric / non-finite feature (bools rejected) |
| 404 / 405 | unknown route / wrong method (JSON body) |
| 500 | unexpected error (generic message, details only in logs) |

## Tests
```bash
pip install -r requirements-dev.txt
pytest
```

## Notes
- `scripts/make_sample_model.py` only regenerates the provided model file for this repo; it is not used by the service.
- Model files are loaded with joblib (pickle): only load files you trust.
- The model loads once at startup; a missing file stops the service immediately.
- AI usage is disclosed in `chat.md`.
