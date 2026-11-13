import json

import pytest

from app.main import DEFAULT_MODEL_PATH, create_app, load_model

VALID = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = False  # exercise the real 500 handler
    return app.test_client()


def test_predict_ok(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    assert r.get_json() == {"prediction": "setosa", "class_id": 0}


def test_predict_other_class_and_ints(client):
    r = client.post("/predict", json={"sepal_length": 6, "sepal_width": 3, "petal_length": 5, "petal_width": 2})
    assert r.status_code == 200 and r.get_json()["prediction"] == "virginica"


@pytest.mark.parametrize("missing", list(VALID))
def test_missing_feature(client, missing):
    body = {k: v for k, v in VALID.items() if k != missing}
    r = client.post("/predict", json=body)
    assert r.status_code == 400 and missing in r.get_json()["message"]


@pytest.mark.parametrize("bad", ["5.1", None, True, [1], {"a": 1}])
def test_wrong_type(client, bad):
    r = client.post("/predict", json={**VALID, "sepal_length": bad})
    assert r.status_code == 400


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_non_finite(client, bad):
    raw = json.dumps(VALID).replace("5.1", bad)
    r = client.post("/predict", data=raw, content_type="application/json")
    assert r.status_code == 400


def test_unknown_feature(client):
    assert client.post("/predict", json={**VALID, "extra": 1}).status_code == 400


@pytest.mark.parametrize("body", [[], [1, 2], "text", 5])
def test_non_object_body(client, body):
    assert client.post("/predict", json=body).status_code == 400


def test_invalid_json_and_empty_body(client):
    assert client.post("/predict", data="{oops", content_type="application/json").status_code == 400
    assert client.post("/predict").status_code == 400


def test_wrong_method_and_route(client):
    assert client.get("/predict").status_code == 405
    assert client.get("/nope").status_code == 404


def test_health(client):
    assert client.get("/health").get_json() == {"status": "ok"}


def test_unexpected_error_returns_500(client):
    class Boom:
        def predict(self, _):
            raise RuntimeError("model exploded")

    client.application.config["MODEL"] = Boom()
    r = client.post("/predict", json=VALID)
    assert r.status_code == 500 and r.get_json()["error"] == "internal_server_error"
    assert "exploded" not in r.get_data(as_text=True)  # no internals leaked


def test_missing_model_file_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_app(tmp_path / "missing.joblib")


def test_load_model():
    model, classes = load_model(DEFAULT_MODEL_PATH)
    assert classes == ["setosa", "versicolor", "virginica"] and hasattr(model, "predict")
