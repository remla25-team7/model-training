import time
import pytest
import joblib

@pytest.fixture(scope="module")
def vectorizer():
    return joblib.load("artifacts/vectorizer.pkl")

@pytest.fixture(scope="module")
def model():
    return joblib.load("artifacts/model.pkl")

def test_inference_time(model, vectorizer):

    review = "The staff was friendly and the food was amazing!"

    start = time.time()
    _ = model.predict(vectorizer.transform([review]))
    duration = time.time() - start

    assert duration < 0.5, f"Inference took too long: {duration:.3f} seconds"