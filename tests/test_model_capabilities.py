import joblib
import pytest

@pytest.fixture(scope="module")
def model_and_vectorizer():
    model = joblib.load("artifacts/model.pkl")
    vectorizer = joblib.load("artifacts/vectorizer.pkl")
    return model, vectorizer

def predict_sentiment(model, vectorizer, text):
    X = vectorizer.transform([text])
    return model.predict(X)[0]

