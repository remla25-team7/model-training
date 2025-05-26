import pytest
import joblib

@pytest.fixture(scope="module")
def vectorizer():
    return joblib.load("artifacts/vectorizer.pkl")

@pytest.fixture(scope="module")
def model():
    return joblib.load("artifacts/model.pkl")

def test_negation_robustness(vectorizer, model):
    original = "The food was good"
    negated = "The food was not good"
    X_orig = vectorizer.transform([original])
    X_neg = vectorizer.transform([negated])
    pred_orig = model.predict(X_orig)[0]
    pred_neg = model.predict(X_neg)[0]
    assert pred_orig != pred_neg, "Negation did not change the prediction as expected"

# def test_negation_positive_to_negative(vectorizer, model):
#     original = "The service was excellent"
#     negated = "The service was not excellent"
#     X_orig = vectorizer.transform([original])
#     X_neg = vectorizer.transform([negated])
#     pred_orig = model.predict(X_orig)[0]
#     pred_neg = model.predict(X_neg)[0]
#     assert pred_orig != pred_neg, "Negation did not change the prediction as expected (positive to negative)"
