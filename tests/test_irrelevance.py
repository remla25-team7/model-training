import pytest
import joblib

@pytest.fixture(scope="module")
def vectorizer():
    return joblib.load("artifacts/vectorizer.pkl")

@pytest.fixture(scope="module")
def model():
    return joblib.load("artifacts/model.pkl")

def test_irrelevance(vectorizer, model):
    """
    Test that the model does not confidently predict sentiment for irrelevant or nonsensical input.
    """
    irrelevant_inputs = [
        "asdfghjkl",
        "1234567890",
        "!@#$%^&*()",
        "lorem ipsum dolor sit amet",
        "",
        "The quick brown fox jumps over the lazy dog"
    ]
    X_irrelevant = vectorizer.transform(irrelevant_inputs)
    probs = model.predict_proba(X_irrelevant)
    for i, p in enumerate(probs):
        max_prob = max(p)
        assert max_prob < 0.6, f"Model is too confident ({max_prob:.2f}) for irrelevant input: '{irrelevant_inputs[i]}'" 
        
def test_irrelevance_metamorphic(vectorizer, model):
    """
    Test that adding irrelevant information (e.g., a URL) does not change the sentiment prediction.
    """
    base_review = "The food was great!"
    irrelevant_review = "The food was great! Visit www.myrestaurant.com to get a discount! Maybe we will see you there!"
    X_base = vectorizer.transform([base_review])
    X_irrelevant = vectorizer.transform([irrelevant_review])
    pred_base = model.predict(X_base)[0]
    pred_irrelevant = model.predict(X_irrelevant)[0]
    assert pred_base == pred_irrelevant, "Irrelevant information changed the prediction"