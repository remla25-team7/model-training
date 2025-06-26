import joblib
import numpy as np

def test_prediction_determinism():
    # Load artifacts
    vectorizer = joblib.load('artifacts/vectorizer.pkl')
    model = joblib.load('artifacts/model.pkl')

    # Sample input to test
    sample_text = "The food was excellent and the waiter was very prompt."
    X = vectorizer.transform([sample_text])

    # Predict multiple times
    preds = [model.predict(X)[0] for _ in range(10)]
    # Ensure all predictions are identical
    assert all(p == preds[0] for p in preds), f"Predictions vary across runs: {preds}"

    # Check probability outputs are identical
    base_proba = model.predict_proba(X)[0]
    for i in range(10):
        proba = model.predict_proba(X)[0]
        assert np.allclose(proba, base_proba), f"Probabilities vary across runs at iteration {i}: {proba}"