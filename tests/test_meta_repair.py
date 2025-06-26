import joblib
import re

def clean_input(text: str, vectorizer) -> str:
    """
    Simple repair: remove tokens not recognized by the vectorizer vocabulary.
    """
    tokens = text.split()
    vocab = vectorizer.vocabulary_
    filtered = [t for t in tokens if t.lower() in vocab]
    return " ".join(filtered)

def test_metamorphic_with_auto_repair():
    # Load artifacts
    vectorizer = joblib.load('artifacts/vectorizer.pkl')
    model = joblib.load('artifacts/model.pkl')

    # Original and mutated inputs
    original = "The food was great and the service was excellent."
    mutated = original + " qwertyuiop 12345 http://example.com"

    # Baseline prediction
    X_orig = vectorizer.transform([original])
    pred_orig = model.predict(X_orig)[0]

    # Mutated prediction
    X_mut = vectorizer.transform([mutated])
    pred_mut = model.predict(X_mut)[0]

    # If inconsistency detected, attempt repair
    if pred_mut != pred_orig:
        repaired_text = clean_input(mutated, vectorizer)
        X_rep = vectorizer.transform([repaired_text])
        pred_rep = model.predict(X_rep)[0]
        # Assert that repair restores consistency
        assert pred_rep == pred_orig, (
            f"Repair failed: mutated prediction {pred_mut}, "
            f"repaired prediction {pred_rep}, "
            f"original {pred_orig}" )
    else:
        # No inconsistency detected
        assert True
