import time
import joblib

def test_feature_extraction_latency():
    # Load the vectorizer
    vectorizer = joblib.load('artifacts/vectorizer.pkl')

    # Prepare a batch of sample texts
    sample_texts = ["The service was quick and the food was delicious." for _ in range(1000)]

    # Measure vectorization time
    start = time.time()
    _ = vectorizer.transform(sample_texts)
    elapsed = time.time() - start

    # Assert that feature extraction is sufficiently fast
    assert elapsed < 0.5, f"Feature extraction too slow: {elapsed:.2f}s (> 0.5s)"
