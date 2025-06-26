import os
import psutil
import joblib

def test_memory_usage():
    # Measure process memory before loading model
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss

    # Load artifacts and run inference
    vectorizer = joblib.load('artifacts/vectorizer.pkl')
    model = joblib.load('artifacts/model.pkl')
    sample_text = "The food was excellent and the waiter was very prompt."
    X = vectorizer.transform([sample_text])
    _ = model.predict(X)

    # Measure process memory after inference
    mem_after = proc.memory_info().rss

    # Assert memory usage stays under threshold (500 MB)
    assert mem_after < 500 * 1024 * 1024, \
        f"Memory usage too high: {mem_after / (1024 ** 2):.2f} MB > 500 MB"
