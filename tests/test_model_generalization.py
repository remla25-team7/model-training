import pytest
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

@pytest.fixture(scope="module")
def vectorizer():
    return joblib.load("artifacts/vectorizer.pkl")

@pytest.fixture(scope="module")
def model():
    return joblib.load("artifacts/model.pkl")

@pytest.fixture(scope="module")
def train_data():
    return pd.read_csv("data/processed/train.csv")

@pytest.fixture(scope="module")
def test_data():
    return pd.read_csv("data/processed/test.csv")

def test_generalization(vectorizer, model, train_data, test_data):
    """
    Check that model performance is similar on train and test sets.
  
    This test ensures the model isn't overfitting by comparing:
    - Accuracy gap between train and test sets (should be < 0.20)
    - Log-loss gap between train and test sets (should be < 0.30)
  
    """
    # Transform text data using the vectorizer
    X_train = vectorizer.transform(train_data["Review"])
    y_train = train_data["Liked"]
    X_test = vectorizer.transform(test_data["Review"])
    y_test = test_data["Liked"]    # Calculate performance metrics
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    train_ll = log_loss(y_train, model.predict_proba(X_train)[:, 1])
    test_ll = log_loss(y_test, model.predict_proba(X_test)[:, 1])    # Assert that performance gaps are within acceptable thresholds
    assert abs(train_acc - test_acc) < 0.20, f"Train/test accuracy gap too large: {train_acc:.3f} vs {test_acc:.3f}"
    assert abs(train_ll - test_ll) < 0.30, f"Train/test log-loss gap too large: {train_ll:.3f} vs {test_ll:.3f}"

def test_model_beats_baseline(vectorizer, model, test_data):
    """
    Ensure model outperforms a majority-class baseline.
  
    This test verifies that the model provides value beyond simply predicting
    the most common class in the dataset.
  
    """
    # Transform test data
    X = vectorizer.transform(test_data["Review"])
    y = test_data["Liked"]
  
    # Create baseline predictions (majority class)
    baseline = np.full_like(y, np.bincount(y).argmax())
  
    # Compare model accuracy with baseline accuracy
    baseline_acc = accuracy_score(y, baseline)
    model_acc = accuracy_score(y, model.predict(X))
    assert model_acc > baseline_acc, f"Model accuracy ({model_acc:.3f}) not above baseline ({baseline_acc:.3f})"

def test_hyperparameter_tuning(model):
    """
    Check that model hyperparameters are not default (indicating tuning was done).
  
    This test ensures that hyperparameter tuning was performed by checking
    if key hyperparameters have been modified from their default values.
  
    """
    if hasattr(model, "var_smoothing"):
        assert model.var_smoothing != 1e-9, "var_smoothing is default; tune your model!" 