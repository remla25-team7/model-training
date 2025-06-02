"""
Test suite for evaluating model robustness against typos and spelling variations.
These tests ensure that the model can handle common spelling mistakes and text variations
while maintaining consistent predictions.
"""

import pytest
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import random
import string

@pytest.fixture(scope="module")
def vectorizer():
    return joblib.load("artifacts/vectorizer.pkl")

@pytest.fixture(scope="module")
def model():
    return joblib.load("artifacts/model.pkl")

@pytest.fixture(scope="module")
def test_data():
    return pd.read_csv("data/processed/test.csv")

def introduce_typos(text, typo_probability=0.1):
    """
    Introduce random typos into text with given probability.
    
    """
    if not text:
        return text
        
    result = list(text)
    i = 0
    while i < len(result):
        if random.random() < typo_probability:
            typo_type = random.choice(['delete', 'insert', 'substitute', 'transpose'])
            
            if typo_type == 'delete' and len(result) > 1:
                result.pop(i)
                continue  # Skip increment since we removed a character
            elif typo_type == 'insert':
                result.insert(i, random.choice(string.ascii_lowercase))
                i += 1  # Skip the inserted character
            elif typo_type == 'substitute':
                result[i] = random.choice(string.ascii_lowercase)
            elif typo_type == 'transpose' and i < len(result) - 1:
                result[i], result[i + 1] = result[i + 1], result[i]
                i += 1  # Skip the transposed character
        i += 1
                
    return ''.join(result)

def test_typo_robustness(vectorizer, model, test_data):
    """
    Test model's robustness against typos in input text.
    
    This test:
    1. Takes a sample of test reviews
    2. Introduces random typos
    3. Compares predictions between original and typo-ridden text
    4. Ensures prediction changes are within acceptable limits
    """

    sample_size = min(50, len(test_data))
    sample_indices = random.sample(range(len(test_data)), sample_size)
    sample_reviews = test_data.iloc[sample_indices]["Review"].tolist()
    
    # Get original predictions
    X_original = vectorizer.transform(sample_reviews)
    original_predictions = model.predict(X_original)
    
    # Introduce typos and get new predictions
    typo_reviews = [introduce_typos(review, typo_probability=0.03) for review in sample_reviews]  # Reduced typo probability
    X_typo = vectorizer.transform(typo_reviews)
    typo_predictions = model.predict(X_typo)
    
    prediction_changes = np.sum(original_predictions != typo_predictions)
    change_rate = prediction_changes / sample_size
    
    assert change_rate < 0.20, f"Too many prediction changes ({change_rate:.2%}) due to typos"

def test_common_typo_patterns(vectorizer, model):
    """
    Test model's robustness against common typo patterns.

    """
    test_cases = [
        ("The food was amazing!", "The fd ws mzng!", 1),  # Missing vowels
        ("The service was bad", "The servvice was badd", 0),  # Double letters 
        ("Great atmosphere and friendly staff", "Great atnosphere and friendly stafg", 1),  # Keyboard typos
        ("Excellent food and service", "Excelllent food and servvice", 1),  # Double letters 
        ("The food was delicious", "The food was deliciuos", 1),  # Common spelling mistake
        ("The restaurant was nice", "The resturant was nise", 1),  # Common spelling mistakes
        ("The food was good", "The food was gud", 1),  # Common abbreviation
    ]
    
    # Track how many predictions changed
    changed_predictions = 0
    total_predictions = len(test_cases)
    
    for original, typo, expected in test_cases:
        # Get predictions for both versions
        X_original = vectorizer.transform([original])
        X_typo = vectorizer.transform([typo])
        
        original_pred = model.predict(X_original)[0]
        typo_pred = model.predict(X_typo)[0]
        
        if original_pred != typo_pred:
            changed_predictions += 1
            print(f"Prediction changed for: '{original}' -> '{typo}'")
            print(f"Original prediction: {original_pred}, Typo prediction: {typo_pred}")
    
    change_rate = changed_predictions / total_predictions
    assert change_rate < 0.25, f"Too many prediction changes ({change_rate:.2%}) due to typos"

def test_typo_impact_on_confidence(vectorizer, model, test_data):
    """
    Test that typos don't significantly impact model confidence.
    
    This test ensures that introducing typos doesn't cause
    large changes in prediction probabilities.
    """
    sample_size = min(30, len(test_data))
    sample_indices = random.sample(range(len(test_data)), sample_size)
    sample_reviews = test_data.iloc[sample_indices]["Review"].tolist()
    
    X_original = vectorizer.transform(sample_reviews)
    original_probs = model.predict_proba(X_original)
    
    # Introduce typos and get new probabilities
    typo_reviews = [introduce_typos(review, typo_probability=0.03) for review in sample_reviews]  # Reduced typo probability
    X_typo = vectorizer.transform(typo_reviews)
    typo_probs = model.predict_proba(X_typo)
    
    # Calculate maximum probability change
    max_prob_change = np.max(np.abs(original_probs - typo_probs))
    
    assert max_prob_change < 0.45, \
        f"Typos caused too large probability changes (max change: {max_prob_change:.2f})" 