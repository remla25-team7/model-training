"""Training script for sentiment model."""

import os
import argparse
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from lib_ml.preprocessing import clean_review, tokenize_review

def train_and_save(
    data_path: str,
    vec_out: str,
    model_out: str
) -> None:
    """Train a sentiment model and save the vectorizer and model."""
    df = pd.read_csv(data_path)
    reviews, labels = df["Review"].tolist(), df["Liked"].values

    vec = TfidfVectorizer(
        tokenizer=tokenize_review,
        preprocessor=clean_review,
        ngram_range=(1, 2),
        max_features=5000
    )
    x = vec.fit_transform(reviews)

    clf = LogisticRegression(
        solver="liblinear",
        random_state=0
    ).fit(x, labels)

    acc = clf.score(x, labels)
    print(f"Train accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(vec_out), exist_ok=True)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(vec, vec_out)
    joblib.dump(clf, model_out)

    # Save training accuracy for DVC
    with open("output/train_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"train_accuracy": acc}, f)

def main():
    """Parse arguments and run training."""
    p = argparse.ArgumentParser(description="Train a sentiment model.")
    p.add_argument("--data", required=True, help="Path to training data CSV.")
    p.add_argument("--vectorizer", required=True, help="Path to save vectorizer .pkl.")
    p.add_argument("--model", required=True, help="Path to save model .pkl.")
    args = p.parse_args()
    train_and_save(args.data, args.vectorizer, args.model)

if __name__ == "__main__":
    main()
