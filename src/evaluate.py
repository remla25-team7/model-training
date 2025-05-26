"""Evaluation script for sentiment model. """

import json
import argparse
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

def evaluate(model_path: str, vectorizer_path: str, test_data_path: str) -> None:
    """Evaluate a trained model and save metrics to output/metrics.json."""
    # Load artifacts
    vec = joblib.load(vectorizer_path)
    clf = joblib.load(model_path)

    # Load test set
    df = pd.read_csv(test_data_path)
    reviews = df["Review"].tolist()
    y_true = df["Liked"].values

    # Feature extraction
    x_test = vec.transform(reviews)

    # Predictions & metrics
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, clf.predict_proba(x_test)[:, 1])

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 score: {f1:.4f}")

    with open("output/metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc
        }, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained sentiment model")
    parser.add_argument("--model", required=True, help="Path to saved model .pkl")
    parser.add_argument("--vectorizer", required=True, help="Path to saved vectorizer .pkl")
    parser.add_argument("--test-data", required=True, help="Path to processed test CSV")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        vectorizer_path=args.vectorizer,
        test_data_path=args.test_data
    )
