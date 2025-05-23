import argparse
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate(model_path: str, vectorizer_path: str, test_data_path: str) -> None:
    # Load artifacts
    vec = joblib.load(vectorizer_path)
    clf = joblib.load(model_path)

    # Load test set
    df = pd.read_csv(test_data_path)
    reviews = df["Review"].tolist()
    y_true  = df["Liked"].values

    # Feature extraction
    X_test = vec.transform(reviews)

    # Predictions & metrics
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 score: {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained sentiment model")
    parser.add_argument("--model",      required=True, help="Path to saved model .pkl")
    parser.add_argument("--vectorizer", required=True, help="Path to saved vectorizer .pkl")
    parser.add_argument("--test-data",  required=True, help="Path to processed test CSV")
    args = parser.parse_args()

    evaluate(
        model_path       = args.model,
        vectorizer_path  = args.vectorizer,
        test_data_path   = args.test_data
    )
