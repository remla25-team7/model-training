import os
import argparse
import joblib
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from lib_ml.preprocessing    import clean_review, tokenize_review

def train_and_save(
    data_path: str,
    vec_out: str,
    model_out: str
) -> None:
    df = pd.read_csv(data_path)
    reviews, labels = df["Review"].tolist(), df["Liked"].values

    vec = TfidfVectorizer(
        tokenizer=tokenize_review,
        preprocessor=clean_review,
        ngram_range=(1,2),       
        max_features=5000
    )
    X = vec.fit_transform(reviews)

    clf = LogisticRegression(
        solver="liblinear",      
        random_state=0
    ).fit(X, labels)

    acc = clf.score(X, labels)
    print(f"Train accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(vec_out), exist_ok=True)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(vec, vec_out)
    joblib.dump(clf, model_out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",      required=True)
    p.add_argument("--vectorizer",required=True)
    p.add_argument("--model",     required=True)
    args = p.parse_args()
    train_and_save(args.data, args.vectorizer, args.model)

if __name__ == "__main__":
    main()
