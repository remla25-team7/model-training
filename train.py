import os
import argparse
import joblib
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from lib_ml.preprocessing import clean_review, tokenize_review


def train_and_save(data_path: str, vec_out: str, model_out: str) -> None:
    df = pd.read_csv(data_path, delimiter="\t", quoting=3)
    reviews = df["Review"].tolist()
    labels = df["Liked"].values

    cv = CountVectorizer(
        tokenizer=tokenize_review,
        preprocessor=clean_review,
        max_features=1420,
    )
    X = cv.fit_transform(reviews).toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=0
    )
    clf = GaussianNB().fit(X_train, y_train)
    print(f"Test accuracy: {clf.score(X_test, y_test):.4f}")

    os.makedirs(os.path.dirname(vec_out), exist_ok=True)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(cv, vec_out)
    joblib.dump(clf, model_out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--vectorizer", required=True)
    p.add_argument("--model", required=True)
    args = p.parse_args()
    train_and_save(args.data, args.vectorizer, args.model)
