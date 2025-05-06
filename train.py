import os
import pickle
import joblib
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from lib_ml.preprocessing import clean_review, tokenize_review


def train_model(
    data_path: str,
    vectorizer_out: str,
    model_out: str
):
    # 1. Load TSV dataset
    df = pd.read_csv(data_path, delimiter='\t', quoting=3)
    reviews = df['Review'].tolist()
    labels = df['Liked'].values

    # 2. Build CountVectorizer with our preprocessing
    cv = CountVectorizer(
        tokenizer=tokenize_review,
        preprocessor=clean_review,
        max_features=1420
    )
    X = cv.fit_transform(reviews).toarray()

    # 3. Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=0
    )

    # 4. Train Gaussian Naive Bayes
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    # 5. Evaluate
    acc = clf.score(X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")

    # 6. Save artifacts
    os.makedirs(os.path.dirname(vectorizer_out), exist_ok=True)
    pickle.dump(cv, open(vectorizer_out, 'wb'))
    joblib.dump(clf, model_out)
    print(f"Saved vectorizer to {vectorizer_out}")
    print(f"Saved classifier to {model_out}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/a1_RestaurantReviews_HistoricDump.tsv')
    p.add_argument('--vec_out', default='models/vectorizer.pkl')
    p.add_argument('--model_out', default='models/classifier.joblib')
    args = p.parse_args()
    train_model(args.data, args.vec_out, args.model_out)