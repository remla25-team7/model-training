import os
import joblib
import pandas as pd

from lib_ml.preprocessing import clean_review, tokenize_review
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def test_model(clf, X_test, y_test):
    """
    Tests the accuracy of the trained model on the test set.
    """
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2%}")
    return accuracy

def main():
    df = pd.read_csv("data/a1_RestaurantReviews_HistoricDump.tsv", 
                     sep="\t", quoting=3)
    texts = df.Review.tolist()
    labels = df.Liked

    # Build vectorizer & train-test split
    vect = CountVectorizer(preprocessor=clean_review,
                           tokenizer=tokenize_review)
    X = vect.fit_transform(texts).toarray()
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train classifier
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    # Test classifier
    test_model(clf, X_test, y_test)

    # Serialize artifacts
    version = os.getenv("MODEL_VERSION", "v0.1.0")
    os.makedirs("artifacts", exist_ok=True)
    vect_path = f"artifacts/vectorizer-{version}.pkl"
    modl_path = f"artifacts/classifier-{version}.pkl"
    joblib.dump(vect, vect_path)
    joblib.dump(clf, modl_path)
    print(f"Saved vectorizer → {vect_path}")
    print(f"Saved classifier → {modl_path}")



if __name__ == "__main__":
    main()