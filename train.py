import os
import joblib
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from lib_ml.preprocessing import clean_review, tokenize_review
from lib_ml.version import __version__  # make sure you’ve published lib-version too

def generate_model_url(model_filename: str) -> str:
    """
    Construct the GitHub Releases URL for a versioned model artifact.
    Expects:
      - GITHUB_OWNER: GitHub org/user (default fallback)
      - GITHUB_REPO: repo name (default fallback)
      - MODEL_VERSION: semver tag (falls back to lib-version)
    """
    owner   = os.getenv("GITHUB_OWNER",   "remla25-team7")
    repo    = os.getenv("GITHUB_REPO",     "model-training")
    version = os.getenv("MODEL_VERSION",   __version__)
    artifact = os.path.basename(model_filename)
    return (
        f"https://github.com/{owner}/{repo}"
        f"/releases/download/v{version}/{artifact}"
    )

def train_model(
    data_path: str,
    vectorizer_out: str,
    model_out: str
):
    # 1. Load TSV dataset
    df = pd.read_csv(data_path, delimiter='\t', quoting=3)
    reviews = df['Review'].tolist()
    labels  = df['Liked'].values

    # 2. Build CountVectorizer with our preprocessing
    cv = CountVectorizer(
        tokenizer    = tokenize_review,
        preprocessor = clean_review,
        max_features = 1420
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

    # 6. Persist artifacts
    joblib.dump(cv, vectorizer_out)
    joblib.dump(clf, model_out)

    # 7. Emit the public URL for this versioned model
    model_url = generate_model_url(model_out)
    print(f"Model available at: {model_url}")
    return model_url
