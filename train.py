# train.py
import os
import sys
import argparse
import subprocess
import joblib
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from lib_ml.preprocessing import clean_review, tokenize_review

def train_and_save(data_path, vec_out, model_out):
    df = pd.read_csv(data_path, delimiter='\t', quoting=3)
    reviews = df['Review'].tolist()
    labels  = df['Liked'].values

    cv = CountVectorizer(
        tokenizer    = tokenize_review,
        preprocessor = clean_review,
        max_features = 1420
    )
    X = cv.fit_transform(reviews).toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=0
    )
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")

    joblib.dump(cv, vec_out)
    joblib.dump(clf, model_out)

def create_github_release(tag, assets):
    # Requires GH_TOKEN in env, and gh CLI installed
    cmd = ["gh", "release", "create", tag] + assets + [
        "--title", tag,
        "--notes", f"Automated model build for {tag}"
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       required=True)
    p.add_argument("--vectorizer", required=True)
    p.add_argument("--model",      required=True)
    args = p.parse_args()

    # 1) Train and dump artifacts
    os.makedirs(os.path.dirname(args.vectorizer), exist_ok=True)
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    train_and_save(args.data, args.vectorizer, args.model)

    # 2) Figure out the tag (must be run in a tagged workflow)
    ref = os.getenv("GITHUB_REF", "")
    if not ref.startswith("refs/tags/"):
        print("ERROR: GITHUB_REF is not a tag:", ref, file=sys.stderr)
        sys.exit(1)
    tag = ref.split("/", 2)[-1]

    # 3) Create release & upload
    create_github_release(tag, [args.vectorizer, args.model])

    # 4) Print public URLs
    repo = os.getenv("GITHUB_REPOSITORY")  # e.g. "owner/repo"
    base = f"https://github.com/{repo}/releases/download/{tag}"
    print("Vectorizer URL:", f"{base}/{os.path.basename(args.vectorizer)}")
    print("Model URL:     ", f"{base}/{os.path.basename(args.model)}")

if __name__ == "__main__":
    main()
