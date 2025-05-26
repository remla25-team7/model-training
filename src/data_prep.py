"""Data preparation script for sentiment model. Downloads, splits, and validates data."""

import os
import argparse
import requests
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_URL = (
    "https://raw.githubusercontent.com/"
    "proksch/restaurant-sentiment/main/"
    "a1_RestaurantReviews_HistoricDump.tsv"
)


def download_data(url: str, out_path: str) -> None:
    """Download data from a URL and save to out_path."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"Downloaded raw data to {out_path}")


def split_data(
    raw_path: str,
    train_out: str,
    test_out: str,
    test_size: float = 0.2,
    random_state: int = 0,
) -> None:
    """Split raw data into train and test sets and save as CSV."""
    try:
        df = pd.read_csv(raw_path, sep="\t", quoting=3)
    except Exception as e:
        raise RuntimeError(f"Failed to read TSV at {raw_path}: {e}") from e

    expected = {"Review", "Liked"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Expected columns {expected}, got {set(df.columns)}")
    
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["Liked"]
    )
    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    os.makedirs(os.path.dirname(test_out), exist_ok=True)
    train.to_csv(train_out, index=False, encoding="utf-8")
    test.to_csv(test_out,   index=False, encoding="utf-8")
    print(f"Wrote {len(train)} train / {len(test)} test samples")

def main():
    """Download and split data for sentiment analysis."""
    p = argparse.ArgumentParser(description="Download and split restaurant review data.")
    p.add_argument(
        "--raw-out", default="data/raw/reviews.tsv",
        help="where to save the downloaded TSV"
    )
    p.add_argument(
        "--train-out", default="data/processed/train.csv",
        help="where to save the train split"
    )
    p.add_argument(
        "--test-out", default="data/processed/test.csv",
        help="where to save the test split"
    )
    args = p.parse_args()

    download_data(RAW_URL, args.raw_out)
    split_data(args.raw_out, args.train_out, args.test_out)

if __name__ == "__main__":
    main()
