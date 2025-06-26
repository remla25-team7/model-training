import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/reviews.tsv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

def split_data(
    raw_path: str,
    train_out: str,
    test_out: str,
    test_size: float = 0.2,
    random_state: int = 0
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
    test.to_csv(test_out, index=False, encoding="utf-8")
    print(f"Wrote {len(train)} train / {len(test)} test samples")

def main():
    split_data(RAW_PATH, TRAIN_PATH, TEST_PATH)

if __name__ == "__main__":
    main()
