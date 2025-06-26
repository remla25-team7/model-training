import os
import requests

RAW_URL = "https://raw.githubusercontent.com/proksch/restaurant-sentiment/main/a1_RestaurantReviews_HistoricDump.tsv"
OUT_PATH = "data/raw/reviews.tsv"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
resp = requests.get(RAW_URL, timeout=30)
resp.raise_for_status()
with open(OUT_PATH, "wb") as f:
    f.write(resp.content)
print(f"Downloaded raw data to {OUT_PATH}")
