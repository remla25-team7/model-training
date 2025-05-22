# Restaurant Sentiment — Model-Training

This repository contains the data-prep, training, and evaluation stages for a simple restaurant-review sentiment model.

---

## Installation (local)

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-org/model-training.git
   cd model-training
    ````

2. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ````
---

## Running the pipeline locally

All pipeline code lives under `src/`. By default we download data into `data/raw/` and write processed splits into `data/processed/`, and artifacts into `artifacts/`.

1. **Download & split data**

   ```bash
   python src/data_prep.py \
     --raw-out   data/raw/a1_RestaurantReviews.tsv \
     --train-out data/processed/train.csv \
     --test-out  data/processed/test.csv
   ```

2. **Train the model**

   ```bash
   python src/train.py \
     --data       data/processed/train.csv \
     --vectorizer artifacts/vectorizer.pkl \
     --model      artifacts/model.pkl
   ```

3. **Evaluate on held-out test set**

   ```bash
   python src/evaluate.py \
     --model      artifacts/model.pkl \
     --vectorizer artifacts/vectorizer.pkl \
     --test-data  data/processed/test.csv
   ```

You should see printed metrics (accuracy, F1).

---

## Running the pipeline remotely (CI/CD)

A GitHub Actions workflow (`.github/workflows/train_publish_model.yml`) is configured to run **on every new tag** matching `v*.*.*`. It:

### To trigger a remote run

```bash
# create and push a new version tag
git tag v0.1.0
git push origin v0.1.0
```

Then visit the **Actions** tab in GitHub to watch the workflow. When it succeeds, your artifacts will be attached to release **v0.1.0**.
