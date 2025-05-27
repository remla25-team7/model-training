# Restaurant Sentiment — Model-Training

This repository contains the data-prep, training, and evaluation stages for a simple restaurant-review sentiment model.

---

## Installation (local)

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-org/model-training.git
   cd model-training
   ```

2. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

3. **Reproducing and Rolling back the Pipeline**

   1. Follow the Cloud Remote Setup described bellow
   2. dvc repro
   3. To roll back to a previous version of the pipeline:
      git checkout <commit-hash>
      dvc checkout
   4. To see the current metrics: dvc metrics show
   5. Compare metrics across experiments: dvc exp show

## Cloud Remote Setup (Google Drive)

This project uses a cloud-based remote on Google Drive to store data files and model artifacts using DVC.

If using a service account:

Ensure that the service account key file (.json) is downloaded from the Google Cloud Console.

Share the target Google Drive folder with the service account's email (visible under "client_email" in the JSON file).

Configure DVC to use the service account by running:

```bash
dvc remote modify myremote gdrive_use_service_account true
dvc remote modify myremote gdrive_service_account_json_file_path /absolute/path/to/your-key.json
```

dvc pull # Download data/models from Google Drive
dvc push # Upload new data/models to Google Drive

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

---

## Testing, Linting, and Coverage

- **Run tests:**
  ```bash
  pytest
  ```
- **Run linting:**
  ```bash
  pylint src/
  ```
- **Check coverage:**
  ```bash
  coverage run -m pytest
  coverage report
  ```

## Test Coverage

![coverage](https://github.com/remla25-team7/model-training/blob/badge-badges/coverage.svg)
![Pylint Score](https://github.com/remla25-team7/model-training/blob/badge-badges/pylint.svg)
![Test Adequacy](https://img.shields.io/badge/tests-ML_Test_Score_Complete-blue)





To check coverage locally:

```bash
coverage run -m pytest
coverage report
coverage html
```