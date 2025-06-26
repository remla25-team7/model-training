# Code Smell Report

## Linting with Pylint

```plaintext
************* Module download_data
src/download_data.py:4:0: C0301: Line too long (117/100) (line-too-long)
************* Module data_prep
src/data_prep.py:18:8: C0103: Variable name "df" doesn't conform to snake_case naming style (invalid-name)
src/data_prep.py:19:4: C0103: Variable name "e" doesn't conform to snake_case naming style (invalid-name)
src/data_prep.py:36:0: C0116: Missing function or method docstring (missing-function-docstring)
************* Module train
src/train.py:1:0: F0002: src/train.py: Fatal error while checking 'src/train.py'. Please open an issue in our bug tracker so we address this. There is a pre-filled template that you can use in '/Users/mabhatti/Library/Caches/pylint/pylint-crash-2025-06-26-23-18-51.txt'. (astroid-error)
************* Module evaluate
src/evaluate.py:14:4: C0103: Variable name "df" doesn't conform to snake_case naming style (invalid-name)
src/evaluate.py:24:4: C0103: Variable name "f1" doesn't conform to snake_case naming style (invalid-name)
src/evaluate.py:32:63: C0103: Variable name "f" doesn't conform to snake_case naming style (invalid-name)
src/evaluate.py:1:0: C0411: standard import "import json" should be placed before "import joblib" (wrong-import-order)
src/evaluate.py:2:0: C0411: standard import "import argparse" should be placed before "import joblib" (wrong-import-order)
src/evaluate.py:2:0: C0412: Imports from package argparse are not grouped (ungrouped-imports)
src/evaluate.py:3:0: C0412: Imports from package pandas are not grouped (ungrouped-imports)
src/evaluate.py:4:0: C0412: Imports from package joblib are not grouped (ungrouped-imports)
src/evaluate.py:5:0: C0412: Imports from package sklearn are not grouped (ungrouped-imports)

-----------------------------------
Your code has been rated at 0.00/10

```
