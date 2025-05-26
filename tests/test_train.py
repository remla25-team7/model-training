#Smoke-test for train.py
def test_train_runs():
    import subprocess
    result = subprocess.run(
        ["python", "src/train.py", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0 