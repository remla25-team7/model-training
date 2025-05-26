#Smoke-test evaluate.py
def test_evaluate_runs():
    import subprocess
    result = subprocess.run(
        ["python", "src/evaluate.py", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0 