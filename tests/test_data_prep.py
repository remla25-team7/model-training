#Smoke-test data_prep.py
def test_data_prep_runs():
    import subprocess
    result = subprocess.run(
        ["python", "src/data_prep.py", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0

def test_failure_example():
    assert 1 == 2  # This will definitely fail