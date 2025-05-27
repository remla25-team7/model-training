import yaml

RUBRIC_STRUCTURE = {
    "data": 7,
    "model": 7,
    "infra": 7,
    "monitoring": 7
}

def main():
    # Load ml_test_score.yaml
    with open("ml_test_score.yaml", "r") as f:
        test_data = yaml.safe_load(f)

    scores = {}
    for section in RUBRIC_STRUCTURE:
        tests = test_data.get(section, {})
        section_score = sum(1.0 for status in tests.values() if status == "auto") + \
                        sum(0.5 for status in tests.values() if status == "manual")
        scores[section] = section_score

    final_score = min(scores.values())

    # Terminal output
    print(" ML Test Score Breakdown:")
    for section in RUBRIC_STRUCTURE:
        out_of = RUBRIC_STRUCTURE[section]
        actual = scores.get(section, 0)
        print(f"- {section.capitalize()}: {actual:.1f} / {out_of}")

    print(f" Final ML Test Score: {final_score:.1f} / 7")

    # Export to badge input
    with open("ml_test_score.txt", "w") as f:
        f.write(f"{final_score:.1f}/7")

if __name__ == "__main__":
    main()
