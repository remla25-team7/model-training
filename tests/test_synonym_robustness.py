import pytest
from joblib import load
import nltk
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def find_synonyms(word):
    """Return a set of synonyms for a word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            # Exclude the original word and multi-word synonyms
            if name.lower() != word.lower() and " " not in name:
                synonyms.add(name)
    return list(synonyms)

def generate_synonym_variants(sentence):
    """Generate new sentences by replacing each word with a synonym (if available)."""
    words = sentence.split()
    variants = []
    for i, word in enumerate(words):
        for synonym in find_synonyms(word):
            new_words = words.copy()
            new_words[i] = synonym
            variants.append(" ".join(new_words))
    return variants

@pytest.fixture(scope="module")
def model_and_vectorizer():
    # Adjust these paths as needed
    model = load("artifacts/model.pkl")
    vectorizer = load("artifacts/vectorizer.pkl")
    return model, vectorizer

@pytest.mark.parametrize("sentence", [
    "The food was delicious and the staff was friendly",
    "The atmosphere was cozy and inviting",
    "The food was great!",
    "The food was not bad",
    "The food was not bad",
])
def test_synonym_robustness(model_and_vectorizer, sentence):
    model, vectorizer = model_and_vectorizer
    original_vec = vectorizer.transform([sentence])
    original_pred = model.predict(original_vec)[0]

    for variant in generate_synonym_variants(sentence):
        variant_vec = vectorizer.transform([variant])
        variant_pred = model.predict(variant_vec)[0]
        assert variant_pred == original_pred, (
            f"Prediction changed!\nOriginal: '{sentence}' -> {original_pred}\n"
            f"Variant: '{variant}' -> {variant_pred}"
        ) 