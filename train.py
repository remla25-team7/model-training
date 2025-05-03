import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from lib_ml.preprocessing import clean_text



def train_model():
    """
    Train a sentiment analysis model using logistic regression.
    """
    # Dummy training data
    texts = ["Great food!", "Terrible service."]
    labels = [1, 0]  # 1=positive, 0=negative

    # Preprocess
    X = [clean_text(text) for text in texts]

    # Vectorize
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    # Train
    model = LogisticRegression()
    model.fit(X_vec, labels)

    # Save model + vectorizer together
    joblib.dump((model, vectorizer), "sentiment_model.pkl")

    return model, vectorizer



def test_model(model, vectorizer, text):
    """
    Test the model with a new text input.
    """
    # Preprocess the text
    text_clean = clean_text(text)
    
    # Vectorize the text
    text_vec = vectorizer.transform([text_clean])
    
    # Predict
    prediction = model.predict(text_vec)
    
    return prediction[0] 

# Example usage of the test_model function
if __name__ == "__main__":
    # Train the model
    #model, vectorizer = train_model()

    # Load the model and vectorizer
    model, vectorizer = joblib.load("sentiment_model.pkl")


    # Test the model with a new text input
    test_text = "The food was amazing!"
    prediction = test_model(model, vectorizer, test_text)
    
    print(f"Prediction for '{test_text}': {'Positive' if prediction == 1 else 'Negative'}")

