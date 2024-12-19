import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocess import tweets_to_glove_features
from sklearn.model_selection import train_test_split
import fasttext


# Glove

def evaluate_and_predict_glove(model, X_val, y_val, test_tweets, glove_embeddings, embedding_dim, submission_path):
    """
    Evaluate the model, predict on the test set, and save predictions.
    Args:
        model: Trained machine learning model.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        test_tweets (list): List of test tweets.
        glove_embeddings (dict): Preloaded GloVe embeddings.
        embedding_dim (int): Dimension of GloVe embeddings.
        submission_path (str): Path to save the submission CSV file.
    Returns:
        float: Validation accuracy.
    """
    # Evaluate the model on validation set
    print("Evaluating model on validation set...")
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
   

    # Predict on test set
    print("Predicting on test set...")
    test_features = tweets_to_glove_features(test_tweets, glove_embeddings, embedding_dim=embedding_dim)
    test_predictions = model.predict(test_features)

    # Convert predictions to {-1, 1} and save to CSV
    submission_predictions = np.where(test_predictions == 0, -1, 1)
    test_ids = range(1, len(test_predictions) + 1)
    submission = pd.DataFrame({'Id': test_ids, 'Prediction': submission_predictions})
    submission.to_csv(submission_path, index=False)
    print(f"Predictions saved to {submission_path}")

    # Return validation accuracy for consistency
    return val_accuracy

# TF-IDF

def evaluate_and_predict_tfidf(model, X_val, y_val, test_tweets, vectorizer, submission_path):
    """
    Evaluate the model on validation data, predict on the test set, and save predictions.
    Args:
        model: Trained machine learning model.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        test_tweets (list): List of test tweets.
        vectorizer: Fitted vectorizer to transform text data.
        submission_path (str): Path to save the submission CSV file.
    Returns:
        float: Validation accuracy.
    """
    # Evaluate on validation set
    print("Evaluating the model on validation set...")
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
   
    
    # Predict on test set
    print("Predicting on test set...")
    test_features = vectorizer.transform(test_tweets)
    test_predictions = model.predict(test_features)

    # Convert predictions to {-1, 1} and save to CSV
    submission_predictions = np.where(test_predictions == 0, -1, 1)
    test_ids = range(1, len(test_tweets) + 1)
    submission = pd.DataFrame({'Id': test_ids, 'Prediction': submission_predictions})
    submission.to_csv(submission_path, index=False)
    print(f"Predictions saved to {submission_path}")
    
    return val_accuracy


# FastText

def evaluate_fasttext(tweets, labels, model_path):
    """
    Evaluate a FastText model on validation data.
    Args:
        tweets: List of tweets.
        labels: List of labels.
        model_path: Path to the saved FastText model.
    Returns:
        float: Validation accuracy.
    """
    model = fasttext.load_model(model_path)
    X_train, X_val, y_train, y_val = train_test_split(tweets, labels, test_size=0.2, random_state=42)

    val_predictions = [
        int(model.predict(tweet.strip())[0][0].replace("__label__", ""))
        for tweet in X_val
    ]
    return accuracy_score(y_val, val_predictions)

def predict_fasttext(test_tweets, model_path, output_path):
    """
    Use a trained FastText model to predict on test data and save results.
    """
    # Load the trained model
    model = fasttext.load_model(model_path)

    # Predict labels for test tweets
    predictions = []
    for tweet in test_tweets:
        clean_tweet = tweet.strip()
        label, _ = model.predict(clean_tweet)
        predictions.append(int(label[0].replace("__label__", "")))

    # Convert predictions from {0, 1} to {-1, 1}
    predictions = [-1 if p == 0 else 1 for p in predictions]

    # Save predictions
    test_ids = range(1, len(test_tweets) + 1)
    submission = pd.DataFrame({'Id': test_ids, 'Prediction': predictions})
    submission.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")