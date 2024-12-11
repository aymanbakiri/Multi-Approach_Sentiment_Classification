import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utilities import * 
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments


def main():
    # Define paths
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    pos_train_path = os.path.join(data_dir, "train_pos_full.txt")
    neg_train_path = os.path.join(data_dir, "train_neg_full.txt")
    test_path = os.path.join(data_dir, "test_data.txt")
    submission_path = os.path.join(os.path.dirname(__file__), "submission.csv")

    # Load data
    pos_tweets, neg_tweets = load_data(pos_train_path, neg_train_path)

    # Preprocess data
    tweets, labels = preprocess_data(pos_tweets, neg_tweets)

    

    method = 'NeuralNetwork'

    if method == 'regression':

        # Build features with tf idf 
        #features, vectorizer = build_features_tfidf(tweets)

        # Define the path to GloVe embeddings
        glove_path = os.path.join(os.path.dirname(__file__), "glove.twitter.27B.100d.txt")


        # Load GloVe embeddings
        embeddings_index = load_glove_embeddings(glove_path)

        # Generate tweet embeddings
        embedding_dim = 100  # Match the dimension of the GloVe file
        features = get_tweet_embeddings(tweets, embeddings_index, embedding_dim=embedding_dim)



        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Train the classifier
        classifier = train_classifier(X_train, y_train)

        # Evaluate on the validation set
        val_predictions = classifier.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        # Load test data
        with open(test_path, 'r', encoding='utf-8') as f:
            test_tweets = f.readlines()

        predict_and_save_glove(test_tweets, embeddings_index, classifier, submission_path)

        print(f"Predictions saved to {submission_path}")
    
    elif method == 'NeuralNetwork':

        # Define paths for FastText
        fasttext_model_path = os.path.join(os.path.dirname(__file__), "fasttext_model.bin")

        # Train FastText model
        print("Training FastText model...")
        fasttext_model = train_fasttext(tweets, labels, fasttext_model_path)

        # Evaluate on validation set
        X_train, X_val, y_train, y_val = train_test_split(tweets, labels, test_size=0.2, random_state=42)
        val_predictions = [
        int(fasttext_model.predict(tweet.strip())[0][0].replace("__label__", ""))

        for tweet in X_val
        ]

        val_accuracy = accuracy_score(y_val, val_predictions)
        print(f"Validation Accuracy (FastText): {val_accuracy:.4f}")

        # Load test data
        with open(test_path, 'r', encoding='utf-8') as f:
            test_tweets = f.readlines()
        

        # Predict on test data and save results
        predict_fasttext(test_tweets, fasttext_model_path, submission_path)

        
    


if __name__ == "__main__":
    main()
