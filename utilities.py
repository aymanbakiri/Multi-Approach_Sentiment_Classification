import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from datasets import Dataset
import torch
import fasttext



# Loading and preparing the data 

def load_data(pos_path, neg_path):
    """
    Load positive and negative training tweets.
    """
    with open(pos_path, 'r', encoding='utf-8') as f:
        pos_tweets = f.readlines()
    with open(neg_path, 'r', encoding='utf-8') as f:
        neg_tweets = f.readlines()
    return pos_tweets, neg_tweets

def preprocess_data(pos_tweets, neg_tweets):
    """
    Combine and label data for supervised learning.
    """
    tweets = pos_tweets + neg_tweets
    labels = [1] * len(pos_tweets) + [0] * len(neg_tweets)
    return tweets, labels

# Tokenizing and preparing the datasets

def prepare_datasets(tweets, labels, tokenizer):
    """
    Tokenize tweets and prepare datasets for training and validation.
    """
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",   # Ensure all sequences are padded to the same length
            truncation=True,        # Truncate sequences longer than max_length
            max_length=128          # Define a fixed maximum length (e.g., 128)
        )

    # Create a dataset
    dataset = Dataset.from_dict({"text": tweets, "label": labels})

    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def build_features_tfidf(tweets):
    """
    Build text features using TF-IDF.
    """
    vectorizer = TfidfVectorizer(min_df=5, max_features=10000, stop_words='english')
    features = vectorizer.fit_transform(tweets)
    return features, vectorizer

def load_glove_embeddings(glove_path):
    """
    Load GloVe embeddings from a file.
    """
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    print(f"Loaded {len(embeddings_index)} word vectors.")
    return embeddings_index

def get_tweet_embeddings(tweets, embeddings_index, embedding_dim=100):
    """
    Generate tweet embeddings by averaging word embeddings for words in each tweet.
    """
    tweet_embeddings = []
    for tweet in tweets:
        words = tweet.split()
        word_vectors = [embeddings_index[word] for word in words if word in embeddings_index]
        if word_vectors:
            # Average word vectors to get tweet embedding
            tweet_vector = np.mean(word_vectors, axis=0)
        else:
            # If no words found in GloVe, use a zero vector
            tweet_vector = np.zeros(embedding_dim)
        tweet_embeddings.append(tweet_vector)
    return np.array(tweet_embeddings)




# Training 


def train_classifier(features, labels):
    """
    Train the classifier (Log Reg here)
    """
    classifier = LogisticRegression(random_state=42, max_iter=10000)
    classifier.fit(features, labels)
    return classifier

def train_fasttext(tweets, labels, output_model_path):
    """
    Train a FastText model for text classification.
    """
    # Create a temporary file for training data in FastText format
    temp_file = "fasttext_train.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        for tweet, label in zip(tweets, labels):
            # FastText requires the label to be in the format "__label__<label>"
            f.write(f"__label__{label} {tweet}\n")

    # Train the FastText model
    model = fasttext.train_supervised(
        input=temp_file,
        lr=0.1,
        epoch=20,
        wordNgrams=2,
        dim=100
    )
    
    # Save the model
    model.save_model(output_model_path)
    print(f"FastText model saved to {output_model_path}")
    return model



# Evaluation and saving predictions

def predict_and_save(test_tweets, vectorizer, classifier, output_path):
    """
    Predict on the test set and save predictions for submission.
    """
    test_features = vectorizer.transform(test_tweets)
    predictions = classifier.predict(test_features)

    # Convert predictions from {0, 1} to {-1, 1}
    predictions = np.where(predictions == 0, -1, 1)

    test_ids = range(1, len(test_tweets) + 1)  # Assuming tweet IDs start from 1
    submission = pd.DataFrame({'Id': test_ids, 'Prediction': predictions})
    submission.to_csv(output_path, index=False)

def predict_and_save_glove(test_tweets, embeddings_index, classifier, output_path, embedding_dim=100):
    """
    Predict on the test set using GloVe embeddings and save predictions for submission.
    """
    # Generate GloVe embeddings for test tweets
    test_features = get_tweet_embeddings(test_tweets, embeddings_index, embedding_dim=embedding_dim)
    
    # Make predictions
    predictions = classifier.predict(test_features)

    # Convert predictions from {0, 1} to {-1, 1}
    predictions = np.where(predictions == 0, -1, 1)

    # Prepare submission
    test_ids = range(1, len(test_tweets) + 1)  # Assuming tweet IDs start from 1
    submission = pd.DataFrame({'Id': test_ids, 'Prediction': predictions})
    submission.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")


def predict_fasttext(test_tweets, model_path, output_path):
    """
    Use a trained FastText model to predict on test data and save results.
    """
    # Load the trained model
    model = fasttext.load_model(model_path)

    # Predict labels for test tweets
    predictions = []
    for tweet in test_tweets:
        clean_tweet = tweet.strip()  # Remove any leading/trailing spaces or newline characters
        label, _ = model.predict(clean_tweet)
        # Extract label (e.g., "__label__1" -> 1)
        predictions.append(int(label[0].replace("__label__", "")))

    # Convert predictions from {0, 1} to {-1, 1}
    predictions = [-1 if p == 0 else 1 for p in predictions]

    # Save predictions
    test_ids = range(1, len(test_tweets) + 1)
    submission = pd.DataFrame({'Id': test_ids, 'Prediction': predictions})
    submission.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")
