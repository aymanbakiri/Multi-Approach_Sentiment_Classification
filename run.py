import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.utilities import *
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from datasets import Dataset

def main():
    # Detect device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # Choose the method
    method = input("Choose a method (tfidf, glove, fasttext, distilbert): ").strip().lower()

    if method == 'tfidf':
        print("Using Logistic Regression with TF-IDF")
        # Build features with TF-IDF
        features, vectorizer = build_features_tfidf(tweets)

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Train the classifier
        classifier = train_classifier(X_train, y_train)

        # Evaluate on the validation set
        val_predictions = classifier.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        print(f"Validation Accuracy (TF-IDF): {val_accuracy:.4f}")

        # Load test data
        with open(test_path, 'r', encoding='utf-8') as f:
            test_tweets = f.readlines()

        # Predict and save
        predict_and_save(test_tweets, vectorizer, classifier, submission_path)

        print(f"Predictions saved to {submission_path}")

    elif method == 'glove':
        print("Using Logistic Regression with GloVe Embeddings")
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
        print(f"Validation Accuracy (GloVe): {val_accuracy:.4f}")

        # Load test data
        with open(test_path, 'r', encoding='utf-8') as f:
            test_tweets = f.readlines()

        # Predict and save
        predict_and_save_glove(test_tweets, embeddings_index, classifier, submission_path)

        print(f"Predictions saved to {submission_path}")

    elif method == 'fasttext':  # FastText method
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

        # Predict and save results
        predict_fasttext(test_tweets, fasttext_model_path, submission_path)

        print(f"Predictions saved to {submission_path}")    

    elif method == 'distilbert':
        print("Using Pre-trained DistilBERT for Sentiment Classification")

        # Initialize DistilBERT tokenizer and model
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

        # Split data into training and validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(tweets, labels, test_size=0.2, random_state=42)        

        # Tokenize datasets with increased max_length
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

        # Prepare datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': train_labels
        })

        val_dataset = Dataset.from_dict({
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask'],
            'labels': val_labels
        })

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            num_train_epochs=3,  # Increase epochs
            weight_decay=0.01,
            logging_dir="./logs",
            lr_scheduler_type="linear",  # Use learning rate scheduler
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # Train and evaluate
        trainer.train()
        eval_results = trainer.evaluate()
        print(f"Validation Accuracy (DistilBERT): {eval_results['eval_loss']:.4f}")

        # Load test data
        with open(test_path, 'r', encoding='utf-8') as f:
            test_tweets = f.readlines()

        # Preprocess and predict on test data
        test_encodings = tokenizer(test_tweets, truncation=True, padding=True, max_length=128)
        test_inputs = torch.tensor(test_encodings['input_ids']).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(test_inputs)
            predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        # Convert predictions
        predictions = [-1 if p == 0 else 1 for p in predictions]
        test_ids = range(1, len(test_tweets) + 1)
        submission = pd.DataFrame({'Id': test_ids, 'Prediction': predictions})
        submission.to_csv(submission_path, index=False)

        print(f"Predictions saved to {submission_path}")

    else:
        print("Invalid method chosen. Please select from 'tfidf', 'glove', 'fasttext', or 'distilbert'.")

if __name__ == "__main__":
    main()
