import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.preprocess import preprocess_data, preprocess_tfidf, compute_tfidf, load_glove_embeddings, tweets_to_glove_features, load_data, load_test_data
from src.train import train_logistic_regression, tune_fasttext_with_optuna, distilbert_hyperparameter_tuning, train_fasttext_with_params
from src.evaluate import evaluate_and_predict_glove, evaluate_and_predict_tfidf, predict_fasttext, evaluate_fasttext
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import fasttext
from datasets import Dataset
import optuna
import torch



def main():
    
    # Check and set the appropriate device
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon Metal Performance Shaders (MPS)
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA CUDA GPU
    else:
        device = torch.device("cpu")  # Fallback to CPU
    print(f"Using device: {device}")

    # Define paths
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    pos_train_path = os.path.join(data_dir, "train_pos_full.txt")
    neg_train_path = os.path.join(data_dir, "train_neg_full.txt")
    pos_train_small_path = os.path.join(data_dir, "train_pos.txt")  # small dataset
    neg_train_small_path = os.path.join(data_dir, "train_neg.txt")  # small dataset
    test_path = os.path.join(data_dir, "test_data.txt")
    submission_path = os.path.join(os.path.dirname(__file__), "submission.csv")

    # Load data
    pos_tweets, neg_tweets = load_data(pos_train_path, neg_train_path)
    pos_tweets_small, neg_tweets_small = load_data(pos_train_small_path, neg_train_small_path)
    test_tweets = load_test_data(test_path)

    # Lightly preprocess the data for all the methods
    tweets, labels = preprocess_data(pos_tweets, neg_tweets)
    tweets_small, labels_small = preprocess_data(pos_tweets_small, neg_tweets_small)

    # Choose the method
    method = 'fasttext'  # Choose from 'tfidf', 'glove', 'fasttext', 'distilbert', 'roberta'

    if method == 'glove':

        # Define paths
        glove_path = "glove.twitter.27B.100d.txt"

        # Load GloVe embeddings
        print("Loading GloVe embeddings")
        glove_embeddings = load_glove_embeddings(glove_path)
        print("GloVe embeddings loaded.")

        # Split the data into training and validation sets
        print("Splitting data into train and validation sets")
        X_train, X_val, y_train, y_val = train_test_split(tweets, labels, test_size=0.2, random_state=42)

        # Convert tweets to GloVe feature vectors
        print("Converting tweets to GloVe feature vectors")
        X_train_glove = tweets_to_glove_features(X_train, glove_embeddings, embedding_dim=100)
        X_val_glove = tweets_to_glove_features(X_val, glove_embeddings, embedding_dim=100)

        # Train a Logistic Regression model
        print("Training Logistic Regression model")
        model = train_logistic_regression(X_train_glove, y_train)

        # Evaluate and save predictions
        evaluate_and_predict_glove(
            model=model,
            X_val=X_val_glove,
            y_val=y_val,
            test_tweets=test_tweets,
            glove_embeddings=glove_embeddings,
            embedding_dim=100,  # Match the GloVe embedding dimension
            submission_path="submission.csv"
        )

    elif method == 'tfidf':

        # Further Preprocessing of the data for TF-IDF
        tweets = preprocess_tfidf(tweets)
    
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(tweets, labels, test_size=0.2, random_state=42)

        # Compute TF-IDF features for training and validation
        print("Computing TF-IDF features")
        vectorizer, X_train_tfidf = compute_tfidf(X_train)  # Compute TF-IDF for training data
        X_val_tfidf = vectorizer.transform(X_val)  # Transform validation data

        # Train a Logistic Regression model
        print("Training Logistic Regression model")
        model = train_logistic_regression(X_train_tfidf, y_train)

        # Evaluate and predict
        print("Evaluating and predicting...")
        evaluate_and_predict_tfidf(
            model=model,
            X_val=X_val_tfidf,
            y_val=y_val,
            test_tweets=test_tweets,
            vectorizer=vectorizer,
            submission_path=submission_path
        )


    elif method == 'fasttext':
        # Perform hyperparameter tuning on the smaller dataset
        print("Tuning FastText hyperparameters on the smaller dataset...")
        best_params = tune_fasttext_with_optuna(tweets_small, labels_small)

        print(f"Best Hyperparameters from Optuna: {best_params}")

 
        # Train FastText model with the best hyperparameters on the full dataset
        print("Training FastText model on the full dataset...")
        fasttext_model_path = os.path.join(os.path.dirname(__file__), "fasttext_model.bin")
        train_file = "fasttext_train_full.txt"
        with open(train_file, "w", encoding="utf-8") as f:
            for tweet, label in zip(tweets, labels):
                f.write(f"__label__{label} {tweet}\n")

        fasttext_model = fasttext.train_supervised(
            input=train_file,
            lr=best_params["lr"],
            epoch=best_params["epoch"],
            wordNgrams=best_params["wordNgrams"],
            dim=best_params["dim"],
            loss=best_params["loss"]
        )

        # Evaluate on validation set
        print("Evaluating FastText model...")
        X_train, X_val, y_train, y_val = train_test_split(tweets, labels, test_size=0.2, random_state=42)
        val_predictions = [
            int(fasttext_model.predict(tweet.strip())[0][0].replace("__label__", ""))
            for tweet in X_val
        ]

        val_accuracy = accuracy_score(y_val, val_predictions)
        print(f"Validation Accuracy (FastText): {val_accuracy:.4f}")

        # Load test data
        print("Loading test data...")
        with open(test_path, 'r', encoding='utf-8') as f:
            test_tweets = f.readlines()

        # Predict and save results
        print("Predicting on test data...")
        predict_fasttext(test_tweets, fasttext_model_path, submission_path)

        print(f"Predictions saved to {submission_path}")




    elif method == 'distilbert':
        print("Using Pre-trained DistilBERT for Sentiment Classification with Hyperparameter Tuning")

        # Initialize DistilBERT tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        # Split data into training and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            tweets, labels, test_size=0.2, random_state=42
        )

        # Tokenize datasets
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

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

        # Define model initializer
        def model_init():
            return DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

        # Perform hyperparameter tuning
        best_hyperparams = distilbert_hyperparameter_tuning(train_dataset, val_dataset, model_init)

        # Use best hyperparameters to re-train the model
        print("Training DistilBERT with Best Hyperparameters...")
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=best_hyperparams['learning_rate'],
            per_device_train_batch_size=best_hyperparams['batch_size'],
            per_device_eval_batch_size=best_hyperparams['batch_size'],
            num_train_epochs=best_hyperparams['num_train_epochs'],
            weight_decay=best_hyperparams['weight_decay'],
            warmup_steps=best_hyperparams['warmup_steps'],
            logging_dir="./logs",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=False,
            save_total_limit=2
        )

        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # Train and evaluate
        trainer.train()
        eval_results = trainer.evaluate()
        print(f"Final Validation Loss: {eval_results['eval_loss']:.4f}")

        # Load and predict test data
        with open(test_path, 'r', encoding='utf-8') as f:
            test_tweets = f.readlines()

        test_encodings = tokenizer(test_tweets, truncation=True, padding=True, max_length=128)
        test_dataset = Dataset.from_dict({
            "input_ids": test_encodings["input_ids"],
            "attention_mask": test_encodings["attention_mask"]
        })

        predictions = trainer.predict(test_dataset)
        predicted_classes = np.argmax(predictions.predictions, axis=1)
        predicted_classes = [-1 if p == 0 else 1 for p in predicted_classes]

        test_ids = range(1, len(test_tweets) + 1)
        submission = pd.DataFrame({"Id": test_ids, "Prediction": predicted_classes})
        submission.to_csv(submission_path, index=False)
        print(f"Predictions saved to {submission_path}")


    elif method == 'roberta':
    
        print("Using Pre-trained RoBERTa-base for Sentiment Classification")

        # Initialize RoBERTa tokenizer and model
        from transformers import RobertaTokenizer, RobertaForSequenceClassification

        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)
        # Split data into training and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(tweets, labels, test_size=0.2, random_state=42)

        # Tokenize datasets
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

        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,  # Initial learning rate
            warmup_steps=200,  # Warmup steps
            lr_scheduler_type="cosine",  # Cosine decay for smoother transitions
            per_device_train_batch_size=8,  # Larger batch size
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch size
            num_train_epochs=7,  # Increase epochs
            weight_decay=0.01,  # Adjust weight decay
            logging_dir="./logs",
            save_total_limit=3,  # Keep the best 3 models
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,  # Lower loss is better
            fp16=True,  # Mixed precision
            gradient_checkpointing=True,  # Save memory
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
        print(f"Validation Loss: {eval_results['eval_loss']:.4f}")

        val_preds = np.argmax(trainer.predict(val_dataset).predictions, axis=1)
        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"Validation Accuracy: {val_accuracy:.4f}")



        # Tokenize and prepare test dataset
        test_encodings = tokenizer(test_tweets, truncation=True, padding=True, max_length=128)
        test_dataset = Dataset.from_dict({
            "input_ids": test_encodings["input_ids"],
            "attention_mask": test_encodings["attention_mask"]
        })

        # Predict on test dataset
        predictions = trainer.predict(test_dataset)
        predicted_classes = np.argmax(predictions.predictions, axis=1)

        # Convert predictions
        predicted_classes = [-1 if p == 0 else 1 for p in predicted_classes]
        test_ids = range(1, len(test_tweets) + 1)
        submission = pd.DataFrame({"Id": test_ids, "Prediction": predicted_classes})
        submission.to_csv(submission_path, index=False)

        print(f"Predictions saved to {submission_path}")

    else:
        print("Invalid method chosen. Please select from 'tfidf', 'glove', 'fasttext', or 'distilbert'.")

if __name__ == "__main__":
    main()
