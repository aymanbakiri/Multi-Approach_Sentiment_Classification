import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import optuna
import fasttext
from transformers import Trainer, TrainingArguments



# Logistic Regression for GLOVE and TF-IDF

def train_logistic_regression(X_train, y_train, results_file="gridsearch_results.csv"):
    """
    Train Logistic Regression with GridSearchCV and save results.
    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        results_file: Path to save GridSearchCV results.
    Returns:
        model: Trained Logistic Regression model with the best hyperparameters.
    """
    # Define the hyperparameter grid
    param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear']}
    
    # Initialize Logistic Regression and GridSearchCV
    model = LogisticRegression(max_iter=10000)
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,  # 5-fold cross-validation
        verbose=1,
        return_train_score=True,  # Capture train scores
    )
    
    # Train the model
    grid.fit(X_train, y_train)
    
    # Save GridSearch results to a CSV file
    results = pd.DataFrame(grid.cv_results_)  # Extract all results
    results.to_csv(results_file, index=False)
    print(f"GridSearch results saved to {results_file}")
    
    # Print the best parameters
    print(f"Best Parameters: {grid.best_params_}")
    
    return grid.best_estimator_  # Return the best model


# FastText

def train_fasttext_with_params(tweets, labels, output_model_path, best_params):
    """
    Train a FastText model using the best hyperparameters.

    Args:
        tweets: List of tweets to be used as training data.
        labels: List of labels corresponding to the tweets.
        output_model_path: File path where the trained FastText model will be saved.
        best_params: Dictionary containing the best hyperparameters for training FastText.

    Returns:
        model: Trained FastText model.
    """
    # File to store the training data in the format required by FastText
    train_file = "fasttext_train_full.txt"
    
    # Write the training data to the file in FastText's required format: "__label__<label> <text>"
    with open(train_file, "w", encoding="utf-8") as f:
        for tweet, label in zip(tweets, labels):
            # Each line contains the label prefixed by "__label__" followed by the tweet text
            f.write(f"__label__{label} {tweet}\n")

    # Train the FastText model using the provided hyperparameters
    model = fasttext.train_supervised(
        input=train_file,               # Path to the training data file
        lr=best_params["lr"],          # Learning rate
        epoch=best_params["epoch"],    # Number of training epochs
        wordNgrams=best_params["wordNgrams"],  # Maximum number of word n-grams
        dim=best_params["dim"],        # Embedding dimension
        loss=best_params["loss"]       # Loss function (e.g., "softmax", "ova")
    )

    # Save the trained model to the specified file path
    model.save_model(output_model_path)
    print(f"FastText model saved to {output_model_path}")

    # Return the trained model for further use or evaluation
    return model



def tune_fasttext_with_optuna(tweets, labels):
    """
    Use Optuna to tune FastText hyperparameters.

    Args:
        tweets: List of tweet text data.
        labels: List of corresponding labels for the tweets.

    Returns:
        best_params: The best hyperparameters found by Optuna.
    """
    # Split the data into training and validation sets (80% training, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(tweets, labels, test_size=0.2, random_state=42)

    # Create a temporary file to store training data in FastText's required format
    train_file = "fasttext_train_optuna.txt"
    with open(train_file, "w", encoding="utf-8") as f:
        for tweet, label in zip(X_train, y_train):
            # Format for FastText: "__label__<label> <text>"
            f.write(f"__label__{label} {tweet}\n")

    def objective(trial):
        """
        Define the objective function for Optuna to optimize.

        Args:
            trial: An Optuna trial object for suggesting hyperparameters.

        Returns:
            A metric to minimize (1 - validation accuracy).
        """
        # Define the hyperparameter search space
        lr = trial.suggest_float("lr", 0.001, 0.5, log=True)  # Learning rate (log scale)
        epoch = trial.suggest_int("epoch", 5, 100)  # Number of epochs
        wordNgrams = trial.suggest_int("wordNgrams", 1, 4)  # Word n-grams up to 4
        dim = trial.suggest_categorical("dim", [50, 100, 200, 300])  # Embedding dimensions
        loss = trial.suggest_categorical("loss", ["softmax", "hs", "ns"])  # Loss functions

        # Train the FastText model with the suggested hyperparameters
        model = fasttext.train_supervised(
            input=train_file,
            lr=lr,
            epoch=epoch,
            wordNgrams=wordNgrams,
            dim=dim,
            loss=loss
        )

        # Validate the model on the validation data
        val_predictions = [model.predict(tweet.strip())[0][0] for tweet in X_val]  # Get predictions
        val_predictions = [int(pred.replace("__label__", "")) for pred in val_predictions]  # Extract numeric labels
        accuracy = accuracy_score(y_val, val_predictions)  # Compute validation accuracy
        
        # Optuna minimizes the objective, so return 1 - accuracy
        return 1 - accuracy

    # Create an Optuna study to minimize the objective function
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # Run the optimization for 20 trials

    # Output the best hyperparameters found
    if study.best_params:
        print(f"Best Hyperparameters: {study.best_params}")
    else:
        print("No valid hyperparameters found.")
    
    # Return the best hyperparameters
    return study.best_params

def tune_fasttext_hyperparams(tweets, labels):
    """
    Perform hyperparameter tuning for FastText using Optuna.

    Args:
        tweets: List of tweet text data.
        labels: List of corresponding labels for the tweets.

    Returns:
        best_params: The best hyperparameters found by Optuna.
        best_value: The highest validation accuracy achieved during tuning.
    """
    def objective(trial):
        """
        Define the objective function to optimize FastText hyperparameters.

        Args:
            trial: An Optuna trial object for suggesting hyperparameters.

        Returns:
            accuracy: Validation accuracy of the FastText model trained with the trial's hyperparameters.
        """
        # Define the search space for hyperparameters
        lr = trial.suggest_loguniform("lr", 0.01, 0.5)  # Learning rate (logarithmic scale)
        epoch = trial.suggest_int("epoch", 5, 50)      # Number of epochs
        wordNgrams = trial.suggest_int("wordNgrams", 1, 3)  # Word n-grams (1 to 3)
        dim = trial.suggest_categorical("dim", [50, 100, 300])  # Embedding dimensions
        loss = trial.suggest_categorical("loss", ["softmax", "hs", "ns"])  # Loss function type

        # Split the dataset into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(tweets, labels, test_size=0.2, random_state=42)

        # Prepare the training file in the required format for FastText
        train_file = "fasttext_train.txt"
        with open(train_file, "w", encoding="utf-8") as f:
            for tweet, label in zip(X_train, y_train):
                # Format: "__label__<label> <text>"
                f.write(f"__label__{label} {tweet}\n")

        # Train the FastText model with the suggested hyperparameters
        model = fasttext.train_supervised(
            input=train_file,
            lr=lr,                      # Learning rate
            epoch=epoch,                # Number of epochs
            wordNgrams=wordNgrams,      # Maximum word n-grams
            dim=dim,                    # Embedding dimension
            loss=loss                   # Loss function
        )

        # Predict labels for the validation set
        val_predictions = [
            int(model.predict(tweet.strip())[0][0].replace("__label__", ""))  # Remove "__label__" prefix
            for tweet in X_val
        ]

        # Calculate validation accuracy
        accuracy = accuracy_score(y_val, val_predictions)
        return accuracy  # Optuna will maximize this value

    # Create an Optuna study to maximize validation accuracy
    study = optuna.create_study(direction="maximize")
    
    # Run the optimization process for 30 trials
    study.optimize(objective, n_trials=30)

    # Return the best hyperparameters and the highest accuracy achieved
    return study.best_params, study.best_value


def train_fasttext_with_params(tweets, labels, output_model_path, best_params):
    """
    Train a FastText model using the best hyperparameters.

    Args:
        tweets: List of tweet text data.
        labels: List of corresponding labels for the tweets.
        output_model_path: Path where the trained FastText model will be saved.
        best_params: Dictionary of the best hyperparameters obtained from tuning.

    Returns:
        model: The trained FastText model.
    """
    # Create a temporary file to store the training data in FastText's required format
    temp_file = "fasttext_train.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        for tweet, label in zip(tweets, labels):
            # Format each line as "__label__<label> <text>" for FastText training
            f.write(f"__label__{label} {tweet}\n")

    # Train the FastText model using the provided hyperparameters
    model = fasttext.train_supervised(
        input=temp_file,                # Path to the training data file
        lr=best_params["lr"],          # Learning rate
        epoch=best_params["epoch"],    # Number of training epochs
        wordNgrams=best_params["wordNgrams"],  # Maximum word n-grams to consider
        dim=best_params["dim"],        # Embedding dimension
        loss=best_params["loss"]       # Loss function 
    )

    # Save the trained model to the specified path
    model.save_model(output_model_path)
    print(f"FastText model saved to {output_model_path}")

    # Return the trained model for further use or evaluation
    return model



# DistilBERT

def distilbert_hyperparameter_tuning(train_dataset, val_dataset, model_init, output_dir="./results"):
    """
    Perform hyperparameter tuning for DistilBERT using Optuna.

    Args:
        train_dataset: The training dataset formatted for the Hugging Face Trainer.
        val_dataset: The validation dataset formatted for the Hugging Face Trainer.
        model_init: A callable to initialize the model. Typically a function that returns a DistilBERT model.
        output_dir: Directory to save the training results and logs.

    Returns:
        best_params: Dictionary of the best hyperparameters found by Optuna.
    """
    def objective(trial):
        """
        Objective function for Optuna to optimize hyperparameters.

        Args:
            trial: An Optuna trial object that suggests hyperparameter values.

        Returns:
            eval_loss: The validation loss achieved with the trial's hyperparameters.
        """
        # Define the search space for hyperparameters
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)  # Learning rate
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])    # Batch size options
        weight_decay = trial.suggest_loguniform("weight_decay", 0.01, 0.1)  # Weight decay for regularization
        num_train_epochs = trial.suggest_int("num_train_epochs", 3, 5)      # Number of training epochs
        warmup_steps = trial.suggest_int("warmup_steps", 0, 200)            # Warmup steps for learning rate scheduler

        # Define the training arguments for Hugging Face Trainer
        training_args = TrainingArguments(
            output_dir=output_dir,                       # Output directory for model and results
            evaluation_strategy="epoch",                # Evaluate at the end of each epoch
            save_strategy="epoch",                      # Save model at the end of each epoch
            learning_rate=learning_rate,                # Learning rate from trial
            per_device_train_batch_size=batch_size,     # Batch size for training
            per_device_eval_batch_size=batch_size,      # Batch size for evaluation
            num_train_epochs=num_train_epochs,          # Number of epochs from trial
            weight_decay=weight_decay,                  # Weight decay for regularization
            warmup_steps=warmup_steps,                  # Warmup steps from trial
            logging_dir="./logs",                       # Directory for logging
            load_best_model_at_end=True,                # Load the best model based on validation metric
            metric_for_best_model="eval_loss",          # Validation loss as the evaluation metric
            fp16=False,                                 # Disable mixed precision
            save_total_limit=1                          # Limit the number of saved models
        )

        # Initialize the Hugging Face Trainer
        trainer = Trainer(
            model_init=model_init,          # Model initialization function
            args=training_args,             # Training arguments
            train_dataset=train_dataset,    # Training dataset
            eval_dataset=val_dataset        # Validation dataset
        )

        # Train the model and evaluate on the validation set
        trainer.train()                      # Train the model
        eval_results = trainer.evaluate()    # Evaluate on validation set

        # Return validation loss to minimize as the objective
        return eval_results["eval_loss"]

    # Create an Optuna study to minimize the validation loss
    study = optuna.create_study(direction="minimize")

    # Run the optimization for a specified number of trials
    study.optimize(objective, n_trials=5)  # Perform 5 trials (adjust as needed)

    # Print the best hyperparameters found and return them
    print("Best Hyperparameters:", study.best_params)
    return study.best_params
