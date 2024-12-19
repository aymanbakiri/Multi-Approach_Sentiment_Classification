import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import optuna
import fasttext
from transformers import Trainer, TrainingArguments



# METHOD 1: Logistic Regression Training with GridSearchCV
# This method trains a Logistic Regression model using GridSearchCV to identify the best hyperparameters.
# It saves the results of the GridSearch to a CSV file and returns the best model for evaluation or prediction.

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

# METHOD 2: FastText Hyperparameter Tuning with Optuna
# This method uses Optuna to perform hyperparameter optimization for FastText.
# It defines a search space for key parameters and evaluates model performance on validation data.
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


# METHOD 3: Hyperparameter Tuning for DistilBERT with Optuna
# This method tunes DistilBERT's hyperparameters using Optuna.
# It utilizes Hugging Face's Trainer API for training and evaluation.
def distilbert_hyperparameter_tuning(train_dataset, val_dataset, model_init, output_dir="./results"):
    """
    Perform hyperparameter tuning for DistilBERT using Optuna.
    Args:
        train_dataset: Hugging Face Dataset object for training data.
        val_dataset: Hugging Face Dataset object for validation data.
        model_init: Callable function to initialize a DistilBERT model.
        output_dir: Directory to save training results and logs.
    Returns:
        dict: The best hyperparameters found by Optuna.
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
