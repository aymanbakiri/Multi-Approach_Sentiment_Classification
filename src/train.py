import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import optuna
import fasttext
from transformers import Trainer, TrainingArguments



# Logistic Regression for glove and tfidf

def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression with GridSearchCV.
    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
    Returns:
        model: Trained Logistic Regression model.
    """
    param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear']}
    grid = GridSearchCV(LogisticRegression(max_iter=10000), param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    print(f"Best Parameters: {grid.best_params_}")
    return grid.best_estimator_


# Fasttext

def train_fasttext_with_params(tweets, labels, output_model_path, best_params):
    """
    Train a FastText model using the best hyperparameters.
    """
    train_file = "fasttext_train_full.txt"
    with open(train_file, "w", encoding="utf-8") as f:
        for tweet, label in zip(tweets, labels):
            f.write(f"__label__{label} {tweet}\n")

    model = fasttext.train_supervised(
        input=train_file,
        lr=best_params["lr"],
        epoch=best_params["epoch"],
        wordNgrams=best_params["wordNgrams"],
        dim=best_params["dim"],
        loss=best_params["loss"]
    )

    model.save_model(output_model_path)
    print(f"FastText model saved to {output_model_path}")
    return model


def tune_fasttext_with_optuna(tweets, labels):
    """
    Use Optuna to tune FastText hyperparameters.
    """
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(tweets, labels, test_size=0.2, random_state=42)

    # Create a temporary file for training data in FastText format
    train_file = "fasttext_train_optuna.txt"
    with open(train_file, "w", encoding="utf-8") as f:
        for tweet, label in zip(X_train, y_train):
            f.write(f"__label__{label} {tweet}\n")

    def objective(trial):
        # Define the hyperparameter search space
        lr = trial.suggest_float("lr", 0.001, 0.5, log=True)  # Wider range for learning rate
        epoch = trial.suggest_int("epoch", 5, 100)  # Increase maximum epochs
        wordNgrams = trial.suggest_int("wordNgrams", 1, 4)  # Allow up to 4-grams
        dim = trial.suggest_categorical("dim", [50, 100, 200, 300])  # Add 200 dimensions
        loss = trial.suggest_categorical("loss", ["softmax", "hs", "ns"])  # Keep existing loss types


        # Train FastText model
        model = fasttext.train_supervised(
            input=train_file,
            lr=lr,
            epoch=epoch,
            wordNgrams=wordNgrams,
            dim=dim,
            loss=loss
        )

        # Validate on validation data
        val_predictions = [model.predict(tweet.strip())[0][0] for tweet in X_val]
        val_predictions = [int(pred.replace("__label__", "")) for pred in val_predictions]
        accuracy = accuracy_score(y_val, val_predictions)
        return 1 - accuracy

    # Run Optuna optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)  # Adjust n_trials for faster/slower tuning

    if study.best_params:
        print(f"Best Hyperparameters: {study.best_params}")
    else:
        print("No valid hyperparameters found.")
    
    return study.best_params


# Hyperparameter tuning
def tune_fasttext_hyperparams(tweets, labels):
    """
    Perform hyperparameter tuning for FastText using Optuna.
    """
    def objective(trial):
        # Define hyperparameter search space
        lr = trial.suggest_loguniform("lr", 0.01, 0.5)
        epoch = trial.suggest_int("epoch", 5, 50)
        wordNgrams = trial.suggest_int("wordNgrams", 1, 3)
        dim = trial.suggest_categorical("dim", [50, 100, 300])
        loss = trial.suggest_categorical("loss", ["softmax", "hs", "ns"])

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(tweets, labels, test_size=0.2, random_state=42)

        # Create a temporary file for training data in FastText format
        train_file = "fasttext_train.txt"
        with open(train_file, "w", encoding="utf-8") as f:
            for tweet, label in zip(X_train, y_train):
                f.write(f"__label__{label} {tweet}\n")

        # Train the FastText model
        model = fasttext.train_supervised(
            input=train_file,
            lr=lr,
            epoch=epoch,
            wordNgrams=wordNgrams,
            dim=dim,
            loss=loss
        )

        # Validate the model
        val_predictions = [
            int(model.predict(tweet.strip())[0][0].replace("__label__", ""))
            for tweet in X_val
        ]
        accuracy = accuracy_score(y_val, val_predictions)
        return accuracy

    # Run Optuna optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    return study.best_params, study.best_value

# Train FastText model with tuned parameters
def train_fasttext_with_params(tweets, labels, output_model_path, best_params):
    """
    Train a FastText model using the best hyperparameters.
    """
    # Create a temporary file for training data in FastText format
    temp_file = "fasttext_train.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        for tweet, label in zip(tweets, labels):
            f.write(f"__label__{label} {tweet}\n")

    # Train the FastText model
    model = fasttext.train_supervised(
        input=temp_file,
        lr=best_params["lr"],
        epoch=best_params["epoch"],
        wordNgrams=best_params["wordNgrams"],
        dim=best_params["dim"],
        loss=best_params["loss"]
    )

    # Save the model
    model.save_model(output_model_path)
    print(f"FastText model saved to {output_model_path}")
    return model



# DistilBERT

def distilbert_hyperparameter_tuning(train_dataset, val_dataset, model_init, output_dir="./results"):
    """
    Perform hyperparameter tuning for DistilBERT using Optuna.
    """
    def objective(trial):
        # Define the search space for hyperparameters
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)
        batch_size = trial.suggest_categorical("batch_size", [8, 16,32])
        weight_decay = trial.suggest_loguniform("weight_decay", 0.01, 0.1)
        num_train_epochs = trial.suggest_int("num_train_epochs", 3, 5)
        warmup_steps = trial.suggest_int("warmup_steps", 0, 200)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs, 
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            logging_dir="./logs",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=False,
            save_total_limit=1
        )

        # Initialize Trainer
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # Train and evaluate
        trainer.train()
        eval_results = trainer.evaluate()
        return eval_results["eval_loss"]  # Objective: minimize validation loss

    # Run Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)  # Perform 20 trials

    print("Best Hyperparameters:", study.best_params)
    return study.best_params

