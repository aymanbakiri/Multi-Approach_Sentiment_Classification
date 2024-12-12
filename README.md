# **ML Text Classification Project**

This repository contains the implementation of a text classification system for analyzing sentiment in tweets for the second project of the ML EPFL course.

---

## **Overview**
The goal of this project is to classify tweets into categories (e.g., positive or negative sentiment) using machine learning techniques. Our approach includes:
- **Exploratory Data Analysis (EDA)** to understand the dataset.
- **Preprocessing** to clean and prepare the data.
- **Model Training** with baseline and advanced machine learning models.
- **Evaluation** using robust metrics to measure model performance.
- **Ethical Analysis** to address potential risks in the project.

---

## **Repository Structure**
```plaintext
ml-text-classification/
├── data/
│   ├── train.csv           # Processed training data
│   ├── test.csv            # Processed test data
│   └──                     # Raw data (optional for reproducibility)
├── src/
│   ├── preprocess.py       # Preprocessing script
│   ├── train.py            # Model training script
│   ├── evaluate.py         # Evaluation script
│   ├── utils.py            # Helper functions
│   └── models/             # Directory for custom models
├── notebooks/
│   ├── eda.ipynb           # Notebook for exploratory data analysis
│   └── experiments.ipynb   # Notebook for model experimentation
├── outputs/
│   ├── predictions.csv     # Final predictions for submission
│   ├── models/             # Saved models (optional)
│   └── logs/               # Training logs (optional)
├── requirements.txt        # List of Python dependencies
├── README.md               # Overview of the project and instructions
├── run.py                  # Entry point script to train/evaluate the model
└── report/
    ├── main.pdf            # Final report
    ├── latex/              # LaTeX source files (optional)
    └── figures/            # Figures used in the report
