# **ML Text Classification Project**

This project explores using multiple machine learning tools to predict the sentiment of a tweet (positive or ngative) as part of the second project for the ML EPFL course CS433. The task here is text classficiation. The files in this project implement a text classification model  through a complete data science pipeline, including data analysis, data preprocessing, model training, and evaluation. Finally we also have an ethical analysis where we address the potential risks in this project.


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
```

---

## **Main Python Files**

- **`EDA.ipynb`**: Notebook containing our code and findings during the Exploratory Data Analysis performed before starting work on the ML models. It focuses on the following key aspects:
  - **Data Overview**: Summary of dataset characteristics, such as size, columns, distribution of data, word frequencies, etc...
  - **Data Visualization**: Distribution plots and word clouds used to get some insight.
 
  **`run.py`**: Contains implementations of the text classification models. First the script reads the data and perform preprocessing. Then one of the 4 methods for text classification is used (is every case we split to an 80/20 validation training split):
  - **Logistic Regression with TF-IDF**: First we convert the text data into a matrix of TF-IDF features. We then use this matrix to perform logistic regression.
  - **Logistic Regression with GloVe Embeddings**: We use GloVe embeddings to create a numeric vector embedding for each tweet. We then use these embeddings to perform logistic regression.
  - **FastText Model**: We train the text classification FastText model with the data we have. Then we use that model to infer results for the validation set.
  - **Using Pre-trained DistilBERT**: We load the pre-trained DistilBERT tokenizer, which will convert the tweets into tokens. We then also load the pre-trained DistilBERT model. Then we tokenize the input text, and change the data format to fit the requirements of the DistilBERT model. Then we start training and evaluating the model on our dataset.

- **`utilities.py`**: The script provides multiple helper functions used by run.py. These helper functions perform many tasks:
  - **Data Loading**: Functions to load the data from the txt files.
  - **Data Preprocessing**: Functions to perform data cleaning and preprocessing.
  - **Converting tweets to numeric vectors/matrices**: These include the functions that transform the tweets to provide matrix of TF-IDF features, or functions providing GloVe embeddings.
  - **Supporting ML models**: Support for Logistic Regression model, SVC (Support vector classification), and FastText models.
  - **Evaluating models and saving csv submissions**: Functions that use the models with the test set to create predictions and save them as csv files to be ready for submission.
  
## **Usage**

**Setup**: Ensure that all the dependencies have been installed:
  * numpy
  * matplotlib
  * os
  * pickle
  * pandas
  * scikit-learn
  * transformers
  * torch
  * datasets
  * fasttext
  * re
  * nltk: Some functionalities require:
1. Running python:
```plaintext
python3
```
2. Downloading the necessary nltk datasets:
```plaintext
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

exit()
```

**Data**: Ensure all data files are located within a folder called "data" in the same directory where the main run.py script is located, with utilities.py being in the "src" directory which is also in the same directory as the main run.py script. These data files are:
  * train_pos_full.txt 
  * train_neg_full.txt
  * test_data.txt

**Execution**: Run the main script `run.py` to start the training. To choose between the 4 different methods, the script will output a line asking which method the user wants. Then the user choose one of the 4 methods as text input ("tfidf", "glove", "fasttext", "distilbert").

**Output**: Predictions for the test dataset are saved to a CSV file for submission. File will be named submission.csv and will be in the same directory where the main run.py is located.

## **Contributors**

- Bakiri Ayman
- Ben Mohamed Nizar
- Chahed Ouazzani Adam

    
    
    
    
    
    
    
    
    
    
    
    
    
