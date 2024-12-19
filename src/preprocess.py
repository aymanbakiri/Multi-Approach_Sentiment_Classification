import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
import nltk


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

def load_test_data(test_path):
    """
    Load test tweets.
    """
    with open(test_path, 'r', encoding='utf-8') as f:
        test_tweets = f.readlines()
    return test_tweets


def preprocess_tweets(tweets):
    """
    Remove URLs, and the word user and <user>" from tweets.
    """
    cleaned_tweets = []
    for tweet in tweets:
        # Remove URLs
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        # Remove user mentions
        tweet = re.sub(r"@[A-Za-z0-9]+", '', tweet)
        # Remove user mentions
        tweet = re.sub(r"<user>", '', tweet)
        # Remove user mentions
        tweet = re.sub(r"user", '', tweet)
        cleaned_tweets.append(tweet)
    return cleaned_tweets
    

def preprocess_data(pos_tweets, neg_tweets):
    """
    Preprocess training tweets and combine with labels.
    """
    tweets = pos_tweets + neg_tweets
    labels = [1] * len(pos_tweets) + [0] * len(neg_tweets)
    tweets = preprocess_tweets(tweets)
    return tweets, labels



# Glove 

def load_glove_embeddings(glove_file_path):
    """
    Load GloVe embeddings from file into a dictionary.
    Args:
        glove_file_path (str): Path to GloVe file.
    Returns:
        dict: A dictionary mapping words to their embedding vectors.
    """
    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def tweet_to_glove_vector(tweet, embeddings, embedding_dim=50):
    """
    Convert a tweet into an averaged GloVe vector.
    Args:
        tweet (str): The tweet text.
        embeddings (dict): Pre-loaded GloVe embeddings.
        embedding_dim (int): Dimension of embeddings.
    Returns:
        np.array: Averaged GloVe vector for the tweet.
    """
    words = tweet.split()
    vectors = [embeddings[word] for word in words if word in embeddings]
    if len(vectors) == 0:
        return np.zeros(embedding_dim)  # Return a zero vector if no words are in embeddings
    return np.mean(vectors, axis=0)

def tweets_to_glove_features(tweets, embeddings, embedding_dim=50):
    """
    Convert a list of tweets into feature vectors.
    Args:
        tweets (list of str): List of tweets.
        embeddings (dict): Pre-loaded GloVe embeddings.
        embedding_dim (int): Dimension of embeddings.
    Returns:
        np.array: Feature vectors for all tweets.
    """
    return np.array([tweet_to_glove_vector(tweet, embeddings, embedding_dim) for tweet in tweets])


# TF-IDF

def preprocess_tfidf(tweets):
    """
    Preprocess a list of tweets (one tweet per line):
    - Removes stopwords, punctuations, repeating characters, numbers
    - Applies stemming and lemmatization
    Args:
        tweets (list of str): List of tweets (one tweet per line).
    Returns:
        list of str: Preprocessed tweets.
    """
    # Initialize components
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    processed_tweets = []

    for tweet in tweets:
        # Lowercase the text
        tweet = tweet.lower()
        
        # Remove punctuation
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        
        # Remove repeating characters (e.g., heeellooo → hello)
        tweet = re.sub(r'(.)\1+', r'\1', tweet)
        
        # Remove numbers
        tweet = re.sub(r'\d+', '', tweet)
        
        # Tokenize text
        tokens = word_tokenize(tweet)
        
        # Remove stopwords and apply stemming/lemmatization
        cleaned_tokens = []
        for word in tokens:
            if word not in stop_words:  # Remove stopwords
                stemmed_word = stemmer.stem(word)  # Apply stemming
                lemmatized_word = lemmatizer.lemmatize(stemmed_word)  # Apply lemmatization
                cleaned_tokens.append(lemmatized_word)
        
        # Join tokens back into a single string
        processed_tweet = ' '.join(cleaned_tokens)
        processed_tweets.append(processed_tweet)

    return processed_tweets


def compute_tfidf(tweets, ngram_range=(1, 2), max_features=30000):
    """
    Compute TF-IDF features for the given tweets.
    Args:
        tweets (list of str): List of tweet texts.
        ngram_range (tuple): Range of n-grams to consider.
        max_features (int): Maximum number of features for the TF-IDF matrix.
    Returns:
        TfidfVectorizer, sparse matrix: Fitted TF-IDF vectorizer and feature matrix.
    """
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    tfidf_features = vectorizer.fit_transform(tweets)
    return vectorizer, tfidf_features


