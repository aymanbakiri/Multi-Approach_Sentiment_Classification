import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
import nltk

# Loading and preparing 

def load_data(pos_path, neg_path):
    """
    Load positive and negative training tweets from files.

    Args:
        pos_path: Path to the file containing positive tweets, one tweet per line.
        neg_path: Path to the file containing negative tweets, one tweet per line.

    Returns:
        pos_tweets: A list of positive tweets.
        neg_tweets: A list of negative tweets.
    """
    # Open and read positive tweets
    with open(pos_path, 'r', encoding='utf-8') as f:
        pos_tweets = f.readlines()
    
    # Open and read negative tweets
    with open(neg_path, 'r', encoding='utf-8') as f:
        neg_tweets = f.readlines()
    
    return pos_tweets, neg_tweets


def load_test_data(test_path):
    """
    Load test tweets from a file.

    Args:
        test_path: Path to the file containing test tweets, one tweet per line.

    Returns:
        test_tweets: A list of test tweets.
    """
    # Open and read test tweets
    with open(test_path, 'r', encoding='utf-8') as f:
        test_tweets = f.readlines()
    
    return test_tweets


def preprocess_tweets(tweets):
    """
    Preprocess tweets by removing URLs and specific placeholders like 'user' and '<user>'.

    Args:
        tweets: A list of raw tweet strings.

    Returns:
        cleaned_tweets: A list of cleaned tweet strings.
    """
    cleaned_tweets = []
    for tweet in tweets:
        # Remove URLs (e.g., http://example.com, https://example.com, www.example.com)
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        # Remove user mentions (e.g., @username)
        tweet = re.sub(r"@[A-Za-z0-9]+", '', tweet)
        # Remove placeholder '<user>'
        tweet = re.sub(r"<user>", '', tweet)
        # Remove the word 'user'
        tweet = re.sub(r"user", '', tweet)
        # Append cleaned tweet to the list
        cleaned_tweets.append(tweet)
    
    return cleaned_tweets


def preprocess_data(pos_tweets, neg_tweets):
    """
    Combine positive and negative tweets into a single dataset, preprocess them, and assign labels.

    Args:
        pos_tweets: A list of positive tweet strings.
        neg_tweets: A list of negative tweet strings.

    Returns:
        tweets: A list of preprocessed tweets.
        labels: A list of corresponding labels (1 for positive, 0 for negative).
    """
    # Combine positive and negative tweets into a single list
    tweets = pos_tweets + neg_tweets
    # Create labels: 1 for positive tweets, 0 for negative tweets
    labels = [1] * len(pos_tweets) + [0] * len(neg_tweets)
    # Preprocess tweets (e.g., remove URLs, user mentions, and placeholders)
    tweets = preprocess_tweets(tweets)
    
    return tweets, labels


# Glove 

def load_glove_embeddings(glove_file_path):
    """
    Load GloVe embeddings from a file into a dictionary.

    Args:
        glove_file_path (str): Path to the GloVe file, where each line contains a word followed by its vector.

    Returns:
        dict: A dictionary mapping words (keys) to their embedding vectors (values as NumPy arrays).
    """
    embeddings = {}
    # Open the GloVe file in read mode
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Split each line into the word and its corresponding vector
            values = line.split()
            word = values[0]  # The first element is the word
            vector = np.asarray(values[1:], dtype='float32')  # The rest are the embedding vector components
            embeddings[word] = vector  # Add the word and its vector to the dictionary
    return embeddings


def tweet_to_glove_vector(tweet, embeddings, embedding_dim=50):
    """
    Convert a single tweet into an averaged GloVe vector.

    Args:
        tweet (str): The tweet text.
        embeddings (dict): Pre-loaded GloVe embeddings dictionary.
        embedding_dim (int): Dimension of the embeddings (e.g., 50, 100, 300).

    Returns:
        np.array: The averaged GloVe vector for the tweet. If no words in the tweet are found
                  in the embeddings, returns a zero vector of the specified embedding dimension.
    """
    # Split the tweet into individual words
    words = tweet.split()
    # Retrieve GloVe vectors for words present in the embeddings
    vectors = [embeddings[word] for word in words if word in embeddings]
    if len(vectors) == 0:
        # If no words are found in the embeddings, return a zero vector
        return np.zeros(embedding_dim)
    # Compute the mean of all word vectors in the tweet
    return np.mean(vectors, axis=0)


def tweets_to_glove_features(tweets, embeddings, embedding_dim=50):
    """
    Convert a list of tweets into a matrix of feature vectors using GloVe embeddings.

    Args:
        tweets (list of str): List of tweets to be converted into feature vectors.
        embeddings (dict): Pre-loaded GloVe embeddings dictionary.
        embedding_dim (int): Dimension of the embeddings (e.g., 50, 100, 300).

    Returns:
        np.array: A 2D NumPy array where each row corresponds to the averaged GloVe vector
                  of a tweet, and the number of columns matches the embedding dimension.
    """
    # Use list comprehension to convert each tweet into its GloVe vector
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

    # Download necessary resources
    nltk.download('punkt')  # Download the tokenizer
   
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


