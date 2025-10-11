# data_utils.py
# Utility functions for text preprocessing, sentiment scoring, and preparing data for sentiment analysis.
# Includes tokenization, stopword removal, stemming, lemmatization, sentiment scoring, and sequence padding.

import re
import pickle
from typing import Iterable, List, Tuple
import string

import numpy as np
import pandas as pd

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

def preprocess_text(text, stemming=True, lemmatizing=True):
    """
    Preprocess a text string: tokenize, lowercase, remove punctuation, stopwords, numbers,
    and apply stemming and lemmatization.
    """
    # Tokenization
    tokens = word_tokenize(text)

    # Convert to lowercase
    tokens = [token.lower() for token in tokens]

    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Remove numbers and special characters
    tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens]

    # Apply stemming
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

    # Apply lemmatization
    if lemmatizing:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into a string
    processed_text = ' '.join(tokens)

    return processed_text

def get_sentiment_score(text):
    """
    Get the compound sentiment score for a given text using VADER.
    """
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores['compound']

def get_sentiment_label(text):
    """
    Assign a sentiment label (0=negative, 1=neutral, 2=positive) based on compound score.
    """
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)

    # Decide sentiment label based on compound score
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        return 2
    elif compound_score <= -0.05:
        return 0
    else:
        return 1

max_len = 100  # Maximum sequence length for padding

def preprocess_text_row(row):
    """
    Preprocess a row from the DataFrame: clean text, vectorize, pad, and assign sentiment.
    """
    global count

    headline = row['headline']
    processed_headline = preprocess_text(headline)

    # Vectorization and Padding
    tokenizer = Tokenizer(num_words=5000)  # Limit to 5000 most common words
    tokenizer.fit_on_texts(processed_headline)
    sequences = tokenizer.texts_to_sequences([processed_headline])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

    # For demonstration, assuming these functions exist
    sentiment = get_sentiment_label(headline)
    sentiment_score = get_sentiment_score(headline)

    return pd.Series({
        'text': padded_sequences.flatten(),  # Flattened padded sequence
        'date': row['date'],
        'sentiment_score': sentiment_score,
        'sentiment': sentiment
    })

# Read CSV file with Pandas
df = pd.read_csv('Apple-Twitter-Sentiment-DFE.csv', encoding='latin1')

sample_df = df.sample(frac=0.02, random_state=42)  # Take a 2% sample for demonstration

# Apply preprocessing function to each row
df_processed = sample_df.apply(preprocess_text_row, axis=1)

# Save preprocessed data
df_processed.to_csv('preprocessed_data.csv', index=False)

df_processed.dropna(subset=['headline', 'date'], inplace=True)  # Drop rows with missing values

df_processed = df_processed.drop_duplicates(subset=['headline'], keep='first')  # Remove duplicate headlines