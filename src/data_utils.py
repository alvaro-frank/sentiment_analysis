# ==============================================================================
# FILE: data_utils.py
# DESCRIPTION: Utility functions for text preprocessing (cleaning, tokenizing)
#              and sentiment scoring using VADER (lexicon-based).
# ==============================================================================

import re
import string
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Global Configurations
MAX_LEN = 100

def ensure_nltk():
    """Ensure necessary NLTK resources are downloaded."""
    resources = ['vader_lexicon', 'stopwords', 'punkt', 'wordnet', 'punkt_tab']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}') if 'punkt' in res else nltk.data.find(f'corpora/{res}')
        except LookupError:
            nltk.download(res)

# Run check on import
ensure_nltk()

def preprocess_text(text, stemming=True, lemmatizing=True):
    """
    Cleans text: lowercase, removes stopwords/punctuation, applies stemming/lemmatization.
    """
    if not isinstance(text, str):
        return ""

    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token not in string.punctuation]
    
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Remove numbers and special characters
    tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens if re.sub(r'[^a-zA-Z]', '', token)]

    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

    if lemmatizing:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)

def get_sentiment_score(text):
    """
    Returns the exact VADER compound score (float between -1 and 1).
    Used as the target variable for regression training.
    """
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(str(text))['compound']

def load_tokenizer(path):
    """Loads a saved pickle tokenizer."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def texts_to_padded(tokenizer, texts, max_len=MAX_LEN):
    """Preprocesses and pads a list of texts for model inference."""
    processed = [preprocess_text(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(processed)
    return pad_sequences(seqs, maxlen=max_len, padding='post')