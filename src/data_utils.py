# ==============================================================================
# FILE: src/data_utils.py
# DESCRIPTION: Utility functions for text preprocessing.
# ==============================================================================

import re
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Global Configurations
MAX_LEN = 100

def ensure_nltk():
    """Ensure necessary NLTK resources are downloaded."""
    resources = ['stopwords', 'punkt', 'wordnet', 'punkt_tab']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}') if 'punkt' in res else nltk.data.find(f'corpora/{res}')
        except LookupError:
            print(f">>> Downloading NLTK resource: {res}...")
            nltk.download(res, quiet=True)

# Run check on import
ensure_nltk()

def preprocess_text(text, stemming=False, lemmatizing=True):
    """
    Cleans text for Financial Sentiment Analysis:
    1. Lowercase & Contraction expansion.
    2. Number normalization (10% -> <NUM>).
    3. Smart Stopwords (Preserves negations like 'not', 'no').
    4. Lemmatization.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Basic Contraction Expansion (Crucial for "not")
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    # 3. Number Handling
    # Replaces numbers/percentages with <NUM> tag.
    # Keras Tokenizer usually strips '<' and '>', treating it as the token "num".
    text = re.sub(r'\d+(\.\d+)?%?', ' <NUM> ', text)

    # 4. Tokenization
    tokens = nltk.word_tokenize(text)

    # 5. Smart Stopwords Filtering
    stop_words = set(stopwords.words('english'))
    # IMPORTANT: Remove negations from the ban list
    negations = {'no', 'not', 'nor', 'neither', 'never', 'none'}
    stop_words = stop_words - negations
    
    tokens = [token for token in tokens if token not in stop_words]

    # 6. Remove Punctuation (Keep only alphanumeric and the tag content)
    # Note: Keras Tokenizer also filters punctuation, but we clean here too.
    tokens = [token for token in tokens if token not in string.punctuation]

    # 7. Lemmatization (Default=True) - Better than Stemming for Finance
    if lemmatizing:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)

def load_tokenizer(path):
    """Loads a saved pickle tokenizer."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def texts_to_padded(tokenizer, texts, max_len=MAX_LEN):
    """Preprocesses and pads a list of texts for model inference."""
    # Applies the updated preprocess_text function
    processed = [preprocess_text(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(processed)
    return pad_sequences(seqs, maxlen=max_len, padding='post')