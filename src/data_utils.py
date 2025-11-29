import re
import string
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configurações globais
MAX_LEN = 100

def ensure_nltk():
    """Descarrega recursos necessários do NLTK (igual à célula 3 do notebook)."""
    resources = ['vader_lexicon', 'stopwords', 'punkt', 'wordnet', 'punkt_tab']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}') if res == 'punkt' else nltk.data.find(f'corpora/{res}')
        except LookupError:
            nltk.download(res)

ensure_nltk()

def preprocess_text(text, stemming=True, lemmatizing=True):
    """Limpa o texto aplicando lowercase, remoção de stopwords, stemming e lemmatization."""
    if not isinstance(text, str):
        return ""

    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token not in string.punctuation]
    
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens if re.sub(r'[^a-zA-Z]', '', token)]

    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

    if lemmatizing:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)

def get_sentiment_label(text):
    """Gera label usando VADER: 0=Negativo, 1=Neutro, 2=Positivo."""
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(str(text))['compound']
    if score >= 0.05:
        return 2
    elif score <= -0.05:
        return 0
    else:
        return 1

def load_tokenizer(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def texts_to_padded(tokenizer, texts, max_len=MAX_LEN):
    """Prepara novos textos para previsão."""
    processed = [preprocess_text(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(processed)
    return pad_sequences(seqs, maxlen=max_len, padding='post')