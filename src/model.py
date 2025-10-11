# model.py
# Defines the architecture for the sentiment analysis model using Keras.
# The model uses an embedding layer, dropout, bidirectional LSTM, and a dense output layer.

from __future__ import annotations
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout

def build_model(*, max_words: int, embedding_dim: int, lstm_units: int) -> Sequential:
    """
    Build and compile a sentiment analysis model.

    Args:
        max_words (int): Maximum number of words in the vocabulary.
        embedding_dim (int): Dimension of the embedding vectors.
        lstm_units (int): Number of units in the LSTM layer.

    Returns:
        Sequential: Compiled Keras model.
    """
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim))  # Embedding layer for input text
    model.add(Dropout(0.2))  # Dropout for regularization
    model.add(Bidirectional(LSTM(lstm_units)))  # Bidirectional LSTM for sequence modeling
    model.add(Dense(3, activation="softmax"))  # Output layer for 3 sentiment classes
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])  # Compile model
    return model