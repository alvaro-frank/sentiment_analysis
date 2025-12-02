# ==============================================================================
# FILE: model.py
# DESCRIPTION: Defines the architecture of the Sentiment Analysis model.
#              Uses a Bidirectional LSTM for regression (predicting a score).
# ==============================================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout

def build_model(max_words=5000, embedding_dim=50, lstm_units=128):
    """
    Builds a REGRESSION model to predict sentiment scores (-1 to 1).
    
    Args:
        max_words (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embedding layer.
        lstm_units (int): Number of units in the LSTM layer.
        
    Returns:
        model: Compiled Keras model.
    """
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(lstm_units)))
    
    # OUTPUT LAYER: Single neuron for continuous output
    # ACTIVATION: 'tanh' ensures the output is strictly between -1 and 1
    model.add(Dense(1, activation='tanh'))
    
    # COMPILATION:
    # Loss: 'mean_squared_error' (Standard for regression tasks)
    # Metric: 'mae' (Mean Absolute Error)
    model.compile(optimizer='adam', 
                  loss='mean_squared_error', 
                  metrics=['mae'])
    return model