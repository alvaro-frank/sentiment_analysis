from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout

def build_model(max_words=5000, embedding_dim=50, lstm_units=128):
    """Constrói o modelo Bi-LSTM conforme definido no notebook."""
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(lstm_units)))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model