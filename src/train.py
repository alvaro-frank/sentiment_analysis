import os
import argparse
import pickle
import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from data_utils import preprocess_text, get_sentiment_label
from model import build_model

def main():
    # 1. Configuração e Argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--max_words', type=int, default=5000) # Aumentado para 5000 conforme o notebook
    args = parser.parse_args()

    # 2. Obter Dados (Download automático do Kaggle se necessário)
    print("A obter dataset...")
    path = kagglehub.dataset_download("miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests")
    csv_path = os.path.join(path, "analyst_ratings_processed.csv")
    
    # Se o ficheiro principal não existir, tenta o raw
    if not os.path.exists(csv_path):
        csv_path = os.path.join(path, "raw_analyst_ratings.csv")

    print(f"A ler {csv_path}...")
    # Lemos apenas 50k linhas para teste rápido (remove 'nrows' para treinar com tudo)
    df = pd.read_csv(csv_path, nrows=1000) 
    
    # 3. Pré-processamento
    # Identificar coluna de texto
    text_col = 'title' if 'title' in df.columns else 'headline'
    df = df.dropna(subset=[text_col])
    
    print("A processar textos e gerar sentimentos (VADER)...")
    df['processed_text'] = df[text_col].apply(preprocess_text)
    df['sentiment'] = df[text_col].apply(get_sentiment_label)

    X = df['processed_text'].values
    y = to_categorical(df['sentiment'].values) # Converter para one-hot encoding

    # 4. Tokenização
    tokenizer = Tokenizer(num_words=args.max_words, lower=True)
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=100, padding='post')

    # Guardar tokenizer para usar no predict.py
    os.makedirs("models", exist_ok=True)
    with open('models/tokenizer.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 5. Split Treino/Teste
    X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

    # 6. Construir e Treinar Modelo
    model = build_model(max_words=args.max_words)
    print(model.summary())

    history = model.fit(X_train, y_train, 
                        batch_size=args.batch_size, 
                        epochs=args.epochs, 
                        validation_data=(X_test, y_test))

    # 7. Guardar Modelo e Gráfico
    model.save("models/sentiment_model.h5")
    print("Modelo guardado em models/sentiment_model.h5")
    
    print(history.history['accuracy'])
    print(history.history['val_accuracy'])

    # Gerar gráfico de accuracy (como no notebook)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('models/training_history.png')
    print("Gráfico de treino guardado em models/training_history.png")

if __name__ == "__main__":
    main()