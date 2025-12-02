# ==============================================================================
# FILE: train.py
# DESCRIPTION: Main training script. Downloads data, preprocesses text,
#              generates VADER scores as targets, and trains the regression model.
# ==============================================================================

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

# Custom modules
from data_utils import preprocess_text, get_sentiment_score
from model import build_model

def main():
    # 1. Configuration and Arguments
    parser = argparse.ArgumentParser(description="Train Sentiment Regression Model")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=50, help="Batch size")
    parser.add_argument('--max_words', type=int, default=5000, help="Vocabulary size")
    parser.add_argument('--nrows', type=int, default=50000, help="Number of rows to read from CSV")
    args = parser.parse_args()

    # 2. Data Acquisition (Kaggle)
    print(">>> Downloading/Loading dataset from Kaggle...")
    path = kagglehub.dataset_download("miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests")
    csv_path = os.path.join(path, "analyst_ratings_processed.csv")

    print(f">>> Reading file: {csv_path} (Limit: {args.nrows} rows)...")
    df = pd.read_csv(csv_path, nrows=1000) 
    
    # Identifies the text column
    text_col = 'title' if 'title' in df.columns else 'headline'
    df = df.dropna(subset=[text_col])
    
    # 3. Preprocessing & Target Generation
    print(">>> Processing texts and calculating target SCORES...")
    df['processed_text'] = df[text_col].apply(preprocess_text)
    
    # TARGET: Numeric score (-1 to 1) instead of categorical label
    df['score'] = df[text_col].apply(get_sentiment_score)

    X = df['processed_text'].values
    y = df['score'].values

    # 4. Tokenization
    tokenizer = Tokenizer(num_words=args.max_words, lower=True)
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=100, padding='post')

    # Save tokenizer for inference
    os.makedirs("models", exist_ok=True)
    with open('models/tokenizer.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

    # 6. Build and Train Model
    model = build_model(max_words=args.max_words)
    print(model.summary())

    print(">>> Starting REGRESSION training...")
    history = model.fit(X_train, y_train, 
                        batch_size=args.batch_size, 
                        epochs=args.epochs, 
                        validation_data=(X_test, y_test))

    # 7. Save Artifacts
    model.save("models/sentiment_model.h5")
    print(">>> Model saved to models/sentiment_model.h5")

    # Plot MAE (Mean Absolute Error)
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mae'], label='MAE (Train)')
    plt.plot(history.history['val_mae'], label='MAE (Validation)')
    plt.title('Model Mean Absolute Error (MAE)')
    plt.ylabel('MAE (Lower is better)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/training_history.png')
    print(">>> Training history plot saved.")

if __name__ == "__main__":
    main()