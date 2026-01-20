# ==============================================================================
# FILE: src/train.py
# DESCRIPTION: Trains the LSTM model using the "Silver Standard" dataset
#              (Large dataset annotated by FinBERT).
# ==============================================================================

import os
import argparse
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from data_utils import preprocess_text
from model import build_model

def main():
    # 1. Configuration and Arguments
    parser = argparse.ArgumentParser(description="Train Sentiment Regression Model (Large Dataset)")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--max_words', type=int, default=10000, help="Vocabulary size")
    parser.add_argument('--resume', action='store_true', help="Resume training from existing checkpoint")
    args = parser.parse_args()

    # MLFlow
    mlflow.set_experiment("sentiment_analysis_experiment")
    mlflow.tensorflow.autolog()

    with mlflow.start_run() as run:
        print(f">>> MLflow Run ID: {run.info.run_id}")
        
        mlflow.log_params({
            "max_words": args.max_words,
            "data_source": "Large Financial Sentiment (Distilled from FinBERT)"
        })

        # 2. Load Data
        csv_path = "data/large_financial_sentiment.csv"
    
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} not found.")
            print("Please run 'python src/generate_dataset.py' first to generate the data.")
            return

        print(f">>> Loading generated dataset: {csv_path}...")
        df = pd.read_csv(csv_path)
        
        print(f">>> Dataset loaded. Total rows: {len(df)}")
        
        # 3. Preprocessing
        print(">>> Preprocessing texts...")
        df['processed_text'] = df['sentence'].apply(preprocess_text)
        
        df = df.dropna(subset=['processed_text', 'score'])

        X = df['processed_text'].values
        y = df['score'].values

        # 4. Tokenization
        print(">>> Tokenizing...")
        tokenizer = Tokenizer(num_words=args.max_words, lower=True)
        tokenizer.fit_on_texts(X)
        X_seq = tokenizer.texts_to_sequences(X)
        X_pad = pad_sequences(X_seq, maxlen=100, padding='post')

        # Save tokenizer
        os.makedirs("models", exist_ok=True)
        with open('models/tokenizer.pkl', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        mlflow.log_artifact('models/tokenizer.pkl', artifact_path="tokenizer")

        # 5. Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)
        
        model_path = 'models/sentiment_model.keras'
        model = None

        # 6. Build and Train Model
        if args.resume:
            if os.path.exists(model_path):
                print(f">>> RESUME: Loading existing model from {model_path}...")
                try:
                    model = load_model(model_path)
                    print(">>> RESUME: Model loaded successfully.")
                except Exception as e:
                    print(f">>> RESUME: Failed to load model ({e}). Building from scratch.")
            else:
                print(f">>> RESUME: No model found at {model_path}. Building from scratch.")
        if model is None:
            print(">>> Building new model architecture...")
            model = build_model(max_words=args.max_words)
            
        print(model.summary())

        # Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        # Checkpoint
        checkpoint = ModelCheckpoint(
            filepath='models/sentiment_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )

        print(">>> Starting REGRESSION training...")
        history = model.fit(X_train, y_train, 
                            batch_size=args.batch_size, 
                            epochs=args.epochs, 
                            validation_data=(X_test, y_test),
                            callbacks=[early_stopping, checkpoint])

        # 7. Save Final Model
        print(">>> Model saved to models/sentiment_model.keras")

        # Plot MAE
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['mae'], label='MAE (Train)')
        plt.plot(history.history['val_mae'], label='MAE (Validation)')
        plt.title('Model MAE on Large Dataset')
        plt.ylabel('MAE (Lower is better)')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig('models/training_history.png')
        mlflow.log_artifact('models/training_history.png')
        print(">>> Training history plot saved.")

if __name__ == "__main__":
    main()