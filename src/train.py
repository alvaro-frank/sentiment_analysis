# train.py
# Script to train a sentiment analysis model using configuration from a file or command-line arguments.
# Loads and preprocesses data, builds the model, and starts training.

import os
import argparse

from utils import load_cfg, load_and_prepare, build_model  # Utility functions for config, data, and model

def parse_args():
    """
    Parse command-line arguments for training configuration.
    """
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--input_csv', type=str, help='Input CSV file')
    parser.add_argument('--model_dir', type=str, help='Directory to save model')
    parser.add_argument('--max_words', type=int, help='Maximum number of words')
    parser.add_argument('--max_len', type=int, help='Maximum sequence length')
    parser.add_argument('--embedding_dim', type=int, help='Embedding dimension')
    parser.add_argument('--lstm_units', type=int, help='Number of LSTM units')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--val_size', type=float, help='Validation set size')
    parser.add_argument('--random_state', type=int, help='Random state')
    return parser.parse_args()

def main():
    # Parse arguments and load configuration
    args = parse_args()
    cfg = load_cfg(args.config)

    # Helper to get parameter from CLI or config file
    def get(name, default=None):
        return getattr(args, name) if getattr(args, name) is not None else cfg.get(name, default)

    # Set up paths and hyperparameters
    input_csv = get("input_csv", "data/raw/raw_analyst_ratings.csv")
    model_dir = get("model_dir", "models")
    os.makedirs(model_dir, exist_ok=True)

    max_words = int(get("max_words", 1000))
    max_len = int(get("max_len", 100))
    embedding_dim = int(get("embedding_dim", 50))
    lstm_units = int(get("lstm_units", 128))
    batch_size = int(get("batch_size", 50))
    epochs = int(get("epochs", 10))
    val_size = float(get("val_size", 0.2))
    random_state = int(get("random_state", 42))

    # Load and preprocess data
    X_tr, X_val, y_tr, y_val, tokenizer = load_and_prepare(
        input_csv,
        processed_dir="data/processed",
        max_words=max_words,
        max_len=max_len,
        val_size=val_size,
        random_state=random_state,
    )

    # Build and train the model
    model = build_model(max_words=max_words, embedding_dim=embedding_dim, lstm_units=lstm_units)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

if __name__ == "__main__":
    main()