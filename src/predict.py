# predict.py
# Script to predict sentiment for input texts using a trained model and tokenizer.
# Loads the model and tokenizer, preprocesses input texts, and outputs sentiment predictions.

from __future__ import annotations
import argparse
import os
import numpy as np

from .data_utils import ensure_nltk, load_tokenizer, texts_to_padded
from tensorflow.keras.models import load_model

SENTIMENT_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}  # Mapping from label to sentiment

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict sentiment for input texts")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model and tokenizer")
    parser.add_argument("--texts", nargs="+", required=True, help="One or more quoted texts")
    parser.add_argument("--max_len", type=int, default=100, help="Maximum sequence length for padding")
    args = parser.parse_args()

    ensure_nltk()  # Ensure NLTK resources are available

    # Load trained model and tokenizer
    model = load_model(os.path.join(args.model_dir, "model.h5"))
    tok = load_tokenizer(os.path.join(args.model_dir, "tokenizer.pkl"))

    # Preprocess and pad input texts
    X = texts_to_padded(tok, args.texts, max_len=args.max_len)

    # Predict sentiment
    preds = model.predict(X)
    labels = np.argmax(preds, axis=1)

    # Output predictions
    for t, y in zip(args.texts, labels):
        print(f"Text: {t}\nPredicted Sentiment: {SENTIMENT_MAP[int(y)]}\n")

if __name__ == "__main__":
    main()