import argparse
import os
import numpy as np
from tensorflow.keras.models import load_model
from data_utils import load_tokenizer, texts_to_padded

SENTIMENT_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, nargs="+", help="Texto para analisar")
    args = parser.parse_args()

    # Se não passar texto, usa os exemplos do notebook
    texts = args.text if args.text else [
        "Apple announces record-breaking earnings for Q1 2024",
        "Stock market rallies on positive economic news",
        "Tesla's new electric car receives mixed reviews",
        "Investors concerned about inflation impact on tech stocks"
    ]

    print("A carregar modelo...")
    model = load_model("models/sentiment_model.h5")
    tokenizer = load_tokenizer("models/tokenizer.pkl")

    # Preprocessar e Prever
    X_new = texts_to_padded(tokenizer, texts)
    preds = model.predict(X_new)
    labels = np.argmax(preds, axis=1)

    print("\n--- Resultados ---")
    for text, label in zip(texts, labels):
        print(f"Texto: {text}")
        print(f"Sentimento: {SENTIMENT_MAP[label]}\n")

if __name__ == "__main__":
    main()