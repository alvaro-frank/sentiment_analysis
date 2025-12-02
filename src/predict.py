# ==============================================================================
# FILE: predict.py
# DESCRIPTION: Inference script. Loads the trained regression model and 
#              predicts sentiment scores for new text inputs.
# ==============================================================================

import argparse
import os
import numpy as np
from tensorflow.keras.models import load_model
from data_utils import load_tokenizer, texts_to_padded

def get_label_from_score(score):
    """Converts a numeric score back to a readable label."""
    if score >= 0.5:
        return "Positive"
    elif score <= -0.5:
        return "Negative"
    else:
        return "Neutral"

def main():
    parser = argparse.ArgumentParser(description="Predict sentiment scores for input text")
    parser.add_argument("--text", type=str, nargs="+", help="List of texts to analyze")
    args = parser.parse_args()

    # Default examples if no text provided
    texts = args.text if args.text else [
        "Apple announces record-breaking earnings for Q1 2024",
        "Stock market rallies on positive economic news",
        "Tesla's new electric car receives mixed reviews",
        "Investors concerned about inflation impact on tech stocks"
    ]

    print(">>> Loading model and tokenizer...")
    if not os.path.exists("models/sentiment_model.h5"):
        print("Error: Model not found. Please run 'train.py' first.")
        return

    model = load_model("models/sentiment_model.h5")
    tokenizer = load_tokenizer("models/tokenizer.pkl")

    # Preprocessing
    X_new = texts_to_padded(tokenizer, texts)
    
    # INFERENCE: Model predicts a raw score directly
    predicted_scores = model.predict(X_new)

    # Output Results
    print("\n" + "="*65)
    print(f"{'TEXT':<45} | {'SCORE (Model)':<15} | {'LABEL'}")
    print("="*65)

    for text, score_array in zip(texts, predicted_scores):
        score = float(score_array[0])
        label = get_label_from_score(score)
        
        # Truncate long text for display
        display_text = (text[:42] + '..') if len(text) > 42 else text
        
        print(f"{display_text:<45} | {score: .4f}          | {label}")

    print("="*65 + "\n")

if __name__ == "__main__":
    main()