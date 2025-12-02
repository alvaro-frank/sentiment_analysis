# ==============================================================================
# FILE: evaluate.py
# DESCRIPTION: Evaluates the trained regression model against the original
#              VADER sentiment scores. Calculates metrics (MAE, MSE, R2) and
#              generates a correlation plot.
# ==============================================================================

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

# Custom modules
from data_utils import preprocess_text, get_sentiment_score, load_tokenizer, texts_to_padded

def main():
    parser = argparse.ArgumentParser(description="Evaluate Model vs VADER")
    parser.add_argument('--nrows', type=int, default=1000, help="Number of rows to evaluate")
    args = parser.parse_args()

    # 1. Load Data
    print(">>> Loading dataset for evaluation...")
    try:
        path = kagglehub.dataset_download("miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests")
        csv_path = os.path.join(path, "analyst_ratings_processed.csv")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    print(f">>> Reading {args.nrows} rows from {csv_path}...")
    df = pd.read_csv(csv_path, nrows=args.nrows)
    
    text_col = 'title' if 'title' in df.columns else 'headline'
    df = df.dropna(subset=[text_col])

    # 2. Prepare "Ground Truth" (VADER Scores)
    print(">>> Calculating VADER scores (Ground Truth)...")
    # We calculate the score again to ensure we compare against the exact VADER logic
    df['vader_score'] = df[text_col].apply(get_sentiment_score)

    # 3. Load Model & Tokenizer
    print(">>> Loading trained model and tokenizer...")
    if not os.path.exists("models/sentiment_model.h5"):
        print("Error: Model not found. Run 'make train' first.")
        return

    model = load_model("models/sentiment_model.h5")
    tokenizer = load_tokenizer("models/tokenizer.pkl")

    # 4. Model Prediction
    print(">>> Running Model Predictions...")
    # Preprocess texts same way as training
    X_eval = texts_to_padded(tokenizer, df[text_col].tolist())
    
    # Predict (returns array of shape (N, 1))
    predictions = model.predict(X_eval)
    df['model_score'] = predictions.flatten()

    # 5. Calculate Metrics
    mae = mean_absolute_error(df['vader_score'], df['model_score'])
    mse = mean_squared_error(df['vader_score'], df['model_score'])
    r2 = r2_score(df['vader_score'], df['model_score'])

    print("\n" + "="*40)
    print(" EVALUATION REPORT")
    print("="*40)
    print(f"Samples evaluated : {len(df)}")
    print(f"MAE (Mean Abs Err): {mae:.4f}")
    print(f"MSE (Mean Sq Err) : {mse:.4f}")
    print(f"R2 Score          : {r2:.4f}")
    print("="*40)

    # 6. Visualization
    print(">>> Generating evaluation plot...")
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(df['vader_score'], df['model_score'], alpha=0.3, s=10, label='Data Points')
    
    # Perfect fit line (y=x)
    plt.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Ideal Fit (y=x)')
    
    plt.title(f'Model Predictions vs VADER Scores\nR2: {r2:.3f} | MAE: {mae:.3f}')
    plt.xlabel('VADER Score (Target)')
    plt.ylabel('Model Prediction')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    
    output_plot = 'models/evaluation_plot.png'
    plt.savefig(output_plot)
    print(f">>> Plot saved to {output_plot}")

if __name__ == "__main__":
    main()