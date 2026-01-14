# ==============================================================================
# FILE: src/evaluate.py
# DESCRIPTION: Evaluates the trained regression model.
#              Default: Evaluates on FiQA (Gold Standard - Human Labels).
#              Option: Evaluates on Local CSV (Silver Standard - FinBERT Labels).
# ==============================================================================

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from data_utils import load_tokenizer, texts_to_padded

def load_local_data(path):
    """Loads the local generated CSV."""
    print(f">>> Loading Local Dataset (Silver Standard) from {path}...")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df, 'sentence', 'score', 'Local FinBERT Dataset'

def main():
    parser = argparse.ArgumentParser(description="Evaluate Sentiment Model")
    parser.add_argument('--nrows', type=int, default=1000, help="Number of rows to evaluate")
    args = parser.parse_args()

    # 1. Load Data Strategy
    df = None
    text_col = 'sentence'
    label_col = 'score'
    dataset_name = "Unknown"
        
    local_path = "data/large_financial_sentiment.csv"
    df, text_col, label_col, dataset_name = load_local_data(local_path)

    if df is None:
        print("Error: Could not load any dataset. Please check your internet or run 'src/generate_dataset.py'.")
        return

    if args.nrows and args.nrows < len(df):
        print(f">>> Slicing first {args.nrows} rows...")
        df = df.iloc[:args.nrows]

    print(f">>> Evaluating on {len(df)} samples from: {dataset_name}")

    # 2. Prepare Data
    df = df.dropna(subset=[text_col, label_col])
    
    # 3. Load Model & Tokenizer
    print(">>> Loading trained model and tokenizer...")
    model_path = "models/sentiment_model.keras"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run 'python src/train.py' first.")
        return

    model = load_model(model_path)
    tokenizer = load_tokenizer("models/tokenizer.pkl")

    # 4. Model Prediction
    print(">>> Running Model Predictions...")
    X_eval = texts_to_padded(tokenizer, df[text_col].tolist())
    
    predictions = model.predict(X_eval)
    df['model_prediction'] = predictions.flatten()

    # 5. Calculate Metrics
    mae = mean_absolute_error(df[label_col], df['model_prediction'])
    mse = mean_squared_error(df[label_col], df['model_prediction'])
    r2 = r2_score(df[label_col], df['model_prediction'])

    print("\n" + "="*50)
    print(f" EVALUATION REPORT: {dataset_name}")
    print("="*50)
    print(f"Samples evaluated : {len(df)}")
    print(f"MAE (Mean Abs Err): {mae:.4f}")
    print(f"MSE (Mean Sq Err) : {mse:.4f}")
    print(f"R2 Score          : {r2:.4f}")
    print("="*50)

    # 6. Visualization
    print(">>> Generating evaluation plot...")
    plt.figure(figsize=(10, 6))
    
    plt.scatter(df[label_col], df['model_prediction'], alpha=0.3, s=10, label='Data Points')
    plt.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Ideal Fit (y=x)')
    
    plt.title(f'Predictions vs True Labels ({dataset_name})\nR2: {r2:.3f} | MAE: {mae:.3f}')
    plt.xlabel(f'Target Score ({dataset_name})')
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