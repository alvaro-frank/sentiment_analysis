import os
import argparse
import pandas as pd
import numpy as np
import kagglehub
from transformers import pipeline
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generate Financial Sentiment Dataset using FinBERT")
    parser.add_argument("--nrows", type=int, default=None, help="Number of rows to read from the dataset")
    args = parser.parse_args()
    
    # 1. Download Raw Dataset
    print(">>> Downloading raw dataset (Analyst Ratings)...")
    path = kagglehub.dataset_download("miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests")
    csv_path = os.path.join(path, "analyst_ratings_processed.csv")
    
    print(f">>> Reading rows from {csv_path}...")
    df = pd.read_csv(csv_path, nrows=args.nrows)
    
    text_col = 'title'
    df = df.dropna(subset=[text_col])

    df = df.drop_duplicates(subset=[text_col])
    print(f">>> Unique headlines found: {len(df)}")

    # 2. Generate Sentiment Scores using FinBERT
    print(">>> Loading FinBERT...")
    pipe = pipeline("text-classification", model="ProsusAI/finbert", return_all_scores=True, device=-1)

    print(">>> Generating Sentiment Scores...")
    
    titles = df[text_col].tolist()
    scores = []
    
    batch_size = 32
    
    for i in tqdm(range(0, len(titles), batch_size)):
        batch_texts = titles[i : i + batch_size]
        try:
            results = pipe(batch_texts, truncation=True, max_length=512)
            
            for res in results:
                pos_score = next(item['score'] for item in res if item['label'] == 'positive')
                neg_score = next(item['score'] for item in res if item['label'] == 'negative')
                
                final_score = pos_score - neg_score
                scores.append(final_score)
                
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            scores.extend([0.0] * len(batch_texts))

    # 3. Save Dataset
    df['score'] = scores
    final_df = df[[text_col, 'score']].rename(columns={text_col: 'sentence'})
    
    output_file = "data/large_financial_sentiment.csv"
    os.makedirs("data", exist_ok=True)
    final_df.to_csv(output_file, index=False)
    
    print("="*50)
    print(f"SUCCESS! Dataset saved to {output_file}")
    print(f"Total Rows: {len(final_df)}")
    print("Example Data:")
    print(final_df.head())
    print("="*50)

if __name__ == "__main__":
    main()