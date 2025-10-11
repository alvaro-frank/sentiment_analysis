# Sentiment Analysis

A simple research project for **Sentiment Analysis**.  
It includes data preprocessing, baseline ML models, optional neural/transformer models, and evaluation/visualization utilities.

The code was **originally developed in a Jupyter notebook and later adapted into a structured repository**.

![Sentiment Scores](images/sentiment_scores.png)

## Features
- Classical ML baselines with scikit-learn (e.g., Logistic Regression, SVM).

## Data & Features
The pipeline expects labeled text for **sentiment classification** (e.g., `positive`, `negative`, `neutral`). Typical inputs:
- **Raw text:** user reviews, tweets, or comments.
- **Labels:** one of the supported classes.
- **Features:** token IDs, attention masks (for transformers), TF‑IDF vectors or embeddings depending on the selected model.

## Evaluation Metrics
We report standard classification metrics to assess model quality:
- **Accuracy** for overall correctness.
- **Precision, Recall, F1** per class and macro/weighted averages.
- **Confusion Matrix** to visualize class-wise performance.
- **ROC‑AUC** when applicable (binary).


## Project Structure
```
./
  sentiment_analysis/
    src/
      data_utils.py
      model.py
      predict.py
      requirements.txt
      train.py
    README.md
```

## Requirements
```
pip install -r requirements.txt
```

## Quick Start
Run the end‑to‑end script (train → evaluate → report):
```bash
python sentiment_analysis/src/train.py
```
This will train a sentiment classifier, evaluate it on the validation/test set, and write artifacts to `models/`, `reports/`, and `logs/`.

### Outputs
- `models/` — saved checkpoints (best model, tokenizer, config).
- `reports/` — evaluation reports and plots.
- `logs/` — training logs (loss/metrics over time).

## Configuration
You can tweak key settings in the code or config files (if present):
- **Data paths** and preprocessing options.
- **Model** selection (TF‑IDF + Linear, LSTM/CNN, or Transformer).
- **Training** hyperparameters (epochs, batch size, learning rate).
- **Evaluation** options (validation split, metrics).

## Repro Tips
- Fix random seeds (NumPy/torch/TensorFlow) for reproducibility.
- Pin package versions (see `requirements.txt`).
- Use the same train/validation splits when comparing models.
