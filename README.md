# Sentiment Analysis

A production-grade NLP project that implements Knowledge Distillation to create a lightweight, high-performance sentiment analysis model for financial text.

This project demonstrates how to compress a heavy Transformer model (**FinBERT**) into a fast, deployable **Bi-LSTM** without sacrificing significant accuracy. It features a robust **MLOps** pipeline with data versioning, experiment tracking, and automated testing.

<p align="center">
  <img src="images/sentiment_scores.png" alt="Sentiment Scores" width="400">
</p>

## 🚀 Features
- **Knowledge Distillation**: Uses FinBERT to generate high-quality labels for a lightweight Bi-LSTM.
- **Domain-Specific Preprocessing**: Custom cleaning pipeline handling financial entities (numbers, percentages) and preserving crucial negations (e.g., "not", "no").
- **Resumable Training**: Logic to seamlessly pause and resume model training from checkpoints.
- **MLOps Infrastructure**: Version control for large datasets (DVC) and experiment tracking for metrics (MAE, RMSE and Loss) and artifacts.
- **Inference Engine**: Fast prediction CLI suitable for real-time applications.

## 📂 Project Structure
```
├── .dvc/                    # DVC Configuration
├── data/                    # Data managed by DVC
├── models/                  # Saved models and tokenizer
├── src/
│   ├── data_utils.py        # Text preprocessing & Tokenization logic
│   ├── generate_dataset.py  # Data generation logic
│   ├── model.py             # Model architecture
│   ├── train.py             # Training loop with MLflow
│   ├── predict.py           # Inference CLI
│   └── evaluate.py          # Evaluation scripts
├── tests/                   # Unit tests
├── Makefile                 # Command automation
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## 🛠️ Setup & Requirements

This project uses `make` for automation and `dvc` for data management.

1. **Clone the repository**
```bash
git clone https://github.com/alvaro-frank/sentiment_analysis.git
cd sentiment_analysis
```

2. **Setup Environment**: This command creates a virtual environment, installs dependencies, and pulls data via DVC.
```bash
make setup
```

## ⚡ Quick Start

To run the **full end-to-end pipeline** (Clean -> Setup -> Unit Tests -> Train -> Evaluate -> Predict) in one go:
```bash
make all
```

## 🏃 Usage

You can run individual steps using the `Makefile` shortcuts.

1. **Data Generation**

If you want to regenerate the dataset using the **FinBERT** model:
```bash
# Generate dataset with all rows
make generate-data

# Generate data with a specific number of rows
make generate-data NROWS=10000
```
_Note: Requires internet access to download Kaggle data and HuggingFace models._

2. **Training**

Train the **Bi-LSTM**. Metrics are logged to **MLflow**.

You can override defaults by passing variables on the command line:

| Arg        | Purpose                                   | Default | Examples |
|------------|-------------------------------------------|---------|----------|
| `EPOCHS`    | Number of training epochs     | `5`   | `EPOCHS=10` |
| `BATCH_SIZE` | Batch size for training               | `32`   | `BATCH_SIZE=64` |
| `MAX_WORDS`    | Vocabulary size              | `5000`    | `MAX_WORDS=10000` |
| `RESUME`     | Resume training indicator                              | `False`    | `RESUME=True` |

**Standard Training**
```bash
# Default training
make train

#Train with specific params and vocabulary size
make train EPOCHS=20 BATCH_SIZE=64 MAX_WORDS=5000
```

**Resume Training**: If training was interrupted or you want to continue optimizing an existing checkpoint:
```bash
make train RESUME=True
```

3. **Evaluation**
Evaluate the model against the generated dataset. This calculates R² score, MAE, and generates prediction plots.

| Arg        | Purpose                                            | Default         | Examples |
|------------|----------------------------------------------------|-----------------|----------|
| `EVAL_ROWS`    | Number of rows to use for evaluation               | `100`           | `EVAL_ROWS=500` |

```bash
# Evaluate the model with default number of rows
make evaluate

#Evaluate the model with a specific number of rows
make evaluate EVAL_ROWS=500
```
_Outputs metrics to console and saves evaluation_plot.png to models/.._

3. **Prediction**

Run inference on a custom sentence to test the model.
```bash
# Output: Sentiment Score: e.g., -0.85 (Negative)
make predict TEXT='Revenue dropped by 10% due to poor market conditions'
```

4. **Unit Testing**

Ensure preprocessing logic (negation handling, tokenization) and model architecture are valid.
```bash
make unit-test
```

5. **Experiment Tracking**

Launch the MLflow dashboard to visualize the model metrics and learning curves.
```bash
make mlflow
```

## 🧠 Methodology

**The Teacher-Student Approach**

Instead of training a massive BERT model (slow inference), we use a "Knowledge Distillation" strategy:

1. **The teacher (FinBERT)**: A pre-trained Transformer model specialized in finance processes thousands of headlines to generate continuous sentiment scores (-1 to 1).
2. **The student (Bi-LSTM)**: A lightweight Recurrent Neural Network learns to regress these scores. It is significantly faster and smaller, making it ideal for production.

**Robust Preprocessing**

Financial text requires careful handling. Our pipeline in `src/data_utils.py`:

- **Number Normalization**: Converts `10%`, `5.5M` -> `<NUM>` to reduce vocabulary sparsity.
- **Smart Stopwords**: Removes noise but **preserves negations** (e.g., "not", "won't") which flip sentiment polarity.
- **Contraction Expansion**: Expands `can't` -> `cannot` for better tokenization.
