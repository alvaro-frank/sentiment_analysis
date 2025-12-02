# Sentiment Analysis

A research project for Sentiment Analysis using a Bi-LSTM Regression model.

It includes data preprocessing with VADER, a custom Keras regression model to predict sentiment scores (-1 to 1), and MLflow integration for experiment tracking.

![Sentiment Scores](images/sentiment_scores.png)

## Data & Features
- **Data Pipeline**: Downloads and processes financial news data from Kaggle (src/data_utils.py).
- **Regression Model**: A Bi-LSTM neural network that predicts a continuous sentiment score (src/model.py).
- **Ground Truth Generation**: Uses VADER to generate target sentiment scores for training (src/data_utils.py).
- **MLflow Integration**: Tracks parameters, metrics (MAE, MSE, R2), and artifacts (models, plots) automatically (src/train.py, src/evaluate.py).
- **Evaluation**: Visualizes model performance against VADER scores with correlation plots (src/evaluate.py).

## Model Input & Output
The model processes text data to predict a Sentiment Score:
- **Input**: Raw text (news headlines/titles).
- **Preprocessing**: Tokenization, stopword removal, stemming/lemmatization.
- **Output**: A continuous float value between -1.0 (Negative) and 1.0 (Positive).

## Project Structure
```
./
  sentiment_analysis/      
    models/                 # Saved artifacts (model.h5, tokenizer.pkl, plots)
    src/
      data_utils.py         # Text preprocessing & VADER scoring
      model.py              # Text preprocessing & VADER scoring
      predict.py            # Inference script for new text
      train.py              # Evaluation against VADER ground truth
    requirements.txt        # Python dependencies
    Makefile                # Automation commands
    README.md
```

## Requirements
```
pip install -r requirements.txt
```

---

## Quick Start
### 🚀 Full Pipeline

To setup, train, predict, and evaluate in one go:

```bash
make all
```

This will:
- Create the virtual environment and installs the required packages from `requirements.txt`.
- Download the dataset and train the regression model.
- Run inference on sample texts.
- Evaluate the model against the test set and generate plots.

You can also customize the run using arguments (see below).
### 🐍 Virtual Environment
This section explains how to create and activate the virtual environment and installs the required packages from `requirements.txt`, just use the command line:

```bash
make setup
```

### 🧠 Training
Train the regression model on the financial news dataset:

```bash
make train
```

You can override defaults by passing variables on the command line:

| Arg        | Purpose                                   | Default | Examples |
|------------|-------------------------------------------|---------|----------|
| `EPOCHS`    | Number of training epochs     | `5`   | `EPOCHS=10` |
| `BATCH_SIZE` | Batch size for training               | `32`   | `BATCH_SIZE=64` |
| `MAX_WORDS`    | Vocabulary size              | `5000`    | `MAX_WORDS=10000` |
| `NROWS`     | Number of rows to read from CSV                              | `1000`    | `NROWS=5000` |

Example:
```bash
# Train for 10 epochs with 50000 rows of news.
make train EPOCHS=10 NROWS=50000
```

Artifacts are stored locally in models/ and logged to MLflow.
### 🔮 Prediction
Predict the sentiment score of specific text(s):

```bash
make predict
```

You can pass your own text string (use quotes):
```bash
make predict TEXT="Inflation is rising faster than expected"
```
### 📊 Evaluation
Evaluate the trained model against VADER scores (Ground Truth):

```bash
make evaluate
```

You can target a specific checkpoint and adjust evaluation settings:

| Arg        | Purpose                                            | Default         | Examples |
|------------|----------------------------------------------------|-----------------|----------|
| `EVAL_ROWS`    | Number of rows to use for evaluation               | `100`           | `EVAL_ROWS=500` |

Examples:
```bash
# Compare the model scores with VADER scores on 500 news.
make evaluate EVAL_ROWS=500
```

This generates an evaluation report (MAE, MSE, R2) and a correlation plot at `models/evaluation_plot.png`.

### 📈 Experiment Tracking (MLflow)
This project uses **MLflow** to track training performance and version models.

**How to Launch the Dashboard**

To view training curves and logged artifacts, run the following command:
```bash
make mlflow PORT=5000
```
This will start the MLflow server at **http://127.0.0.1:5000** by default.

What is Logged?
Every time you run `make train` or `make evaluate`, a new experiment run is created logging:

**Params**:
- `batch_size`: Batch size used for training.
- `epochs`: Number of training epochs.
- `max_words`: Vocabulary size limit.
- `nrows`: Number of rows used for training.
- `eval_rows`: Number of rows used for evaluation.
- `data_source`: Dataset identifier.
- `opt_name`: Optimizer used.
- `opt_learning_rate`: Learning rate.

**Metrics**:
- `mae` (Mean Absolute Error): Average error between model prediction and VADER score.: % of bin volume filled.
- `val_mae`: Validation MAE.
- `mse` & `r2_score`: Additional regression metrics (logged during evaluation).

**Artifacts**:
- Model Checkpoints: The final trained model is saved as an MLflow artifact.

---
