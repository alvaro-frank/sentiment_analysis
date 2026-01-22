# ==============================================================================
# FILE: src/api.py
# DESCRIPTION: FastAPI application to serve the Sentiment Analysis model.
#              Exposes endpoints for health checks and real-time sentiment prediction.
# ==============================================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from src.data_utils import load_tokenizer, texts_to_padded
from src.predict import get_label_from_score
import os

app = FastAPI(title="Financial Sentiment API")

MODEL_PATH = "models/sentiment_model.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Modelo não encontrado. Execute o treino primeiro.")

model = load_model(MODEL_PATH)
tokenizer = load_tokenizer(TOKENIZER_PATH)

class SentimentRequest(BaseModel):
    """
    Request schema for sentiment prediction.
    Args:
        text (str): The financial text to be analyzed.
    """
    text: str

@app.get("/")
def read_root():
    """
    Health check endpoint.
    Returns:
        dict: Status of the API and model name.
    """
    return {"status": "online", "model": "Bi-LSTM Financial Sentiment"}

@app.post("/predict")
def predict_sentiment(request: SentimentRequest):
    """
    Inference endpoint.
    Receives a text, runs preprocessing, and returns the sentiment score/label.
    
    Args:
        request (SentimentRequest): JSON body containing the text.
        
    Returns:
        dict: The original text, the regression score, and the classification label.
    """
    try:
        X_new = texts_to_padded(tokenizer, [request.text])
        score = float(model.predict(X_new)[0][0])
        label = get_label_from_score(score)
        
        return {
            "text": request.text,
            "sentiment_score": round(score, 4),
            "label": label
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))