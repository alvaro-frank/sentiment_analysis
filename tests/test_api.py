# ==============================================================================
# FILE: tests/test_api.py
# DESCRIPTION: Integration Tests for the Sentiment Analysis API.
#              Verifies endpoint availability, response formats, and input handling
#              using FastAPI TestClient.
# ==============================================================================
import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_api_root():
    """
    Checks if the API is online and reachable.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_api_prediction_format():
    """
    Checks if the prediction endpoint returns the correct JSON format.
    Ensures that 'sentiment_score' and 'label' are present in the response.
    """
    payload = {"text": "The market is performing exceptionally well today."}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "sentiment_score" in data
    assert "label" in data
    assert data["text"] == payload["text"]

def test_api_empty_input():
    """
    Checks the API behavior when receiving empty input text.
    """
    response = client.post("/predict", json={"text": ""})

    assert response.status_code == 200