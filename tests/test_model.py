"""
Unit Tests for Model Architecture
---------------------------------
Verifies the LSTM architecture construction, output shapes, and activation functions.
"""
import pytest
import sys
import os
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import build_model

class TestModelArchitecture:
    
    @pytest.fixture
    def model(self):
        model = build_model(max_words=100, embedding_dim=10, lstm_units=16)
        model.build(input_shape=(None, 100))
        
        return model

    def test_model_output_shape(self, model):
        """
        The model should output a single scalar (regression score).
        Shape expected: (None, 1)
        """
        output_shape = model.output_shape
        # output_shape é geralmente (None, 1)
        assert output_shape[1] == 1

    def test_output_activation(self, model):
        """
        The output layer must use 'tanh' because sentiment scores
        range from -1 (Negative) to 1 (Positive).
        """
        output_layer = model.layers[-1]
        assert output_layer.activation.__name__ == 'tanh', \
            "Output activation must be 'tanh' to support range [-1, 1]"

    def test_loss_function(self, model):
        """
        Since this is regression, we expect MSE or MAE, not CrossEntropy.
        """
        assert model.loss == 'mean_squared_error'

    def test_compile_status(self, model):
        """Ensure model is compiled and optimizer is set."""
        assert model.optimizer is not None
        assert model.built is True