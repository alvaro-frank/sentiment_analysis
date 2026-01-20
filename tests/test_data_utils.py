"""
Unit Tests for Data Utils
-------------------------
Verifies text cleaning, tokenization, and negation preservation logic.
"""
import pytest
import sys
import os
import numpy as np

# Adicionar src ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_utils import preprocess_text

class TestPreprocessText:
    
    def test_basic_cleaning(self):
        """Test lowercase and basic punctuation removal."""
        text = "Hello World!"
        # "Hello" -> "hello", "World" -> "world", "!" removed
        assert "hello" in preprocess_text(text)
        assert "world" in preprocess_text(text)

    def test_number_normalization(self):
        """Test if numbers/percentages become <NUM>."""
        text = "Revenue grew by 10% and profit 5.5"
        processed = preprocess_text(text)
        assert "NUM" in processed
        assert "10" not in processed
        assert "5.5" not in processed

    def test_negation_preservation(self):
        """
        CRITICAL: Standard stopword removal often kills 'not'.
        We must ensure 'not', 'no', 'never' are kept.
        """
        text = "The profit is not good and implies no growth."
        processed = preprocess_text(text)
        
        assert "is" not in processed.split()
        assert "the" not in processed.split() 
        assert "not" in processed.split()
        assert "no" in processed.split()

    def test_contraction_expansion(self):
        """Test if 'won't', 'can't', etc. are expanded correctly."""
        text = "We won't accept this deal."
        processed = preprocess_text(text)
        assert "not" in processed

    def test_empty_input(self):
        """Test resilience against empty or non-string inputs."""
        assert preprocess_text("") == ""
        assert preprocess_text(None) == ""